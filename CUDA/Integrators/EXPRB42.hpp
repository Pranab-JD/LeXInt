#pragma once

#include <iostream>
#include <iomanip>
#include <vector>
#include <cmath>
#include <functional>

#include "../Phi_functions.hpp"
#include "../real_Leja_phi.hpp"
#include "../Timer.hpp"

//? CUDA 
#include "../error_check.hpp"
#include "../Leja_GPU.hpp"
#include "../Kernels_CUDA_Cpp.hpp"

using namespace std;

//? Phi functions interpolated on real Leja points
template <typename rhs>
void EXPRB42(rhs& RHS,                   //? RHS function
             double* u,                  //? Input state variable(s)
             double* u_exprb2,           //? Output state variable(s) (lower order)
             double* u_exprb4,           //? Output state variable(s) (higher order)
             double* auxillary_expint,   //? Internal auxillary variables (EXPRB42)
             double* auxillary_Leja,     //? Internal auxillary variables (Leja)
             double* auxillary_NL,       //? Internal auxillary variables (EXPRB42 NL)
             size_t N,                   //? Number of grid points
             vector<double>& Leja_X,     //? Array of Leja points
             double c,                   //? Shifting factor
             double Gamma,               //? Scaling factor
             double tol,                 //? Tolerance (normalised desired accuracy)
             double dt,                  //? Step size
             bool GPU,                   //? false (0) --> CPU; true (1) --> GPU
             GPU_handle& cublas_handle   //? CuBLAS handle
             )
{
    //* -------------------------------------------------------------------------

    //! u, u_exprb2, u_exprb4, auxillary_expint, auxillary_Leja, and auxillary_NL
    //! are device vectors if GPU support is activated.

    //*
    //*    Returns
    //*    ----------
    //*     u_exprb2                : state
    //*                                 2nd order solution after time dt
    //*     
    //*     u_exprb4                : state 
    //*                                 4th order solution after time dt
    //*
    //*
    //*    Reference:
    //*         V. T. Luan, Fourth-order two-stage explicit exponential integrators for time-dependent PDEs, Appl. Numer. Math. 112 (2017) 91-103. 
    //*         doi:10.1016/j.apnum.2016.10.008

    //* ------------------------------------------------------------------------- 

    //? RHS evaluated at 'u' multiplied by 'dt'
    double* rhs_u = &auxillary_expint[0];
    RHS(u, rhs_u);
    axpby(dt, rhs_u, rhs_u, N, GPU);

    //? Vertical interpolation of RHS(u) at 3/4 and 1; phi_1({3/4, 1.0} J(u) dt) f(u) dt
    double* u_flux = &auxillary_expint[N];
    real_Leja_phi(RHS, u, rhs_u, u_flux, auxillary_Leja, N, {3./4., 1.0}, 
                    phi_1, Leja_X, c, Gamma, tol, dt, GPU, cublas_handle);

    //? Internal stage 1; a = u + 3/4 phi_1(3/4 J(u) dt) f(u) dt
    double* a = &auxillary_expint[3*N];
    axpby(1.0, u, 3./4., &u_flux[0], a, N, GPU);

    //? Assign memory for nonlinear remainders
    double* NL_u = &auxillary_expint[4*N];
    double* NL_a = &auxillary_expint[5*N];
    double* R_a = &auxillary_expint[6*N];

    //? Difference of nonlinear remainders at a
    Nonlinear_remainder(RHS, u, u, NL_u, auxillary_NL, N, GPU, cublas_handle);
    Nonlinear_remainder(RHS, u, a, NL_a, auxillary_NL, N, GPU, cublas_handle);
    axpby(dt, NL_a, -dt, NL_u, R_a, N, GPU);

    //? phi_3(J(u) dt) R(a) dt
    double* u_nl_3 = &auxillary_expint[7*N];
    real_Leja_phi(RHS, u, R_a, u_nl_3, auxillary_Leja, N, {1.0}, 
                    phi_3, Leja_X, c, Gamma, tol, dt, GPU, cublas_handle);

    //? 2nd order solution; u_2 = u + phi_1(J(u) dt) f(u) dt
    axpby(1.0, u, 1.0, &u_flux[N], u_exprb2, N, GPU);

    //? 4th order solution; u_4 = u_2 + 32/9 phi_3(J(u) dt) R(a) dt
    axpby(1.0, u_exprb2, 32./9., u_nl_3, u_exprb4, N, GPU);
}