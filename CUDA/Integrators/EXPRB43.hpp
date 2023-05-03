#pragma once

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
void EXPRB43(rhs& RHS,                   //? RHS function
             double* u,                  //? Input state variable(s)
             double* u_exprb3,           //? Output state variable(s) (lower order)
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

    //! u, u_exprb3, u_exprb4, auxillary_expint, auxillary_Leja, and auxillary_NL
    //! are device vectors if GPU support is activated.

    //*    Returns
    //*    ----------
    //*     u_exprb3                : double*
    //*                                 3rd order solution after time dt
    //*     
    //*     u_exprb4                : double* 
    //*                                 4th order solution after time dt
    //*
    //*
    //*    Reference:
    //*         M. Hochbruck, A. Ostermann, Exponential Integrators, Acta Numer. 19 (2010) 209-286. 
    //*         doi:10.1017/S0962492910000048

    //* -------------------------------------------------------------------------

    //? RHS evaluated at 'u' multiplied by 'dt'
    double* rhs_u = &auxillary_expint[0];
    RHS(u, rhs_u);
    axpby(dt, rhs_u, rhs_u, N, GPU);

    //? Vertical interpolation of RHS(u) at 0.5 and 1; phi_1({0.5, 1.0} J(u) dt) f(u) dt
    double* u_flux = &auxillary_expint[N];
    real_Leja_phi(RHS, u, rhs_u, u_flux, auxillary_Leja, N, {0.5, 1.0}, 
                    phi_1, Leja_X, c, Gamma, tol, dt, GPU, cublas_handle);

    //? Internal stage 1; a = u + 1/2 phi_1(1/2 J(u) dt) f(u) dt
    double* a = &auxillary_expint[3*N];
    axpby(1.0, u, 0.5, &u_flux[0], a, N, GPU);

    //? Assign memory for nonlinear remainders
    double* NL_u = &auxillary_expint[4*N];
    double* NL_a = &auxillary_expint[5*N];
    double* NL_b = &auxillary_expint[6*N];
    double* R_a = &auxillary_expint[7*N];
    double* R_b = &auxillary_expint[8*N];

    //? Difference of nonlinear remainders at a
    Nonlinear_remainder(RHS, u, u, NL_u, auxillary_NL, N, GPU, cublas_handle);
    Nonlinear_remainder(RHS, u, a, NL_a, auxillary_NL, N, GPU, cublas_handle);
    axpby(dt, NL_a, -dt, NL_u, R_a, N, GPU);

    //? Internal stage 2; phi_1(J(u) dt) R(a) dt
    double* b_nl = &auxillary_expint[9*N];
    real_Leja_phi(RHS, u, R_a, b_nl, auxillary_Leja, N, {1.0}, 
                    phi_1, Leja_X, c, Gamma, tol, dt, GPU, cublas_handle);

    //? b = u + phi_1(J(u) dt) f(u) dt + phi_1(J(u) dt) R(a) dt
    double* b = &auxillary_expint[10*N];
    axpby(1.0, u, 1.0, &u_flux[N], 1.0, b_nl, b, N, GPU);

    //? Difference of nonlinear remainders at b
    Nonlinear_remainder(RHS, u, b, NL_b, auxillary_NL, N, GPU, cublas_handle);
    axpby(dt, NL_b, -dt, NL_u, R_b, N, GPU);

    //? Final nonlinear stages
    double* R_3 = &auxillary_expint[11*N];
    double* R_4 = &auxillary_expint[12*N];
    axpby( 16.0, R_a, -2.0, R_b, R_3, N, GPU);
    axpby(-48.0, R_a, 12.0, R_b, R_4, N, GPU);

    //? phi_3(J(u) dt) (16R(a) - 2R(b)) dt
    double* u_nl_3 = &auxillary_expint[13*N];
    real_Leja_phi(RHS, u, R_3, u_nl_3, auxillary_Leja, N, {1.0},
                    phi_3, Leja_X, c, Gamma, tol, dt, GPU, cublas_handle);
    
    //? phi_4(J(u) dt) (-48R(a) + 12R(b)) dt
    double* u_nl_4 = &auxillary_expint[14*N];
    real_Leja_phi(RHS, u, R_4, u_nl_4, auxillary_Leja, N, {1.0},
                    phi_4, Leja_X, c, Gamma, tol, dt, GPU, cublas_handle);

    //? 3rd order solution; u_3 = u + phi_1(J(u) dt) f(u) dt + phi_3(J(u) dt) (16R(a) - 2R(b)) dt
    axpby(1.0, u, 1.0, &u_flux[N], 1.0, u_nl_3, u_exprb3, N, GPU);

    //? 4th order solution; u_4 = u_3 + phi_4(J(u) dt) (-48R(a) + 12R(b)) dt
    axpby(1.0, u_exprb3, 1.0, u_nl_4, u_exprb4, N, GPU);
}