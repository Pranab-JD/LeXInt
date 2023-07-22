#pragma once

#include "../Phi_functions.hpp"
#include "../real_Leja_phi.hpp"
#include "../Timer.hpp"

//? CUDA 
#include "../error_check.hpp"
#include "../Leja_GPU.hpp"
#include "../Kernels_CUDA_Cpp.hpp"

namespace LeXInt
{
    //? Phi functions interpolated on real Leja points
    template <typename rhs>
    void EXPRB42(rhs& RHS,                   //? RHS function
                 double* u,                  //? Input state variable(s)
                 double* u_exprb2,           //? Output state variable(s) (lower order)
                 double* u_exprb4,           //? Output state variable(s) (higher order)
                 double* auxiliary_expint,   //? Internal auxiliary variables (EXPRB42)
                 double* auxiliary_Leja,     //? Internal auxiliary variables (Leja)
                 size_t N,                   //? Number of grid points
                 vector<double>& Leja_X,     //? Array of Leja points
                 double c,                   //? Shifting factor
                 double Gamma,               //? Scaling factor
                 double tol,                 //? Tolerance (normalised desired accuracy)
                 double dt,                  //? Step size
                 int& iters,                 //? # of iterations needed to converge (iteration variable)
                 bool GPU,                   //? false (0) --> CPU; true (1) --> GPU
                 GPU_handle& cublas_handle   //? CuBLAS handle
                 )
    {
        //* -------------------------------------------------------------------------

        //! u, u_exprb2, u_exprb4, auxiliary_expint, auxiliary_Leja, and auxiliary_NL
        //! are device vectors if GPU support is activated.

        //*
        //*    Returns
        //*    ----------
        //*     u_exprb2                : double*
        //*                                 2nd order solution after time dt
        //*     
        //*     u_exprb4                : double* 
        //*                                 4th order solution after time dt
        //*
        //*
        //*    Reference:
        //*         V. T. Luan, Fourth-order two-stage explicit exponential integrators for time-dependent PDEs, Appl. Numer. Math. 112 (2017) 91-103. 
        //*         doi:10.1016/j.apnum.2016.10.008

        //* ------------------------------------------------------------------------- 

        //? Counters for Leja iterations
        int iters_1 = 0, iters_2 = 0;

        //? Assign required variables
        double* u_flux = &auxiliary_expint[0];

        //? RHS evaluated at 'u' multiplied by 'dt'; u_exprb2 = f(u)*dt
        RHS(u, u_exprb2);
        axpby(dt, u_exprb2, u_exprb2, N, GPU);

        //? Vertical interpolation of RHS(u) at 3/4 and 1; u_flux = phi_1({3/4, 1.0} J(u) dt) f(u) dt
        real_Leja_phi(RHS, u, u_exprb2, u_flux, auxiliary_Leja, N, {3./4., 1.0}, 
                        phi_1, Leja_X, c, Gamma, tol, dt, iters_1, GPU, cublas_handle);

        //? Internal stage 1; u_exprb2 = u + 3/4 phi_1(3/4 J(u) dt) f(u) dt
        axpby(1.0, u, 3./4., &u_flux[0], u_exprb2, N, GPU);

        //? u_exprb2 = (u_exprb4 - temp_vec)*dt
        double* NL = &u_flux[0];
        Nonlinear_remainder(RHS, u, u,        u_exprb4, auxiliary_Leja, N, GPU, cublas_handle);
        Nonlinear_remainder(RHS, u, u_exprb2, NL,       auxiliary_Leja, N, GPU, cublas_handle);
        axpby(dt, NL, -dt, u_exprb4, u_exprb2, N, GPU);

        //? NL_2 = phi_3(J(u) dt) u_exprb2 dt
        real_Leja_phi(RHS, u, u_exprb2, NL, auxiliary_Leja, N, {1.0}, 
                        phi_3, Leja_X, c, Gamma, tol, dt, iters_2, GPU, cublas_handle);

        //? 2nd order solution; u_2 = u + phi_1(J(u) dt) f(u) dt
        axpby(1.0, u, 1.0, &u_flux[N], u_exprb2, N, GPU);

        //? 4th order solution; u_4 = u_2 + 32/9 phi_3(J(u) dt) R(a) dt
        axpby(1.0, u_exprb2, 32./9., NL, u_exprb4, N, GPU);

        //? Total number of Leja iterations
        iters = iters_1 + iters_2;
    }
}