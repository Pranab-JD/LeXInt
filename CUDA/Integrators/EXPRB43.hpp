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
    void EXPRB43(rhs& RHS,                   //? RHS function
                 double* u,                  //? Input state variable(s)
                 double* u_exprb3,           //? Output state variable(s) (lower order)
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

        //! u, u_exprb3, u_exprb4, auxiliary_expint, auxiliary_Leja, and auxiliary_NL
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

        //? Counters for Leja iterations
        int iters_1 = 0, iters_2 = 0, iters_3 = 0, iters_4 = 0;

        //? Assign required variables
        double* u_flux = &auxiliary_expint[0];
        double* a = &auxiliary_expint[2*N];

        //? RHS evaluated at 'u' multiplied by 'dt'; temp_vec_1 = f(u)
        RHS(u, u_exprb3);
        axpby(dt, u_exprb3, u_exprb3, N, GPU);

        //? Vertical interpolation of RHS(u) at 0.5 and 1; phi_1({0.5, 1.0} J(u) dt) f(u) dt
        real_Leja_phi(RHS, u, u_exprb3, u_flux, auxiliary_Leja, N, {0.5, 1.0}, 
                        phi_1, Leja_X, c, Gamma, tol, dt, iters_1, GPU, cublas_handle);

        //? Internal stage 1; a = u + 1/2 phi_1(1/2 J(u) dt) f(u) dt
        axpby(1.0, u, 0.5, &u_flux[0], a, N, GPU);

        //? Difference of nonlinear remainders at a
        Nonlinear_remainder(RHS, u, u, u_exprb3, auxiliary_Leja, N, GPU, cublas_handle);
        Nonlinear_remainder(RHS, u, a, u_exprb4, auxiliary_Leja, N, GPU, cublas_handle);
        axpby(dt, u_exprb4, -dt, u_exprb3, a, N, GPU);

        //? Internal stage 2; phi_1(J(u) dt) R(a) dt
        real_Leja_phi(RHS, u, a, u_exprb4, auxiliary_Leja, N, {1.0}, 
                        phi_1, Leja_X, c, Gamma, tol, dt, iters_2, GPU, cublas_handle);

        //? b = u + phi_1(J(u) dt) f(u) dt + phi_1(J(u) dt) R(a) dt
        double* b = &u_flux[0];
        axpby(1.0, u, 1.0, &u_flux[N], 1.0, u_exprb4, b, N, GPU);

        //? Difference of nonlinear remainders at b
        Nonlinear_remainder(RHS, u, b, u_exprb4, auxiliary_Leja, N, GPU, cublas_handle);
        axpby(dt, u_exprb4, -dt, u_exprb3, b, N, GPU);

        //? Final nonlinear stages
        axpby( 16.0, a, -2.0, b, u_exprb3, N, GPU);
        axpby(-48.0, a, 12.0, b, u_exprb4, N, GPU);

        //? phi_3(J(u) dt) (16R(a) - 2R(b)) dt
        real_Leja_phi(RHS, u, u_exprb3, a, auxiliary_Leja, N, {1.0},
                        phi_3, Leja_X, c, Gamma, tol, dt, iters_3, GPU, cublas_handle);
        
        //? phi_4(J(u) dt) (-48R(a) + 12R(b)) dt
        real_Leja_phi(RHS, u, u_exprb4, b, auxiliary_Leja, N, {1.0},
                        phi_4, Leja_X, c, Gamma, tol, dt, iters_4, GPU, cublas_handle);

        //! 3rd order solution; u_3 = u + phi_1(J(u) dt) f(u) dt + phi_3(J(u) dt) (16R(a) - 2R(b)) dt
        axpby(1.0, u, 1.0, &u_flux[N], 1.0, a, u_exprb3, N, GPU);

        //! 4th order solution; u_4 = u_3 + phi_4(J(u) dt) (-48R(a) + 12R(b)) dt
        axpby(1.0, u_exprb3, 1.0, b, u_exprb4, N, GPU);

        //? Total number of Leja iterations
        iters = iters_1 + iters_2 + iters_3 + iters_4;
    }
}