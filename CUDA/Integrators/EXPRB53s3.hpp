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
    void EXPRB53s3(rhs& RHS,                   //? RHS function
                   double* u,                  //? Input state variable(s)
                   double* u_exprb3,           //? Output state variable(s) (lower order)
                   double* u_exprb5,           //? Output state variable(s) (higher order)
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

        //! u, u_exprb3, u_exprb5, auxiliary_expint, auxiliary_Leja, and auxiliary_NL
        //! are device vectors if GPU support is activated.

        //*
        //*    Returns
        //*    ----------
        //*     u_exprb3                : double*
        //*                                 3rd order solution after time dt
        //*     
        //*     u_exprb5                : double* 
        //*                                 5th order solution after time dt
        //*
        //*
        //*    Reference:
        //*         V. T. Luan, A. Ostermann, Exponential Rosenbrock methods of order five — construction, analysis and numerical comparisons, J. Comput. Appl. Math. 255 (2014) 417-431. 
        //*         doi:10.1016/j.cam.2013.04.041

        //* -------------------------------------------------------------------------

        //? Counters for Leja iterations
        int iters_1 = 0, iters_2 = 0, iters_3 = 0, iters_4 = 0, iters_5 = 0;

        //? Assign required variables
        double* u_flux = &auxiliary_expint[0];
        double* b_nl = &auxiliary_expint[3*N];
        double* a = &auxiliary_expint[5*N];

        //? RHS evaluated at 'u' multiplied by 'dt'; temp_vec_1 = f(u)
        RHS(u, u_exprb3);
        axpby(dt, u_exprb3, u_exprb3, N, GPU);

        //? Vertical interpolation of RHS(u) at 1/2, 9/10, and 1
        real_Leja_phi(RHS, u, u_exprb3, u_flux, auxiliary_Leja, N, {0.5, 0.9, 1.0}, 
                        phi_1, Leja_X, c, Gamma, tol, dt, iters_1, GPU, cublas_handle);

        //? Internal stage 1; a = u + 1/2 phi_1(1/2 J(u) dt) f(u) dt
        axpby(1.0, u, 0.5, &u_flux[0], a, N, GPU);

        //? Nonlinear remainder at u and a;  R_a = F(a) - F(u)
        Nonlinear_remainder(RHS, u, u, u_exprb3, auxiliary_Leja, N, GPU, cublas_handle);
        Nonlinear_remainder(RHS, u, a, u_exprb5, auxiliary_Leja, N, GPU, cublas_handle);
        axpby(dt, u_exprb5, -dt, u_exprb3, a, N, GPU);

        //? Vertical interpolation of R(a) at 1/2 and 9/10
        real_Leja_phi(RHS, u, a, b_nl, auxiliary_Leja, N, {0.5, 0.9}, 
                        phi_3, Leja_X, c, Gamma, tol, dt, iters_2, GPU, cublas_handle);

        //? Internal stage 2; b = u + 9/10 phi_1(9/10 J(u) dt) f(u) dt + (27/25 phi_3(1/2 J(u) dt) + 729/125 phi_3(9/10 J(u) dt)) R(a) dt
        double* b = &u_flux[0];
        axpby(1.0, u, 0.9, &u_flux[N], 27.0/25.0, &b_nl[0], 729.0/125.0, &b_nl[N], b, N, GPU);

        //? R_b = F(b) - F(u)
        Nonlinear_remainder(RHS, u, b, u_exprb5, auxiliary_Leja, N, GPU, cublas_handle);
        axpby(dt, u_exprb5, -dt, u_exprb3, b, N, GPU);

        //? Final nonlinear stages
        double* R_3 = &b_nl[0];
        double* R_4 = &b_nl[N];

        //? a = phi_3(J(u) dt) (2R(a) + 150/81R(b)) dt
        axpby(2.0, a, 150.0/81.0, b, R_3, N, GPU);
        real_Leja_phi(RHS, u, R_3, a, auxiliary_Leja, N, {1.0},
                        phi_3, Leja_X, c, Gamma, tol, dt, iters_3, GPU, cublas_handle);

        //! 3rd order solution; u_3 = u + phi_1(J(u) dt) f(u) dt + phi_3(J(u) dt) (2R(a) + (150/81)R(b)) dt
        axpby(1.0, u, 1.0, &u_flux[2*N], 1.0, a, u_exprb3, N, GPU);

        //? a = phi_3(J(u) dt) (18R(a) - 250/81R(b)) dt
        axpby(18.0, a, -250.0/81.0, b, R_3, N, GPU);
        real_Leja_phi(RHS, u, R_3, a, auxiliary_Leja, N, {1.0},
                        phi_3, Leja_X, c, Gamma, tol, dt, iters_4, GPU, cublas_handle);

        //? b = phi_4(J(u) dt) (-60R(a) + (500/27)R(b)) dt
        axpby(-60.0, a, 500.0/27.0, b, R_4, N, GPU);
        real_Leja_phi(RHS, u, R_4, b, auxiliary_Leja, N, {1.0},
                        phi_4, Leja_X, c, Gamma, tol, dt, iters_5, GPU, cublas_handle);

        //! 5th order solution; u_5 = u + phi_1(J(u) dt) f(u) dt + phi_3(J(u) dt) (18R(a) - (250/81)R(b)) dt + phi_4(J(u) dt) (-60R(a) + (500/27)R(b)) dt
        axpby(1.0, u, 1.0, &u_flux[2*N], 1.0, a, 1.0, b, u_exprb5, N, GPU);

        //? Total number of Leja iterations
        iters = iters_1 + iters_2 + iters_3 + iters_4 + iters_5;
    }
}