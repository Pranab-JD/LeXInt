#pragma once

#include "../Leja.hpp"
#include "../Phi_functions.hpp"

namespace LeXInt
{
    //? Phi functions interpolated on real Leja points
    template <typename rhs>
    void EXPRB54s4(rhs& RHS,                        //? RHS function
                   double* u,                       //? Input state variable(s)
                   double* u_exprb4,                //? Output state variable(s) (lower order)
                   double* u_exprb5,                //? Output state variable(s) (higher order)
                   double& error,                   //? Embedded error estimate
                   double* auxiliary_expint,        //? Internal auxiliary variables (EXPRB54s4)
                   double* auxiliary_Leja,          //? Internal auxiliary variables (Leja)
                   size_t N,                        //? Number of grid points
                   std::vector<double>& Leja_X,     //? Array of Leja points
                   double c,                        //? Shifting factor
                   double Gamma,                    //? Scaling factor
                   double tol,                      //? Tolerance (normalised desired accuracy)
                   double dt,                       //? Step size
                   int& iters,                      //? # of iterations needed to converge (iteration variable)
                   bool GPU,                        //? false (0) --> CPU; true (1) --> GPU
                   GPU_handle& cublas_handle        //? CuBLAS handle
                   )
    {
        //* -------------------------------------------------------------------------

        //! u, u_exprb4, u_exprb5, auxiliary_expint, and auxiliary_Leja
        //! are device vectors if GPU support is activated.

        //*    Returns
        //*    ----------
        //*     u_exprb4                : double*
        //*                                 4rd order solution after time dt
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
        int iters_1 = 0, iters_2 = 0, iters_3 = 0, iters_4 = 0, iters_5 = 0, iters_6 = 0, iters_7 = 0;

        //? Assign names and variables
        double* f_u = &u_exprb4[0]; double* u_flux = &auxiliary_expint[0]; 
        double* a_n = &u_flux[0]; double* b_n = &u_flux[N]; double* c_n = &u_flux[2*N]; 
        double* b_nl = &u_exprb5[0]; double* c_nl = &u_exprb5[0];
        double* NL_u = &u_exprb4[0]; double* error_vector = &u_flux[0];
        double* NL_a = &u_exprb5[0]; double* NL_b = &u_exprb5[0]; double* NL_c = &u_exprb5[0];
        double* R_a = &u_flux[0]; double* R_b = &u_flux[N]; double* R_c = &u_flux[2*N];
        double* R_4a = &u_exprb4[0]; double* u_nl_4_3 = &u_exprb5[0];
        double* R_4b = &u_exprb4[0]; double* u_nl_4_4 = &auxiliary_expint[4*N];
        double* R_5a = &u_flux[0]; double* u_nl_5_3 = &auxiliary_expint[4*N];
        double* R_5b = &u_flux[0]; double* u_nl_5_4 = &u_exprb5[0]; 

        //? RHS evaluated at 'u' multiplied by 'dt'; f_u = RHS(u)*dt
        RHS(u, f_u);
        axpby(dt, f_u, f_u, N, GPU);

        //? Vertical interpolation of RHS(u) at 1/4, 1/2, 9/10, and 1; u_flux = phi_1({0.25, 0.5, 0.9, 1.0} J(u) dt) f(u) dt
        real_Leja_phi(RHS, u, f_u, u_flux, auxiliary_Leja, N, {0.25, 0.5, 0.9, 1.0},
                        phi_1, Leja_X, c, Gamma, tol, dt, iters_1, GPU, cublas_handle);

        //? Internal stage 1; a = u + 1/4 phi_1(1/4 J(u) dt) f(u) dt
        axpby(1.0, u, 0.25, &u_flux[0], a_n, N, GPU);

        //? R_a = (NL_a - NL_u) * dt
        Nonlinear_remainder(RHS, u, u,   NL_u, auxiliary_Leja, N, GPU, cublas_handle);
        Nonlinear_remainder(RHS, u, a_n, NL_a, auxiliary_Leja, N, GPU, cublas_handle);
        axpby(dt, NL_a, -dt, NL_u, R_a, N, GPU);

        //? Interpolation of R(a) at 1/2; b_nl = phi_3(0.5 J(u) dt) R(a) dt
        real_Leja_phi(RHS, u, R_a, b_nl, auxiliary_Leja, N, {0.5},
                        phi_3, Leja_X, c, Gamma, tol, dt, iters_2, GPU, cublas_handle);

        //? Internal stage 2; b = u + 1/2 phi_1(1/2 J(u) dt) f(u) dt + 4 phi_3(1/2 J(u) dt) R(a) dt
        axpby(1.0, u, 0.5, &u_flux[N], 4.0, b_nl, b_n, N, GPU);

        //? R_b = (NL_b - NL_u) * dt
        Nonlinear_remainder(RHS, u, b_n, NL_b, auxiliary_Leja, N, GPU, cublas_handle);
        axpby(dt, NL_b, -dt, NL_u, R_b, N, GPU);

        //? Interpolation of R(b) at 9/10; c_nl = phi_3(0.9 J(u) dt) R(b) dt
        real_Leja_phi(RHS, u, R_b, c_nl, auxiliary_Leja, N, {0.9}, 
                        phi_3, Leja_X, c, Gamma, tol, dt, iters_3, GPU, cublas_handle);

        //? Internal stage 3; c = u + (9/10) phi_1(9/10 J(u) dt) f(u) dt + (729/125) phi_3(9/10 J(u) dt) R(b) dt
        axpby(1.0, u, 9.0/10.0, &u_flux[2*N], 729.0/125.0, c_nl, c_n, N, GPU);

        //? R_c = (NL_c - NL_u) * dt
        Nonlinear_remainder(RHS, u, c_n, NL_c, auxiliary_Leja, N, GPU, cublas_handle);
        axpby(dt, NL_c, -dt, NL_u, R_c, N, GPU);

        //? R_4a = (64R(a) - 8R(b)) dt
        axpby(64.0, R_a, -8.0, R_b, R_4a, N, GPU);
        
        //? u_nl_4_3 = phi_3(J(u) dt) (64R(a) - 8R(b)) dt
        real_Leja_phi(RHS, u, R_4a, u_nl_4_3, auxiliary_Leja, N, {1.0},
                        phi_3, Leja_X, c, Gamma, tol, dt, iters_4, GPU, cublas_handle);

        //? R_4b = (-60R(a) - (285/8)R(b) + (125/8)R(c)) dt
        axpby(-60.0, R_a, -285.0/8.0, R_b, 125.0/8.0, R_c, R_4b, N, GPU);

        //? u_nl_4_4 = phi_4(J(u) dt) (-60R(a) - (285/8)R(b) + (125/8)R(c)) dt
        real_Leja_phi(RHS, u, R_4b, u_nl_4_4, auxiliary_Leja, N, {1.0},
                        phi_4, Leja_X, c, Gamma, tol, dt, iters_5, GPU, cublas_handle);

        //! 4th order solution; u_4 = u + phi_1(J(u) dt) f(u) dt + phi_3(J(u) dt) (64R(a) - 8R(b)) dt + phi_4(J(u) dt) (-60R(a) - (285/8)R(b) + (125/8)R(c)) dt
        axpby(1.0, u, 1.0, &u_flux[3*N], 1.0, u_nl_4_3, 1.0, u_nl_4_4, u_exprb4, N, GPU);

        //? R_5a = (18R(a) - (250/81)R(b)) dt
        axpby(18.0, R_b, -250.0/81.0, R_c, R_5a, N, GPU);
        
        //? u_nl_5_3 = phi_3(J(u) dt) (18R(a) - (250/81)R(b)) dt
        real_Leja_phi(RHS, u, R_5a, u_nl_5_3, auxiliary_Leja, N, {1.0},
                        phi_3, Leja_X, c, Gamma, tol, dt, iters_6, GPU, cublas_handle);

        //? R_5b = (-60R(a) + (500/27)R(b)) dt
        axpby(-60.0, R_b, 500.0/27.0, R_c, R_5b, N, GPU);

        //? u_nl_5_4 = phi_3(J(u) dt) (-60R(a) + (500/27)R(b)) dt
        real_Leja_phi(RHS, u, R_5b, u_nl_5_4, auxiliary_Leja, N, {1.0},
                        phi_4, Leja_X, c, Gamma, tol, dt, iters_7, GPU, cublas_handle);

        //! 5th order solution; u_5 = u + phi_1(J(u) dt) f(u) dt + phi_3(J(u) dt) (18R(a) - (250/81)R(b)) dt + phi_4(J(u) dt) (-60R(a) + (500/27)R(b)) dt
        axpby(1.0, u, 1.0, &u_flux[3*N], 1.0, u_nl_5_3, 1.0, u_nl_5_4, u_exprb5, N, GPU);

        //? Error estimate
        axpby(1.0, u_exprb5, -1.0, u_exprb4, error_vector, N, GPU);
        error = l2norm(error_vector, N, GPU, cublas_handle);

        //? Total number of Leja iterations
        iters = iters_1 + iters_2 + iters_3 + iters_4 + iters_5 + iters_6 + iters_7;
    }
}