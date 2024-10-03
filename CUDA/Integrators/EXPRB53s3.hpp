#pragma once

#include "../Leja.hpp"
#include "../Phi_functions.hpp"

namespace LeXInt
{
    //? Phi functions interpolated on real Leja points
    template <typename rhs>
    void EXPRB53s3(rhs& RHS,                   //? RHS function
                   double* u,                  //? Input state variable(s)
                   double* u_exprb3,           //? Output state variable(s) (lower order)
                   double* u_exprb5,           //? Output state variable(s) (higher order)
                   double& error,              //? Embedded error estimate
                   double* auxiliary_expint,   //? Internal auxiliary variables (EXPRB53s3)
                   double* auxiliary_Leja,     //? Internal auxiliary variables (Leja)
                   size_t N,                   //? Number of grid points
                   vector<double>& Leja_X,     //? Array of Leja points
                   double c,                   //? Shifting factor
                   double Gamma,               //? Scaling factor
                   double rtol,                //? Relative tolerance (normalised desired accuracy)
                   double atol,                //? Absolute tolerance
                   double dt,                  //? Step size
                   int& iters,                 //? # of iterations needed to converge (iteration variable)
                   bool GPU,                   //? false (0) --> CPU; true (1) --> GPU
                   GPU_handle& cublas_handle   //? CuBLAS handle
                   )
    {
        //* -------------------------------------------------------------------------

        //! u, u_exprb3, u_exprb5, auxiliary_expint, and auxiliary_Leja
        //! are device vectors if GPU support is activated.

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

        //? Assign names and variables
        double* f_u = &u_exprb3[0]; double* u_flux = &auxiliary_expint[0];
        double* a = &u_flux[0]; double* b = &u_flux[N]; double* b_nl = &auxiliary_expint[3*N];
        double* NL_u = &u_exprb3[0]; double* NL_a = &u_exprb5[0]; double* NL_b = &u_exprb5[0];
        double* R_a = &u_flux[0]; double* R_b = &u_flux[N];
        double* R_3 = &b_nl[0]; double* R_4 = &b_nl[N]; double* error_vector = &b_nl[0];
        double* u_nl_4_3 = &u_exprb3[0]; double* u_nl_5_3 = &u_flux[0]; double* u_nl_5_4 = &u_flux[N];
        
        //? RHS evaluated at 'u' multiplied by 'dt'; f_u = RHS(u)*dt
        RHS(u, f_u);
        axpby(dt, f_u, f_u, N, GPU);

        //? Vertical interpolation of RHS(u) at 1/2, 9/10, and 1; u_flux = phi_1({0.5, 0.9, 1.0} J(u) dt) f(u) dt
        real_Leja_phi(RHS, u, f_u, u_flux, auxiliary_Leja, N, {0.5, 0.9, 1.0},
                      phi_1, Leja_X, c, Gamma, rtol, atol, dt, iters_1, GPU, cublas_handle);

        //? Internal stage 1; a = u + 1/2 phi_1(1/2 J(u) dt) f(u) dt
        axpby(1.0, u, 0.5, &u_flux[0], a, N, GPU);

        //? R_a = (NL_a - NL_u) * dt
        Nonlinear_remainder(RHS, u, u, NL_u, auxiliary_Leja, N, GPU, cublas_handle);
        Nonlinear_remainder(RHS, u, a, NL_a, auxiliary_Leja, N, GPU, cublas_handle);
        axpby(dt, NL_a, -dt, NL_u, R_a, N, GPU);

        //? Vertical interpolation of R(a) at 1/2 and 9/10, b_nl = phi_3({0.5, 0.9} J(u) dt) R(a) dt
        real_Leja_phi(RHS, u, R_a, b_nl, auxiliary_Leja, N, {0.5, 0.9}, 
                      phi_3, Leja_X, c, Gamma, rtol, atol, dt, iters_2, GPU, cublas_handle);

        //? Internal stage 2; b = u + 9/10 phi_1(9/10 J(u) dt) f(u) dt + (27/25 phi_3(1/2 J(u) dt) + 729/125 phi_3(9/10 J(u) dt)) R(a) dt
        axpby(1.0, u, 0.9, &u_flux[N], 27.0/25.0, &b_nl[0], 729.0/125.0, &b_nl[N], b, N, GPU);

        //? R_b = (NL_b - NL_u) * dt
        Nonlinear_remainder(RHS, u, b, NL_b, auxiliary_Leja, N, GPU, cublas_handle);
        axpby(dt, NL_b, -dt, NL_u, R_b, N, GPU);

        //? R_3 = (2R(a) + 150/81R(b)) dt
        axpby(2.0, R_a, 150.0/81.0, R_b, R_3, N, GPU);

        //? u_nl_3_1 = phi_3(J(u) dt) (2R(a) + 150/81R(b)) dt
        real_Leja_phi(RHS, u, R_3, u_nl_4_3, auxiliary_Leja, N, {1.0},
                      phi_3, Leja_X, c, Gamma, rtol, atol, dt, iters_3, GPU, cublas_handle);

        //! 3rd order solution; u_3 = u + phi_1(J(u) dt) f(u) dt + phi_3(J(u) dt) (2R(a) + (150/81)R(b)) dt
        axpby(1.0, u, 1.0, &u_flux[2*N], 1.0, u_nl_4_3, u_exprb3, N, GPU);

        //? R_3 = (18R(a) - 250/81R(b)) dt
        axpby(18.0, R_a, -250.0/81.0, R_b, R_3, N, GPU);

        //? R_4 = phi_4(J(u) dt) (-60R(a) + (500/27)R(b)) dt
        axpby(-60.0, R_a, 500.0/27.0, R_b, R_4, N, GPU);

        //? u_nl_3_2 = phi_3(J(u) dt) (18R(a) - 250/81R(b)) dt
        real_Leja_phi(RHS, u, R_3, u_nl_5_3, auxiliary_Leja, N, {1.0},
                      phi_3, Leja_X, c, Gamma, rtol, atol, dt, iters_4, GPU, cublas_handle);

        //? u_nl_4 = (-60R(a) + (500/27)R(b)) dt
        real_Leja_phi(RHS, u, R_4, u_nl_5_4, auxiliary_Leja, N, {1.0},
                      phi_4, Leja_X, c, Gamma, rtol, atol, dt, iters_5, GPU, cublas_handle);

        //! 5th order solution; u_5 = u + phi_1(J(u) dt) f(u) dt + phi_3(J(u) dt) (18R(a) - (250/81)R(b)) dt + phi_4(J(u) dt) (-60R(a) + (500/27)R(b)) dt
        axpby(1.0, u, 1.0, &u_flux[2*N], 1.0, u_nl_5_3, 1.0, u_nl_5_4, u_exprb5, N, GPU);

        //? Error estimate
        axpby(1.0, u_exprb5, -1.0, u_exprb3, error_vector, N, GPU);
        error = l2norm(error_vector, N, GPU, cublas_handle);

        //? Total number of Leja iterations
        iters = iters_1 + iters_2 + iters_3 + iters_4 + iters_5;
    }
}