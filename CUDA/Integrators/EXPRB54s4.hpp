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
    void EXPRB54s4(rhs& RHS,                   //? RHS function
                   double* u,                  //? Input state variable(s)
                   double* u_exprb4,           //? Output state variable(s) (lower order)
                   double* u_exprb5,           //? Output state variable(s) (higher order)
                   double* auxillary_expint,   //? Internal auxillary variables (EXPRB54s4)
                   double* auxillary_Leja,     //? Internal auxillary variables (Leja)
                   double* auxillary_NL,       //? Internal auxillary variables (EXPRB54s4 NL)
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

        //! u, u_exprb4, u_exprb5, auxillary_expint, auxillary_Leja, and auxillary_NL
        //! are device vectors if GPU support is activated.

        //*
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

        //? RHS evaluated at 'u' multiplied by 'dt'
        double* rhs_u = &auxillary_expint[0];
        RHS(u, rhs_u);
        axpby(dt, rhs_u, rhs_u, N, GPU);

        //? Vertical interpolation of RHS(u) at 1/4, 1/2, 9/10, and 1
        double* u_flux = &auxillary_expint[N];
        vector<double> coeffs_1 = {0.25, 0.5, 0.9, 1.0};
        real_Leja_phi(RHS, u, rhs_u, u_flux, auxillary_Leja, N, coeffs_1, 
                        phi_1, Leja_X, c, Gamma, tol, dt, GPU, cublas_handle);

        //? Internal stage 1; a = u + 1/4 phi_1(1/4 J(u) dt) f(u) dt
        double* a_n = &auxillary_expint[5*N];
        axpby(1.0, u, 0.25, &u_flux[0], a_n, N, GPU);

        //? Nonlinear remainder at u and a
        double* NL_u = &auxillary_expint[6*N];
        double* NL_a = &auxillary_expint[7*N];
        double* R_a = &auxillary_expint[8*N];

        Nonlinear_remainder(RHS, u, u, NL_u, auxillary_NL, N, GPU, cublas_handle);
        Nonlinear_remainder(RHS, u, a_n, NL_a, auxillary_NL, N, GPU, cublas_handle);
        axpby(dt, NL_a, -dt, NL_u, R_a, N, GPU);

        //? Interpolation of R(a) at 1/2
        double* b_nl = &auxillary_expint[9*N];
        real_Leja_phi(RHS, u, R_a, b_nl, auxillary_Leja, N, {0.5}, 
                        phi_3, Leja_X, c, Gamma, tol, dt, GPU, cublas_handle);

        //? Internal stage 2; b = u + (1/2) phi_1(1/2 J(u) dt) f(u) dt + 4 phi_3(1/2 J(u) dt) R(a) dt
        double* b_n = &auxillary_expint[10*N];
        axpby(1.0, u, 0.5, &u_flux[N], 4.0, b_nl, b_n, N, GPU);

        //? Nonlinear remainder at b
        double* NL_b = &auxillary_expint[11*N];
        double* R_b = &auxillary_expint[12*N];

        Nonlinear_remainder(RHS, u, b_n, NL_b, auxillary_NL, N, GPU, cublas_handle);
        axpby(dt, NL_b, -dt, NL_u, R_b, N, GPU);

        //? Interpolation of R(b) at 9/10
        double* c_nl = &auxillary_expint[13*N];
        real_Leja_phi(RHS, u, R_b, c_nl, auxillary_Leja, N, {0.9}, 
                        phi_3, Leja_X, c, Gamma, tol, dt, GPU, cublas_handle);

        //? Internal stage 2; b = u + (9/10) phi_1(9/10 J(u) dt) f(u) dt + (729/125) phi_3(9/10 J(u) dt) R(b) dt
        double* c_n = &auxillary_expint[14*N];
        axpby(1.0, u, 0.9, &u_flux[2*N], 729.0/125.0, c_nl, c_n, N, GPU);

        //? Nonlinear remainder at c
        double* NL_c = &auxillary_expint[15*N];
        double* R_c = &auxillary_expint[16*N];

        Nonlinear_remainder(RHS, u, c_n, NL_c, auxillary_NL, N, GPU, cublas_handle);
        axpby(dt, NL_c, -dt, NL_u, R_c, N, GPU);

        //? Final nonlinear stages
        double* R_4a = &auxillary_expint[17*N];
        double* R_4b = &auxillary_expint[18*N];
        double* R_5a = &auxillary_expint[19*N];
        double* R_5b = &auxillary_expint[20*N];

        axpby( 64.0, R_a,        -8.0, R_b, R_4a, N, GPU);
        axpby(-60.0, R_a,  -285.0/8.0, R_b, 125.0/8.0, R_c, R_4b, N, GPU);
        axpby( 18.0, R_b, -250.0/81.0, R_c, R_5a, N, GPU);
        axpby(-60.0, R_b,  500.0/27.0, R_c, R_5b, N, GPU);

        //? phi_3(J(u) dt) (64R(a) - 8R(b)) dt
        double* u_nl_4_3 = &auxillary_expint[21*N];
        real_Leja_phi(RHS, u, R_4a, u_nl_4_3, auxillary_Leja, N, {1.0},
                        phi_3, Leja_X, c, Gamma, tol, dt, GPU, cublas_handle);

        //? phi_4(J(u) dt) (-60R(a) - (285/8)R(b) + (125/8)R(c)) dt
        double* u_nl_4_4 = &auxillary_expint[22*N];
        real_Leja_phi(RHS, u, R_4b, u_nl_4_4, auxillary_Leja, N, {1.0},
                        phi_4, Leja_X, c, Gamma, tol, dt, GPU, cublas_handle);

        //? phi_3(J(u) dt) (18R(a) - (250/81)R(b)) dt
        double* u_nl_5_3 = &auxillary_expint[23*N];
        real_Leja_phi(RHS, u, R_5a, u_nl_5_3, auxillary_Leja, N, {1.0},
                        phi_3, Leja_X, c, Gamma, tol, dt, GPU, cublas_handle);

        //? phi_3(J(u) dt) (-60R(a) + (500/27)R(b)) dt
        double* u_nl_5_4 = &auxillary_expint[24*N];
        real_Leja_phi(RHS, u, R_5b, u_nl_5_4, auxillary_Leja, N, {1.0},
                        phi_4, Leja_X, c, Gamma, tol, dt, GPU, cublas_handle);

        //? 4th order solution; u_4 = u + phi_1(J(u) dt) f(u) dt + phi_3(J(u) dt) (64R(a) - 8R(b)) dt + phi_4(J(u) dt) (-60R(a) - (285/8)R(b) + (125/8)R(c)) dt
        axpby(1.0, u, 1.0, &u_flux[3*N], 1.0, u_nl_4_3, 1.0, u_nl_4_4, u_exprb4, N, GPU);

        //? 5th order solution; u_5 = u + phi_1(J(u) dt) f(u) dt + phi_3(J(u) dt) (18R(a) - (250/81)R(b)) dt + phi_4(J(u) dt) (-60R(a) + (500/27)R(b)) dt
        axpby(1.0, u, 1.0, &u_flux[3*N], 1.0, u_nl_5_3, 1.0, u_nl_5_4, u_exprb5, N, GPU);
    }
}