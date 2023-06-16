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

        //! u, u_exprb3, u_exprb5, auxillary_expint, auxillary_Leja, and auxillary_NL
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

        //? RHS evaluated at 'u' multiplied by 'dt'
        double* rhs_u = &auxillary_expint[0];
        RHS(u, rhs_u);
        axpby(dt, rhs_u, rhs_u, N, GPU);

        //? Vertical interpolation of RHS(u) at 1/2, 9/10, and 1
        double* u_flux = &auxillary_expint[N];
        real_Leja_phi(RHS, u, rhs_u, u_flux, auxillary_Leja, N, {0.5, 0.9, 1.0}, 
                        phi_1, Leja_X, c, Gamma, tol, dt, GPU, cublas_handle);

        //? Internal stage 1; a = u + 1/2 phi_1(1/2 J(u) dt) f(u) dt
        double* a = &auxillary_expint[4*N];
        axpby(1.0, u, 0.5, &u_flux[0], a, N, GPU);

        //? Nonlinear remainder at u and a
        double* NL_u = &auxillary_expint[5*N];
        double* NL_a = &auxillary_expint[6*N];
        double* R_a = &auxillary_expint[7*N];

        Nonlinear_remainder(RHS, u, u, NL_u, auxillary_NL, N, GPU, cublas_handle);
        Nonlinear_remainder(RHS, u, a, NL_a, auxillary_NL, N, GPU, cublas_handle);
        axpby(dt, NL_a, -dt, NL_u, R_a, N, GPU);

        //? Vertical interpolation of R(a) at 1/2 and 9/10
        double* b_nl = &auxillary_expint[8*N];
        real_Leja_phi(RHS, u, R_a, b_nl, auxillary_Leja, N, {0.5, 0.9}, 
                        phi_3, Leja_X, c, Gamma, tol, dt, GPU, cublas_handle);

        //? Internal stage 2; b = u + 9/10 phi_1(9/10 J(u) dt) f(u) dt + (27/25 phi_3(1/2 J(u) dt) + 729/125 phi_3(9/10 J(u) dt)) R(a) dt
        double* b = &auxillary_expint[10*N];
        axpby(1.0, u, 0.9, &u_flux[N], 27.0/25.0, &b_nl[0], 729.0/125.0, &b_nl[N], b, N, GPU);

        //? Nonlinear remainder at b
        double* NL_b = &auxillary_expint[11*N];
        double* R_b = &auxillary_expint[12*N];

        Nonlinear_remainder(RHS, u, b, NL_b, auxillary_NL, N, GPU, cublas_handle);
        axpby(dt, NL_b, -dt, NL_u, R_b, N, GPU);

        //? Final nonlinear stages
        double* R_3a = &auxillary_expint[13*N];
        double* R_3b = &auxillary_expint[14*N];
        double* R_4 = &auxillary_expint[15*N];

        axpby(  2.0, R_a,  150.0/81.0, R_b, R_3a, N, GPU);
        axpby( 18.0, R_a, -250.0/81.0, R_b, R_3b, N, GPU);
        axpby(-60.0, R_a,  500.0/27.0, R_b, R_4,  N, GPU);

        //? phi_3(J(u) dt) (2R(a) + 150/81R(b)) dt
        double* u_nl_4_3 = &auxillary_expint[16*N];
        real_Leja_phi(RHS, u, R_3a, u_nl_4_3, auxillary_Leja, N, {1.0},
                        phi_3, Leja_X, c, Gamma, tol, dt, GPU, cublas_handle);

        //? phi_3(J(u) dt) (18R(a) - 250/81R(b)) dt
        double* u_nl_5_3 = &auxillary_expint[17*N];
        real_Leja_phi(RHS, u, R_3b, u_nl_5_3, auxillary_Leja, N, {1.0},
                        phi_3, Leja_X, c, Gamma, tol, dt, GPU, cublas_handle);

        //? phi_4(J(u) dt) (-60R(a) + (500/27)R(b)) dt
        double* u_nl_5_4 = &auxillary_expint[18*N];
        real_Leja_phi(RHS, u, R_4, u_nl_5_4, auxillary_Leja, N, {1.0},
                        phi_4, Leja_X, c, Gamma, tol, dt, GPU, cublas_handle);

        //? 3rd order solution; u_3 = u + phi_1(J(u) dt) f(u) dt + phi_3(J(u) dt) (2R(a) + (150/81)R(b)) dt
        axpby(1.0, u, 1.0, &u_flux[2*N], 1.0, u_nl_4_3, u_exprb3, N, GPU);

        //? 5th order solution; u_5 = u + phi_1(J(u) dt) f(u) dt + phi_3(J(u) dt) (18R(a) - (250/81)R(b)) dt + phi_4(J(u) dt) (-60R(a) + (500/27)R(b)) dt
        axpby(1.0, u, 1.0, &u_flux[2*N], 1.0, u_nl_5_3, 1.0, u_nl_5_4, u_exprb5, N, GPU);
    }
}