#pragma once

#include "../Leja.hpp"
#include "../Phi_functions.hpp"

namespace LeXInt
{
    //? Phi functions interpolated on real Leja points
    template <typename rhs>
    void EXPRB43(rhs& RHS,                   //? RHS function
                 double* u,                  //? Input state variable(s)
                 double* u_exprb3,           //? Output state variable(s) (lower order)
                 double* u_exprb4,           //? Output state variable(s) (higher order)
                 double& error,              //? Embedded error estimate
                 double* auxiliary_expint,   //? Internal auxiliary variables (EXPRB43)
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

        //! u, u_exprb3, u_exprb4, auxiliary_expint, and auxiliary_Leja
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

        //? Assign names and variables
        double* f_u = &auxiliary_expint[2*N]; double* u_flux = &auxiliary_expint[0]; 
        double* a = &u_flux[0]; double* b = &u_flux[0]; double* b_nl = &u_flux[0]; 
        double* NL_u = &auxiliary_expint[2*N]; double* NL_a = &u_exprb4[0]; double* NL_b = &u_exprb4[0];
        double* R_a = &u_exprb3[0]; double* R_b = &auxiliary_expint[2*N];
        double* R_3 = &u_exprb4[0]; double* R_4 = &u_exprb3[0];
        double* u_nl_3 = &u_flux[0]; double* u_nl_4 = &auxiliary_expint[2*N];

        //? RHS evaluated at 'u' multiplied by 'dt'; f_u = RHS(u)*dt
        RHS(u, f_u);
        axpby(dt, f_u, f_u, N, GPU);

        //? Vertical interpolation of RHS(u) at 0.5 and 1; u_flux = phi_1({0.5, 1.0} J(u) dt) f(u) dt
        real_Leja_phi(RHS, u, f_u, u_flux, auxiliary_Leja, N, {0.5, 1.0}, 
                        phi_1, Leja_X, c, Gamma, tol, dt, iters_1, GPU, cublas_handle);

        //? Internal stage 1; a = u + 1/2 phi_1(1/2 J(u) dt) f(u) dt
        axpby(1.0, u, 0.5, &u_flux[0], a, N, GPU);

        //? R_a = (NL_a - NL_u) * dt
        Nonlinear_remainder(RHS, u, u, NL_u, auxiliary_Leja, N, GPU, cublas_handle);
        Nonlinear_remainder(RHS, u, a, NL_a, auxiliary_Leja, N, GPU, cublas_handle);
        axpby(dt, NL_a, -dt, NL_u, R_a, N, GPU);

        //? Internal stage 2; b_nl = phi_1(J(u) dt) R(a) dt
        real_Leja_phi(RHS, u, R_a, &b_nl[0], auxiliary_Leja, N, {1.0}, 
                        phi_1, Leja_X, c, Gamma, tol, dt, iters_2, GPU, cublas_handle);

        //? b = u + phi_1(J(u) dt) f(u) dt + phi_1(J(u) dt) R(a) dt
        axpby(1.0, u, 1.0, &u_flux[N], 1.0, &b_nl[0], b, N, GPU);

        //? R_b = (NL_b - NL_u) * dt
        Nonlinear_remainder(RHS, u, b, NL_b, auxiliary_Leja, N, GPU, cublas_handle);
        axpby(dt, NL_b, -dt, NL_u, R_b, N, GPU);

        //? R_3 = (16R(a) - 2R(b)) dt
        axpby(16.0, R_a, -2.0, R_b, R_3, N, GPU);

        //? R_4 = (-48R(a) + 12R(b)) dt
        axpby(-48.0, R_a, 12.0, R_b, R_4, N, GPU);

        //? u_nl_3 = phi_3(J(u) dt) (16R(a) - 2R(b)) dt
        real_Leja_phi(RHS, u, R_3, &u_nl_3[0], auxiliary_Leja, N, {1.0},
                        phi_3, Leja_X, c, Gamma, tol, dt, iters_3, GPU, cublas_handle);
        
        //? u_nl_4 = phi_4(J(u) dt) (-48R(a) + 12R(b)) dt
        real_Leja_phi(RHS, u, R_4, &u_nl_4[0], auxiliary_Leja, N, {1.0},
                        phi_4, Leja_X, c, Gamma, tol, dt, iters_4, GPU, cublas_handle);

        //! 3rd order solution; u_3 = u + phi_1(J(u) dt) f(u) dt + phi_3(J(u) dt) (16R(a) - 2R(b)) dt
        axpby(1.0, u, 1.0, &u_flux[N], 1.0, u_nl_3, &u_exprb3[0], N, GPU);

        //! 4th order solution; u_4 = u_3 + phi_4(J(u) dt) (-48R(a) + 12R(b)) dt
        axpby(1.0, u_exprb3, 1.0, u_nl_4, u_exprb4, N, GPU);

        //? Embedded error estimate
        error = l2norm(u_nl_4, N, GPU, cublas_handle);

        //? Total number of Leja iterations
        iters = iters_1 + iters_2 + iters_3 + iters_4;
    }
}