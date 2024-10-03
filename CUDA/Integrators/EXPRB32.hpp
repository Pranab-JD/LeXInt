#pragma once

#include "../Leja.hpp"
#include "../Phi_functions.hpp"

namespace LeXInt
{
    //? Phi functions interpolated on real Leja points
    template <typename rhs>
    void EXPRB32(rhs& RHS,                   //? RHS function
                 double* u,                  //? Input state variable(s)
                 double* u_exprb2,           //? Output state variable(s) (lower order)
                 double* u_exprb3,           //? Output state variable(s) (higher order)
                 double& error,              //? Embedded error estimate
                 double* auxiliary_expint,   //? Internal auxiliary variables (EXPRB32)
                 double* auxiliary_Leja,     //? Internal auxiliary variables (Leja and NL remainders)
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

        //! u, u_exprb2, u_exprb3, auxiliary_expint, and auxiliary_Leja,
        //! are device vectors if GPU support is activated.

        //*    Returns
        //*    ----------
        //*     u_exprb2                : double*
        //*                                 2nd order solution after time dt
        //*     
        //*     u_exprb3                : double* 
        //*                                 3rd order solution after time dt
        //*
        //*
        //*    Reference:
        //*         M. Hochbruck, A. Ostermann, Exponential Integrators, Acta Numer. 19 (2010) 209-286. 
        //*         doi:10.1017/S0962492910000048

        //* -------------------------------------------------------------------------

        //? Counters for Leja iterations
        int iters_1 = 0, iters_2 = 0;

        //? Assign names and variables
        double* f_u = &auxiliary_expint[0]; double* u_flux = &u_exprb2[0]; 
        double* NL_u = &auxiliary_expint[0]; double* NL_a = &u_exprb3[0]; double* R_a = &u_exprb3[0]; 
        double* u_nl_3 = &auxiliary_expint[0]; double* error_vector = &auxiliary_expint[0];

        //? RHS evaluated at 'u' multiplied by 'dt'; f_u = RHS(u)*dt
        RHS(u, f_u);
        axpby(dt, f_u, f_u, N, GPU);

        //? Interpolation of RHS(u) at 1; u_flux = phi_1(J(u) dt) f(u) dt
        real_Leja_phi(RHS, u, f_u, u_flux, auxiliary_Leja, N, {1.0}, 
                      phi_1, Leja_X, c, Gamma, rtol, atol, dt, iters_1, GPU, cublas_handle);

        //! Internal stage 1; 2nd order solution; u_2 = u + phi_1(J(u) dt) f(u) dt
        axpby(1.0, u, 1.0, u_flux, u_exprb2, N, GPU);

        //? R_a = (NL_a - NL_u) * dt
        Nonlinear_remainder(RHS, u, u,        NL_u, auxiliary_Leja, N, GPU, cublas_handle);
        Nonlinear_remainder(RHS, u, u_exprb2, NL_a, auxiliary_Leja, N, GPU, cublas_handle);
        axpby(dt, NL_a, -dt, NL_u, R_a, N, GPU);

        //? u_nl_3 = phi_3(J(u) dt) R(a) dt
        real_Leja_phi(RHS, u, R_a, u_nl_3, auxiliary_Leja, N, {1.0}, 
                      phi_3, Leja_X, c, Gamma, rtol, atol, dt, iters_2, GPU, cublas_handle);
                        
        //! 3rd order solution; u_3 = u_2 + 2 phi_3(J(u) dt) R(a) dt
        axpby(1.0, u_exprb2, 2.0, u_nl_3, u_exprb3, N, GPU);

        //? Error estimate
        axpby(2.0, u_nl_3, error_vector, N, GPU);
        error = l2norm(error_vector, N, GPU, cublas_handle)/sqrt(N);

        //? Total number of Leja iterations
        iters = iters_1 + iters_2;
    }
}