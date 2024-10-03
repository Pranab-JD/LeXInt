#pragma once

#include "../Leja.hpp"
#include "../Phi_functions.hpp"

namespace LeXInt
{
    //? Phi functions interpolated on real Leja points
    template <typename rhs>
    void EPIRK4s3B(rhs& RHS,                   //? RHS function
                   double* u,                  //? Input state variable(s)
                   double* u_epirk4,           //? Output state variable(s)
                   double* auxiliary_expint,   //? Internal auxiliary variables (EPIRK4s3B)
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

        //! u, u_epirk4, auxiliary_expint, and auxiliary_Leja
        //! are device vectors if GPU support is activated.

        //*    Returns
        //*    ----------
        //*     
        //*     u_epirk4                : double* 
        //*                                  4th order solution after time dt
        //*
        //*
        //*    Reference:
        //*         G. Rainwater, M. Tokman, A new approach to constructing efficient stiffly accurate EPIRK methods, J. Comput. Phys. 323 (2016) 283-309.
        //*         doi:10.1016/j.jcp.2016.07.026

        //* -------------------------------------------------------------------------

        //? Counters for Leja iterations
        int iters_1 = 0, iters_2 = 0, iters_3 = 0, iters_4 = 0;

        //? Assign names and variables
        double* u_flux = &auxiliary_expint[0]; double* f_u = &auxiliary_expint[3*N];
        double* a = &u_flux[0]; double* b = &u_flux[N]; double* NL_u = &u_epirk4[0]; 
        double* NL_a = &auxiliary_expint[3*N]; double* NL_b = &auxiliary_expint[4*N];
        double* R_a = &auxiliary_expint[3*N]; double* R_b = &auxiliary_expint[4*N];
        double* R_3 = &u_flux[0]; double* R_4 = &u_flux[N];
        double* u_nl_3 = &auxiliary_expint[3*N]; double* u_nl_4 = &auxiliary_expint[4*N];

        //? RHS evaluated at 'u' multiplied by 'dt'; f_u = RHS(u)*dt
        RHS(u, f_u);
        axpby(dt, f_u, f_u, N, GPU);

        //? Vertical interpolation of RHS(u) at 1/2 and 3/4; u_flux[0, 1] = phi_2({1/2, 3/4} J(u) dt) f(u) dt
        real_Leja_phi(RHS, u, f_u, u_flux, auxiliary_Leja, N, {1./2., 3./4.}, 
                      phi_2, Leja_X, c, Gamma, rtol, atol, dt, iters_1, GPU, cublas_handle);

        //? Interpolation of RHS(u) at 1; u_flux[2] = phi_1(J(u) dt) f(u) dt
        real_Leja_phi(RHS, u, f_u, &u_flux[2*N], auxiliary_Leja, N, {1.0}, 
                      phi_1, Leja_X, c, Gamma, rtol, atol, dt, iters_2, GPU, cublas_handle);

        //? Internal stage 1; a = u + 2/3 phi_2(1/2 J(u) dt) f(u) dt
        axpby(1.0, u, 2./3., &u_flux[0], a, N, GPU);

        //? Internal stage 2; b = u + phi_2(3/4 J(u) dt) f(u) dt
        axpby(1.0, u, 1.0, &u_flux[N], b, N, GPU);

        //? R_a = (NL_a - NL_u) * dt; R_b = (NL_b - NL_u) * dt
        Nonlinear_remainder(RHS, u, u, NL_u, auxiliary_Leja, N, GPU, cublas_handle);
        Nonlinear_remainder(RHS, u, a, NL_a, auxiliary_Leja, N, GPU, cublas_handle);
        Nonlinear_remainder(RHS, u, b, NL_b, auxiliary_Leja, N, GPU, cublas_handle);
        axpby(dt, NL_a, -dt, NL_u, R_a, N, GPU);
        axpby(dt, NL_b, -dt, NL_u, R_b, N, GPU);

        //? R_3 = (54R(a) - 16R(b)) dt
        axpby(54.0, R_a, -16.0, R_b, R_3, N, GPU);

        //? R_4 = (-324R(a) + 144R(b)) dt
        axpby(-324.0, R_a, 144.0, R_b, R_4, N, GPU);

        //? u_nl_3 = phi_3(J(u) dt) (54R(a) - 16R(b)) dt
        real_Leja_phi(RHS, u, R_3, u_nl_3, auxiliary_Leja, N, {1.0}, 
                        phi_3, Leja_X, c, Gamma, rtol, atol, dt, iters_3, GPU, cublas_handle);

        //? u_nl_4 = phi_4(J(u) dt) (-324R(a) + 144R(b)) dt
        real_Leja_phi(RHS, u, R_4, u_nl_4, auxiliary_Leja, N, {1.0}, 
                      phi_4, Leja_X, c, Gamma, rtol, atol, dt, iters_4, GPU, cublas_handle);

        //! 4th order solution; u_4 = u + phi_1(J(u) dt) f(u) dt + phi_3(J(u) dt) (54R(a) - 16R(b)) dt + phi_4(J(u) dt) (-324R(a) + 144R(b)) dt
        axpby(1.0, u, 1.0, &u_flux[2*N], 1.0, u_nl_3, 1.0, u_nl_4, u_epirk4, N, GPU);

        //? Total number of Leja iterations
        iters = iters_1 + iters_2 + iters_3 + iters_4;
    }
}