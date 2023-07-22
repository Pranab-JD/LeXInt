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
    void EPIRK4s3B(rhs& RHS,                   //? RHS function
                   double* u,                  //? Input state variable(s)
                   double* u_epirk4,           //? Output state variable(s)
                   double* auxiliary_expint,   //? Internal auxiliary variables (EPIRK4s3B)
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

        //! u, u_epirk4, auxiliary_expint, auxiliary_Leja, and auxiliary_NL
        //! are device vectors if GPU support is activated.

        //*
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

        //? Assign required variables
        double* u_flux = &auxiliary_expint[0];
        double* a = &auxiliary_expint[3*N];
        double* NL_1 = &auxiliary_expint[4*N];

        //? RHS evaluated at 'u' multiplied by 'dt'; u_epirk4 = f(u)
        RHS(u, u_epirk4);
        axpby(dt, u_epirk4, u_epirk4, N, GPU);

        //? Vertical interpolation of RHS(u) at 1/2 and 3/4; phi_2({1/2, 3/4} J(u) dt) f(u) dt
        real_Leja_phi(RHS, u, u_epirk4, u_flux, auxiliary_Leja, N, {1./2., 3./4.}, 
                        phi_2, Leja_X, c, Gamma, tol, dt, iters_1, GPU, cublas_handle);

        //? Interpolation of RHS(u) at 1; phi_1(J(u) dt) f(u) dt
        real_Leja_phi(RHS, u, u_epirk4, &u_flux[2*N], auxiliary_Leja, N, {1.0}, 
                        phi_1, Leja_X, c, Gamma, tol, dt, iters_2, GPU, cublas_handle);

        //? Internal stage 1; a = u + 2/3 phi_2(1/2 J(u) dt) f(u) dt
        axpby(1.0, u, 2./3., &u_flux[0], a, N, GPU);

        //? Internal stage 2; b = u + phi_2(3/4 J(u) dt) f(u) dt
        double* b = &u_flux[0];
        axpby(1.0, u, 1.0, &u_flux[N], b, N, GPU);

        //? Difference of nonlinear remainders at a and b
        double* NL_2 = &u_flux[N];
        Nonlinear_remainder(RHS, u, u, u_epirk4, auxiliary_Leja, N, GPU, cublas_handle);
        Nonlinear_remainder(RHS, u, a, NL_1,     auxiliary_Leja, N, GPU, cublas_handle);
        Nonlinear_remainder(RHS, u, b, NL_2,     auxiliary_Leja, N, GPU, cublas_handle);
        axpby(dt, NL_1, -dt, u_epirk4, a, N, GPU);
        axpby(dt, NL_2, -dt, u_epirk4, b, N, GPU);

        //? Final nonlinear stages
        axpby(  54.0, a, -16.0, b, NL_1, N, GPU);
        axpby(-324.0, a, 144.0, b, NL_2, N, GPU);

        //? phi_3(J(u) dt) (54R(a) - 16R(b)) dt
        real_Leja_phi(RHS, u, NL_1, a, auxiliary_Leja, N, {1.0}, 
                        phi_3, Leja_X, c, Gamma, tol, dt, iters_3, GPU, cublas_handle);

        //? phi_4(J(u) dt) (-324R(a) + 144R(b)) dt
        real_Leja_phi(RHS, u, NL_2, b, auxiliary_Leja, N, {1.0}, 
                        phi_4, Leja_X, c, Gamma, tol, dt, iters_4, GPU, cublas_handle);

        //? 4th order solution; u_4 = u + phi_1(J(u) dt) f(u) dt + phi_3(J(u) dt) (54R(a) - 16R(b)) dt + phi_4(J(u) dt) (-324R(a) + 144R(b)) dt
        axpby(1.0, u, 1.0, &u_flux[2*N], 1.0, a, 1.0, b, u_epirk4, N, GPU);

        //? Total number of Leja iterations
        iters = iters_1 + iters_2 + iters_3 + iters_4;
    }
}