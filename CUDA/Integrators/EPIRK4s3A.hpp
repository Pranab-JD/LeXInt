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
    void EPIRK4s3A(rhs& RHS,                   //? RHS function
                   double* u,                  //? Input state variable(s)
                   double* u_epirk3,           //? Output state variable(s) (lower order)
                   double* u_epirk4,           //? Output state variable(s) (higher order)
                   double* auxillary_expint,   //? Internal auxillary variables (EPIRK4s3A)
                   double* auxillary_Leja,     //? Internal auxillary variables (Leja)
                   double* auxillary_NL,       //? Internal auxillary variables (EPIRK4s3A NL)
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

        //! u, u_epirk3, u_epirk4, auxillary_expint, auxillary_Leja, and auxillary_NL
        //! are device vectors if GPU support is activated.

        //*    Returns
        //*    ----------
        //*     u_epirk3                : double*
        //*                                 3rd order solution after time dt
        //*     
        //*     u_epirk4                : double* 
        //*                                 4th order solution after time dt
        //*
        //*
        //*    Reference:
        //*         G. Rainwater, M. Tokman, A new approach to constructing efficient stiffly accurate EPIRK methods, J. Comput. Phys. 323 (2016) 283-309.
        //*         doi:10.1016/j.jcp.2016.07.026

        //* -------------------------------------------------------------------------

        //? RHS evaluated at 'u' multiplied by 'dt'
        double* rhs_u = &auxillary_expint[0];
        RHS(u, rhs_u);
        axpby(dt, rhs_u, rhs_u, N, GPU);

        //? Vertical interpolation of RHS(u) at 1/2, 2/3, and 1; phi_1({1/2, 2/3, 1.0} J(u) dt) f(u) dt
        double* u_flux = &auxillary_expint[N];
        real_Leja_phi(RHS, u, rhs_u, u_flux, auxillary_Leja, N, {1./2., 2./3., 1.0}, 
                        phi_1, Leja_X, c, Gamma, tol, dt, GPU, cublas_handle);

        //? Internal stage 1; a = u + 1/2 phi_1(1/2 J(u) dt) f(u) dt
        double* a = &auxillary_expint[4*N];
        axpby(1.0, u, 1./3., &u_flux[0], a, N, GPU);

        //? Internal stage 2; b = u + 2/3 phi_1(2/3 J(u) dt) f(u) dt
        double* b = &auxillary_expint[5*N];
        axpby(1.0, u, 2./3., &u_flux[N], b, N, GPU);

        //? Assign memory for nonlinear remainders
        double* NL_u = &auxillary_expint[6*N];
        double* NL_a = &auxillary_expint[7*N];
        double* NL_b = &auxillary_expint[8*N];

        double* R_a = &auxillary_expint[9*N];
        double* R_b = &auxillary_expint[10*N];

        //? Difference of nonlinear remainders at a and b
        Nonlinear_remainder(RHS, u, u, NL_u, auxillary_NL, N, GPU, cublas_handle);
        Nonlinear_remainder(RHS, u, a, NL_a, auxillary_NL, N, GPU, cublas_handle);
        Nonlinear_remainder(RHS, u, b, NL_b, auxillary_NL, N, GPU, cublas_handle);
        axpby(dt, NL_a, -dt, NL_u, R_a, N, GPU);
        axpby(dt, NL_b, -dt, NL_u, R_b, N, GPU);

        //? Final nonlinear stages
        double* R_3 = &auxillary_expint[11*N];
        double* R_4 = &auxillary_expint[12*N];
        axpby(  32.0, R_a, -27.0/2.0, R_b, R_3, N, GPU);
        axpby(-144.0, R_a,      81.0, R_b, R_4, N, GPU);

        //? phi_3(J(u) dt) (32R(a) - (27/2)R(b)) dt
        double* u_nl_3 = &auxillary_expint[13*N];
        real_Leja_phi(RHS, u, R_3, u_nl_3, auxillary_Leja, N, {1.0}, 
                        phi_3, Leja_X, c, Gamma, tol, dt, GPU, cublas_handle);

        //? phi_4(J(u) dt) (-144R(a) + 81R(b)) dt
        double* u_nl_4 = &auxillary_expint[14*N];
        real_Leja_phi(RHS, u, R_4, u_nl_4, auxillary_Leja, N, {1.0}, 
                        phi_4, Leja_X, c, Gamma, tol, dt, GPU, cublas_handle);

        //? 3rd order solution; u_3 = u + phi_1(J(u) dt) f(u) dt + phi_3(J(u) dt) (32R(a) - (27/2)R(b)) dt
        axpby(1.0, u, 1.0, &u_flux[2*N], u_epirk3, N, GPU);

        //? 4th order solution; u_4 = u_3 + phi_4(J(u) dt) (-144R(a) + 81R(b)) dt
        axpby(1.0, u_epirk3, 1.0, u_nl_4, u_epirk4, N, GPU);
    }
}