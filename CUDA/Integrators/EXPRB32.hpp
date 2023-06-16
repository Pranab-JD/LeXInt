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
    void EXPRB32(rhs& RHS,                   //? RHS function
                 double* u,                  //? Input state variable(s)
                 double* u_exprb2,           //? Output state variable(s) (lower order)
                 double* u_exprb3,           //? Output state variable(s) (higher order)
                 double* auxillary_expint,   //? Internal auxillary variables (EXPRB32)
                 double* auxillary_Leja,     //? Internal auxillary variables (Leja)
                 double* auxillary_NL,       //? Internal auxillary variables (EXPRB32 NL)
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

        //! u, u_exprb2, u_exprb3, auxillary_expint, auxillary_Leja, and auxillary_NL
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

        //? RHS evaluated at 'u' multiplied by 'dt'
        double* rhs_u = &auxillary_expint[0];
        RHS(u, rhs_u);
        axpby(dt, rhs_u, rhs_u, N, GPU);
        
        //? Interpolation of RHS(u) at 1; phi_1(J(u) dt) f(u) dt
        double* u_flux = &auxillary_expint[N];
        real_Leja_phi(RHS, u, rhs_u, u_flux, auxillary_Leja, N, {1.0}, 
                        phi_1, Leja_X, c, Gamma, tol, dt, GPU, cublas_handle);

        //? Internal stage 1; 2nd order solution; u_2 = u + phi_1(J(u) dt) f(u) dt
        axpby(1.0, u, 1.0, u_flux, u_exprb2, N, GPU);

        //? Assign memory for nonlinear remainders
        double* NL_u = &auxillary_expint[2*N];
        double* NL_u_exprb2 = &auxillary_expint[3*N];
        double* R_a = &auxillary_expint[4*N];

        //? Difference of nonlinear remainders at u_exprb2
        Nonlinear_remainder(RHS, u, u, NL_u, auxillary_NL, N, GPU, cublas_handle);
        Nonlinear_remainder(RHS, u, u_exprb2, NL_u_exprb2, auxillary_NL, N, GPU, cublas_handle);
        axpby(dt, NL_u_exprb2, -dt, NL_u, R_a, N, GPU);

        //? Final nonlinear stage; phi_3(J(u) dt) R(a) dt
        double* u_nl_3 = &auxillary_expint[5*N];
        real_Leja_phi(RHS, u, R_a, u_nl_3, auxillary_Leja, N, {1.0}, 
                        phi_3, Leja_X, c, Gamma, tol, dt, GPU, cublas_handle);
                        
        //? 3rd order solution; u_3 = u_2 + 2 phi_3(J(u) dt) R(a) dt
        axpby(1.0, u_exprb2, 2.0, u_nl_3, u_exprb3, N, GPU);
    }
}