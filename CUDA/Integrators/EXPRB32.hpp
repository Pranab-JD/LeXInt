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
                 double* auxiliary_expint,   //? Internal auxiliary variables (EXPRB32)
                 double* auxiliary_Leja,     //? Internal auxiliary variables (Leja and NL remainders)
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

        //! u, u_exprb2, u_exprb3, auxiliary_expint, auxiliary_Leja, and auxiliary_NL
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

        //? Assign required variables
        double* temp_vec = &auxiliary_expint[0];

        //? RHS evaluated at 'u' multiplied by 'dt'; u_exprb3 = f(u)
        RHS(u, u_exprb3);
        axpby(dt, u_exprb3, u_exprb3, N, GPU);
        
        //? Interpolation of RHS(u) at 1; u_exprb2 = phi_1(J(u) dt) f(u) dt
        real_Leja_phi(RHS, u, u_exprb3, u_exprb2, auxiliary_Leja, N, {1.0}, 
                        phi_1, Leja_X, c, Gamma, tol, dt, iters_1, GPU, cublas_handle);

        //? Internal stage 1; 2nd order solution; u_2 = u + phi_1(J(u) dt) f(u) dt
        axpby(1.0, u, 1.0, u_exprb2, u_exprb2, N, GPU);

        //? Difference of nonlinear remainders at u_exprb2; u_exprb3 = (u_exprb3 - temp_vec_1)*dt 
        Nonlinear_remainder(RHS, u, u,        temp_vec, auxiliary_Leja, N, GPU, cublas_handle);
        Nonlinear_remainder(RHS, u, u_exprb2, u_exprb3, auxiliary_Leja, N, GPU, cublas_handle);
        axpby(dt, u_exprb3, -dt, temp_vec, u_exprb3, N, GPU);

        //? Final nonlinear stage; temp_vec_1 = phi_3(J(u) dt) R(a) dt
        real_Leja_phi(RHS, u, u_exprb3, temp_vec, auxiliary_Leja, N, {1.0}, 
                        phi_3, Leja_X, c, Gamma, tol, dt, iters_2, GPU, cublas_handle);
                        
        //? 3rd order solution; u_3 = u_2 + 2 phi_3(J(u) dt) R(a) dt
        axpby(1.0, u_exprb2, 2.0, temp_vec, u_exprb3, N, GPU);

        //? Total number of Leja iterations
        iters = iters_1 + iters_2;
    }
}