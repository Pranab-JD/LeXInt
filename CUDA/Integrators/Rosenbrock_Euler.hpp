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
    void Ros_Eu(rhs& RHS,                   //? RHS function
                double* u,                  //? Input state variable(s)
                double* u_exprb2,           //? Output state variable(s)
                double* auxillary_expint,   //? Internal auxillary variables
                double* auxillary_Leja,     //? Internal auxillary variables (Leja)
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

        //! u, u_exprb2, auxillary_expint, auxillary_Leja, and auxillary_NL
        //! are device vectors if GPU support is activated.

        //*
        //*    Returns
        //*    ----------
        //*     u_exprb2                : double*
        //*                                 2nd order solution after time dt
        //*
        //*
        //*    Reference:
        //*         D. A. Pope, An exponential method of numerical integration of ordinary differential equations, Commun. ACM 6 (8) (1963) 491-493.
        //*         doi:10.1145/366707.367592

        //* -------------------------------------------------------------------------

        //? RHS evaluated at 'u' multiplied by 'dt'
        double* rhs_u = &auxillary_expint[0];
        RHS(u, rhs_u);
        axpby(dt, rhs_u, rhs_u, N, GPU);

        //? Interpolation of RHS(u) at 1; phi_1(J(u) dt) f(u) dt
        double* u_flux = &auxillary_expint[N];
        real_Leja_phi(RHS, u, rhs_u, u_flux, auxillary_Leja, N, {1.0}, 
                        phi_1, Leja_X, c, Gamma, tol, dt, GPU, cublas_handle);

        //? 2nd order solution; u_2 = u + phi_1(J(u) dt) f(u) dt
        axpby(1.0, u, 1.0, u_flux, u_exprb2, N, GPU);
    }
}