#pragma once

#include "../Leja.hpp"
#include "../Phi_functions.hpp"

namespace LeXInt
{
    //? Phi functions interpolated on real Leja points
    template <typename rhs>
    void Ros_Eu(rhs& RHS,                   //? RHS function
                double* u,                  //? Input state variable(s)
                double* u_exprb2,           //? Output state variable(s)
                double* auxiliary_expint,   //? Internal auxiliary variables
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

        //! u, u_exprb2, auxiliary_expint, auxiliary_Leja, and auxiliary_NL
        //! are device vectors if GPU support is activated.

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

        //? Assign names and variables
        double* f_u = &auxiliary_expint[0];

        //? RHS evaluated at 'u' multiplied by 'dt'; f_u = RHS(u)*dt
        RHS(u, f_u);
        axpby(dt, f_u, f_u, N, GPU);

        //? Interpolation of RHS(u) at 1; phi_1(J(u) dt) f(u) dt
        real_Leja_phi(RHS, u, f_u, u_exprb2, auxiliary_Leja, N, {1.0}, 
                      phi_1, Leja_X, c, Gamma, rtol, atol, dt, iters, GPU, cublas_handle);

        //? 2nd order solution; u_2 = u + phi_1(J(u) dt) f(u) dt
        axpby(1.0, u, 1.0, u_exprb2, u_exprb2, N, GPU);
    }
}