#pragma once

#include "../Leja.hpp"
#include "../Phi_functions.hpp"

namespace LeXInt
{
    //? Phi functions interpolated on real Leja points
    template <typename rhs>
    void Ros_Eu(rhs& RHS,                       //? RHS function
                double* u,                      //? Input state variable(s)
                double* u_exprb2,               //? Output state variable(s)
                double* auxiliary_expint,       //? Internal auxiliary variables
                double* auxiliary_Leja,         //? Internal auxiliary variables (Leja)
                size_t N,                       //? Number of grid points
                std::vector<double>& Leja_X,    //? Array of Leja points
                double c,                       //? Shifting factor
                double Gamma,                   //? Scaling factor
                double tol,                     //? Tolerance (normalised desired accuracy)
                double dt,                      //? Step size
                int substeps,                   //? Initial guess for substeps/substeps used
                int& iters,                     //? # of iterations needed to converge (iteration variable)
                bool GPU,                       //? false (0) --> CPU; true (1) --> GPU
                GPU_handle& cublas_handle       //? CuBLAS handle
                )
    {
        //* -------------------------------------------------------------------------

        //! u, u_exprb2, auxiliary_expint, and auxiliary_Leja
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
        double* zero_vec = &auxiliary_expint[1];

        //? RHS evaluated at 'u' multiplied by 'dt'; f_u = RHS(u)*dt
        RHS(u, f_u);
        axpby(dt, f_u, f_u, N, GPU);

        std::vector<double> coeffs_1; coeffs_1.push_back(1.0);
        //? Interpolation of RHS(u) at 1; phi_1(J(u) dt) f(u) dt
        real_Leja_phi(RHS, u, f_u, u_exprb2, auxiliary_Leja, N, coeffs_1, 
                        phi_1, Leja_X, c, Gamma, tol, dt, iters, GPU, cublas_handle);

        //? 2nd order solution; u_2 = u + phi_1(J(u) dt) f(u) dt
        axpby(1.0, u, 1.0, u_exprb2, u_exprb2, N, GPU);

        linear_phi(RHS, {zero_vec, f_u}, u_exprb2, auxiliary_Leja, N, dt, substeps, 1.0, Leja_X, c, Gamma, tol, iters, GPU, cublas_handle);

    }
}