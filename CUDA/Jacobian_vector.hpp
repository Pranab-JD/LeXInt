#pragma once

#include "Timer.hpp"
#include "Kernels_CUDA_Cpp.hpp"

//! This function has 21 vector reads and writes.

namespace LeXInt
{
    //? J(u) * y = (F(u + epsilon*y) - F(u - epsilon*y))/(2*epsilon)
    template <typename rhs>
    void Jacobian_vector(rhs& RHS,                      //? RHS function
                         double* u,                     //? Input state variable(s)
                         double* y,                     //? Vector to be multiplied to Jacobian 
                         double* Jac_vec,               //? Output Jacobian-vector product
                         double* auxiliary_Jv,          //? Internal auxiliary variables
                         size_t N,                      //? Number of grid points
                         bool GPU,                      //? false (0) --> CPU; true (1) --> GPU
                         GPU_handle& cublas_handle      //? CuBLAS handle
                         )
    {
        double* rhs_u = &auxiliary_Jv[0];
        RHS(u, rhs_u);

        double rhs_norm = l2norm(rhs_u, N, GPU, cublas_handle);
        double epsilon = 1e-7*rhs_norm;

        //? u_eps = u + epsilon*y
        double* u_eps = &auxiliary_Jv[N];
        axpby(1.0, u, epsilon, y, u_eps, N, GPU); 

        //? RHS(u + epsilon*y)
        double* rhs_u_eps_1 = &auxiliary_Jv[2*N];
        RHS(u_eps, rhs_u_eps_1);

        //? u_eps = u - epsilon*y
        axpby(1.0, u, -epsilon, y, u_eps, N, GPU); 

        //? RHS(u - epsilon*y)
        double* rhs_u_eps_2 = &auxiliary_Jv[3*N];
        RHS(u_eps, rhs_u_eps_2);

        //? J(u) * y = (RHS(u + epsilon*y) - RHS(u - epsilon*y))/(2*epsilon)
        axpby(1.0/(2.0*epsilon), rhs_u_eps_1, -1.0/(2.0*epsilon), rhs_u_eps_2, Jac_vec, N, GPU);
    }

    //? F(y) = f(y) - (J(u) * y)
    template <typename rhs>
    void Nonlinear_remainder(rhs& RHS,                      //? RHS function
                             double* u,                     //? Input state variable(s)
                             double* y,                     //? Vector to be multiplied to Jacobian 
                             double* Nonlinear_y,           //? Output nonlinear remainder       
                             double* auxiliary_Jv,          //? Internal auxiliary variables for Jacobian-vector
                             size_t N,                      //? Number of grid points
                             bool GPU,                      //? false (0) --> CPU; true (1) --> GPU
                             GPU_handle& cublas_handle      //? CuBLAS handle
                             )
    {
        //? J(u) * y = (F(u + epsilon*y) - F(u - epsilon*y))/(2*epsilon)
        double* Linear_y = &auxiliary_Jv[0];
        double* Jv = &auxiliary_Jv[N];
        Jacobian_vector(RHS, u, y, Linear_y, Jv, N, GPU, cublas_handle);

        //? f(y)
        double* rhs_y = &auxiliary_Jv[5*N];
        RHS(y, rhs_y);

        //? F(y) = f(y) - (J(u) * y)
        axpby(1.0, rhs_y, -1.0, Linear_y, Nonlinear_y, N, GPU);
    }
}