#pragma once

#include "Timer.hpp"
#include "Kernels_CUDA_Cpp.hpp"

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
        //! This function has 16 vector reads and writes.

        //? Jac_vec = rhs(u)
        RHS(u, Jac_vec);

        double rhs_norm = l2norm(Jac_vec, N, GPU, cublas_handle);
        double epsilon = 1e-7*rhs_norm;
        
        //? Jac_vec = u + epsilon*y
        axpby(1.0, u, epsilon, y, Jac_vec, N, GPU); 

        //? rhs_u_eps_1 = RHS(u + epsilon*y)
        double* rhs_u_eps_1 = &auxiliary_Jv[0];
        RHS(Jac_vec, rhs_u_eps_1);

        //? Jac_vec = u - epsilon*y
        axpby(1.0, u, -epsilon, y, Jac_vec, N, GPU); 

        //? rhs_u_eps_2 = RHS(u - epsilon*y)
        double* rhs_u_eps_2 = &auxiliary_Jv[N];
        RHS(Jac_vec, rhs_u_eps_2);

        //? Jac_vec = J(u) * y = (RHS(u + epsilon*y) - RHS(u - epsilon*y))/(2*epsilon)
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
        //! This function has 21 vector reads and writes.

        //? J(u) * y = (F(u + epsilon*y) - F(u - epsilon*y))/(2*epsilon)
        double* Linear_y = &auxiliary_Jv[0];
        double* Jv = &auxiliary_Jv[N];
        Jacobian_vector(RHS, u, y, Linear_y, Jv, N, GPU, cublas_handle);

        //? f(y)
        double* rhs_y = &auxiliary_Jv[3*N];
        RHS(y, rhs_y);

        //? F(y) = f(y) - (J(u) * y)
        axpby(1.0, rhs_y, -1.0, Linear_y, Nonlinear_y, N, GPU);
    }
}