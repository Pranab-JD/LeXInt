#pragma once

#include "Timer.hpp"
#include "Kernels_CUDA_Cpp.hpp"

//? J(u) * y = (F(u + epsilon*y) - F(u - epsilon*y))/(2*epsilon)
template <typename rhs>
void Jacobian_vector(rhs& RHS, 
                     double* device_u, 
                     double* device_y, 
                     double* Jac_vec,
                     double* device_auxillary_Jv,
                     size_t N,
                     bool GPU,
                     GPU_handle& cublas_handle
                     )
{
    double* device_rhs_u = &device_auxillary_Jv[0];
    RHS(device_u, device_rhs_u); 

    double rhs_norm = l2norm(device_rhs_u, N, GPU, cublas_handle);
    double epsilon = 1e-7*rhs_norm;
    
    //? u_eps1 = u + epsilon*y
    double* device_u_eps_1 = &device_auxillary_Jv[N];
    axpby(1.0, device_u, epsilon, device_y, device_u_eps_1, N, GPU); 

    //? u_eps2 = u - epsilon*y
    double* device_u_eps_2 = &device_auxillary_Jv[2*N];
    axpby(1.0, device_u, -epsilon, device_y, device_u_eps_2, N, GPU); 

    //? RHS(u + epsilon*y)
    double* device_rhs_u_eps_1 = &device_auxillary_Jv[3*N];
    RHS(device_u_eps_1, device_rhs_u_eps_1);

    //? RHS(u - epsilon*y)
    double* device_rhs_u_eps_2 = &device_auxillary_Jv[4*N];
    RHS(device_u_eps_2, device_rhs_u_eps_2);

    //? J(u) * y = (RHS(u + epsilon*y) - RHS(u - epsilon*y))/(2*epsilon)
    axpby(1.0/(2.0*epsilon), device_rhs_u_eps_1, -1.0/(2.0*epsilon), device_rhs_u_eps_2, Jac_vec, N, GPU);
}

//? F(y) = f(y) - (J(u) * y)
template <typename rhs>
void Nonlinear_remainder(rhs& RHS, 
                         double* device_u, 
                         double* device_y, 
                         double* device_Nonlinear_y,
                         double* device_auxillary_Jv,
                         size_t N,
                         bool GPU,
                         GPU_handle& cublas_handle
                         )
{
    //? J(u) * y = (F(u + epsilon*y) - F(u - epsilon*y))/(2*epsilon)
    double* device_Linear_y = &device_auxillary_Jv[0];
    double* device_Jv = &device_auxillary_Jv[N];
    Jacobian_vector(RHS, device_u, device_y, device_Linear_y, device_Jv, N, GPU, cublas_handle);

    //? f(y)
    double* device_rhs_y = &device_auxillary_Jv[6*N];
    RHS(device_y, device_rhs_y);

    //? F(y) = f(y) - (J(u) * y)
    axpby(1.0, device_rhs_y, -1.0, device_Linear_y, device_Nonlinear_y, N, GPU);
}