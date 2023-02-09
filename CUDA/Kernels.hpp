#pragma once

#include <math.h>
#include "cuda_runtime.h"
#include "cublas_v2.h"


//? y = ax
__global__ void axpby(double a, double *x, 
                                double *y, int N)                    
{
    int ii = blockDim.x * blockIdx.x + threadIdx.x;

    if(ii < N)
    {
        y[ii] = (a * x[ii]);
    }
}

//? z = ax + by
__global__ void axpby(double a, double *x, 
                      double b, double *y, 
                                double *z, int N)
{
    int ii = blockDim.x * blockIdx.x + threadIdx.x;

    if(ii < N) 
    {
        z[ii] = (a * x[ii]) + (b * y[ii]);
    }
}

//? w = ax + by + cz
__global__ void axpby(double a, double *x, 
                      double b, double *y,
                      double c, double *z, 
                                double *w, int N)
{
    int ii = blockDim.x * blockIdx.x + threadIdx.x;

    if(ii < N) 
    {
        w[ii] = (a * x[ii]) + (b * y[ii]) + (c * z[ii]);
    }
}

//? v = ax + by + cz + dw
__global__ void axpby(double a, double *x, 
                      double b, double *y,
                      double c, double *z, 
                      double d, double *w, 
                                double *v, int N)
{
    int ii = blockDim.x * blockIdx.x + threadIdx.x;

    if(ii < N) 
    {
        v[ii] = (a * x[ii]) + (b * y[ii]) + (c * z[ii]) + (d * w[ii]);
    }
}


template <typename rhs>
void Jacobian_vector(rhs& RHS, double* vector_x, double* vector_y, double* Jac_vec, size_t N, cublasHandle_t& cublas_handle)
{
    //! Create a struct for internal vectors
    double* device_rhs_u; cudaMalloc(&device_rhs_u, N * sizeof(double));
    RHS(vector_x, device_rhs_u); 

    double rhs_norm;
    cublasDnrm2(cublas_handle, N, device_rhs_u, 1, &rhs_norm);
    double epsilon = 1e-7*rhs_norm;
    
    //? x_eps1 = x + epsilon*y
    double* device_vector_x_eps_1; cudaMalloc(&device_vector_x_eps_1, N * sizeof(double));
    axpby<<<(N/128) + 1, 128>>>(1.0, vector_x, epsilon, vector_y, device_vector_x_eps_1, N); 

    //? x_eps2 = x - epsilon*y
    double* device_vector_x_eps_2; cudaMalloc(&device_vector_x_eps_2, N * sizeof(double));
    axpby<<<(N/128) + 1, 128>>>(1.0, vector_x, -epsilon, vector_y, device_vector_x_eps_2, N); 

    //? RHS(x + epsilon*y)
    double* device_rhs_u_eps_1; cudaMalloc(&device_rhs_u_eps_1, N * sizeof(double));
    RHS(device_vector_x_eps_1, device_rhs_u_eps_1);

    //? RHS(x - epsilon*y)
    double* device_rhs_u_eps_2; cudaMalloc(&device_rhs_u_eps_2, N * sizeof(double));
    RHS(device_vector_x_eps_2, device_rhs_u_eps_2);

    //? J(u) * y = (RHS(x + epsilon*y) - RHS(x - epsilon*y))/(2*epsilon)
    axpby<<<(N/128) + 1, 128>>>(1.0/(2.0*epsilon), device_rhs_u_eps_1, -1.0/(2.0*epsilon), device_rhs_u_eps_2, Jac_vec, N);

    //? Deallocate memory
    cudaFree(device_rhs_u);
    cudaFree(device_vector_x_eps_1);
    cudaFree(device_vector_x_eps_2);
    cudaFree(device_rhs_u_eps_1);
    cudaFree(device_rhs_u_eps_2);
}