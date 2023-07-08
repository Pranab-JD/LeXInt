#pragma once

//? ----------------------------------------------------------
//?
//? Description:
//?     A pleothera of kernels are defined here that
//?     are used throughout the code.
//?
//? ----------------------------------------------------------


#include "error_check.hpp"
#include "Timer.hpp"

#ifdef __CUDACC__
    #include <cublas_v2.h>
    #include <cuda_runtime.h>
    #include <cuda.h>
#endif

struct GPU_handle
{
    #ifdef __CUDACC__
        cublasHandle_t cublas_handle;
    #endif

    GPU_handle()
    {
        #ifdef __CUDACC__
            cublasCreate_v2(&cublas_handle);
        #endif
    }

    ~GPU_handle()
    {
        #ifdef __CUDACC__
            cublasDestroy(cublas_handle);
        #endif
    }
};

namespace LeXInt
{
    #ifdef __CUDACC__

    //? Set x = y
    __global__ void copy_CUDA(double *x, double *y, size_t N)                    
    {
        int ii = blockDim.x * blockIdx.x + threadIdx.x;

        if(ii < N)
        {
            y[ii] = x[ii];
        }
    }

    //? ones(y) = (y[0:N] =) 1.0
    __global__ void ones_CUDA(double *x, size_t N)                    
    {
        int ii = blockDim.x * blockIdx.x + threadIdx.x;

        if(ii < N)
        {
            x[ii] = 1.0;
        }
    }

    //? ones(y) = (y[0:N] =) 1.0
    __global__ void eigen_ones_CUDA(double *x, size_t N)                    
    {
        int ii = blockDim.x * blockIdx.x + threadIdx.x;

        if(ii < N)
        {
            x[ii] = 1.0;
        }

        x[N] = 2.5;
    }

    //? y = ax
    __global__ void axpby_CUDA(double a, double *x, 
                                         double *y, size_t N)                    
    {
        int ii = blockDim.x * blockIdx.x + threadIdx.x;

        if(ii < N)
        {
            y[ii] = (a * x[ii]);
        }
    }

    //? z = ax + by
    __global__ void axpby_CUDA(double a, double *x, 
                               double b, double *y, 
                                         double *z, size_t N)
    {
        int ii = blockDim.x * blockIdx.x + threadIdx.x;

        if(ii < N) 
        {
            z[ii] = (a * x[ii]) + (b * y[ii]);
        }
    }

    //? w = ax + by + cz
    __global__ void axpby_CUDA(double a, double *x, 
                               double b, double *y,
                               double c, double *z, 
                                         double *w, size_t N)
    {
        int ii = blockDim.x * blockIdx.x + threadIdx.x;

        if(ii < N) 
        {
            w[ii] = (a * x[ii]) + (b * y[ii]) + (c * z[ii]);
        }
    }

    //? v = ax + by + cz + dw
    __global__ void axpby_CUDA(double a, double *x,
                               double b, double *y,
                               double c, double *z,
                               double d, double *w,
                                         double *v, size_t N)
    {
        int ii = blockDim.x * blockIdx.x + threadIdx.x;

        if(ii < N) 
        {
            v[ii] = (a * x[ii]) + (b * y[ii]) + (c * z[ii]) + (d * w[ii]);
        }
    }

    #endif
}