#pragma once

#include <math.h>
#include "error_check.hpp"

#include "Timer.hpp"

#ifdef __CUDACC__
    #include "cublas_v2.h"
    #include <cub/cub.cuh>
    #include "cuda_runtime.h"
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


#ifdef __CUDACC__

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