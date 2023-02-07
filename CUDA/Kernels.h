#ifndef KERNEL_CUH_
#define KERNEL_CUH_

#include "cuda_runtime.h"
#include <math.h>

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

#endif
