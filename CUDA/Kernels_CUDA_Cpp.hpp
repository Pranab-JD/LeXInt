#pragma once

#include "Kernels.hpp"
#include "functions.hpp"

//?l2 norm
double l2norm(double *x, size_t N, bool GPU, GPU_handle& cublas_handle)
{
    double norm;

    if (GPU == true)
    {
        #ifdef __CUDACC__
            //* CUDA
            cublasDnrm2(cublas_handle.cublas_handle, N, x, 1, &norm);
        #else

        cout << "Error. Compiled with gcc, not nvcc." << endl;
        exit(1);
        #endif
    }
    else
    {
        //* C++
        norm = l2norm_Cpp(x, N);
    }

    return norm;
}

//? y = ax
void axpby(double a, double *x, 
                     double *y, size_t N, bool GPU)
{
    if (GPU == true)
    {
        #ifdef __CUDACC__

        //* CUDA
        axpby_CUDA<<<(N/128) + 1, 128>>>(a, x, y, N);

        #else
        cout << "Error. Compiled with gcc, not nvcc." << endl;
        exit(1);
        #endif
    }
    else
    {
        //* C++
        axpby_Cpp(a, x, y, N);
    }
}


//? z = ax + by
void axpby(double a, double *x, 
           double b, double *y, 
                     double *z, size_t N, bool GPU)
{
    if (GPU == true)
    {
        #ifdef __CUDACC__

        //* CUDA
        axpby_CUDA<<<(N/128) + 1, 128>>>(a, x, b, y, z, N);

        #else
        cout << "Error. Compiled with gcc, not nvcc." << endl;
        exit(1);
        #endif
    }
    else
    {
        //* C++
        axpby_Cpp(a, x, b, y, z, N);
    }
}

//? w = ax + by + cz
void axpby(double a, double *x, 
           double b, double *y, 
           double c, double *z, 
                     double *w, size_t N, bool GPU)
{
    if (GPU == true)
    {
        #ifdef __CUDACC__

        //* CUDA
        axpby_CUDA<<<(N/128) + 1, 128>>>(a, x, b, y, c, z, w, N);

        #else
        cout << "Error. Compiled with gcc, not nvcc." << endl;
        exit(1);
        #endif
    }
    else
    {
        //* C++
        axpby_Cpp(a, x, b, y, c, z, w, N);
    }
}

//? v = ax + by + cz + dw
void axpby(double a, double *x, 
           double b, double *y, 
           double c, double *z, 
           double d, double *w, 
                     double *v, size_t N, bool GPU)
{
    if (GPU == true)
    {
        #ifdef __CUDACC__

        //* CUDA
        axpby_CUDA<<<(N/128) + 1, 128>>>(a, x, b, y, c, z, d, w, v, N);

        #else
        cout << "Error. Compiled with gcc, not nvcc." << endl;
        exit(1);
        #endif
    }
    else
    {
        //* C++
        axpby_Cpp(a, x, b, y, c, z, d, w, v, N);
    }
}