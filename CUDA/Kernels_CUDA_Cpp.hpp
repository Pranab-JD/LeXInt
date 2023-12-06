#pragma once

#include <iostream>

//? ----------------------------------------------------------
//?
//? Description:
//?     Kernels and functions are "unified" for proper usage
//?     depending on whether GPU support is activated or not. 
//?
//? ----------------------------------------------------------

#include "Kernels.hpp"
#include "functions.hpp"

namespace LeXInt
{
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

            ::std::cout << "Error. Compiled with gcc, not nvcc." << ::std::endl;
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

    //? Set x = y
    void copy(double *x, double *y, size_t N, bool GPU)
    {
        if (GPU == true)
        {
            #ifdef __CUDACC__

            //* CUDA
            copy_CUDA<<<(N/128) + 1, 128>>>(x, y, N);

            #else
            ::std::cout << "Error. Compiled with gcc, not nvcc." << ::std::endl;
            exit(1);
            #endif
        }
        else
        {
            //* C++
            copy_Cpp(x, y, N);
        }
    }

    //? ones(y) = (y[0:N] =) 1.0
    void ones(double *x, size_t N, bool GPU)
    {
        if (GPU == true)
        {
            #ifdef __CUDACC__

            //* CUDA
            ones_CUDA<<<(N/128) + 1, 128>>>(x, N);

            #else
            ::std::cout << "Error. Compiled with gcc, not nvcc." << ::std::endl;
            exit(1);
            #endif
        }
        else
        {
            //* C++
            ones_Cpp(x, N);
        }
    }

    //? ones(y) = (y[0:N] =) 1.0
    void eigen_ones(double *x, size_t N, bool GPU)
    {
        if (GPU == true)
        {
            #ifdef __CUDACC__

            //* CUDA
            eigen_ones_CUDA<<<(N/128) + 1, 128>>>(x, N);

            #else
            ::std::cout << "Error. Compiled with gcc, not nvcc." << ::std::endl;
            exit(1);
            #endif
        }
        else
        {
            //* C++
            eigen_ones_Cpp(x, N);
        }
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
            ::std::cout << "Error. Compiled with gcc, not nvcc." << ::std::endl;
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
            ::std::cout << "Error. Compiled with gcc, not nvcc." << ::std::endl;
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
            ::std::cout << "Error. Compiled with gcc, not nvcc." << ::std::endl;
            exit(1);
            #endif
        }
        else
        {
            //* C++
            axpby_Cpp(a, x, b, y, c, z, d, w, v, N);
        }
    }
}