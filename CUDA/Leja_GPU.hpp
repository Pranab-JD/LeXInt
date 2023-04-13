#pragma once

#include <iostream>
#include <iomanip>
#include <vector>
#include <cmath>
#include <functional>

#include "Divided_Differences.hpp"

//? CUDA 
#include "Kernels.hpp"
#include "cublas_v2.h"
#include "error_check.hpp"

using namespace std;

template <typename rhs>
struct Leja_GPU
{
    int N;
    size_t N_size;                                   //? Memory size = # elements * size of each element
    int threads_per_block = 128;
    int blocks_per_grid;
    
    //? Allocate memory on device
    double* device_y;
    double* device_Jacobian_function;

    struct timespec total_start, total_finish, total_elapsed;

    cublasHandle_t &cublas_handle;                  //? CuBLAS handle

    //! Constructor
    Leja_GPU(int _N, cublasHandle_t &_cublas_handle) : 
                                        N{_N}, 
                                        N_size{N * sizeof(double)}, 
                                        blocks_per_grid{(N/threads_per_block) + 1},
                                        cublas_handle{_cublas_handle}

    {
        //? Allocate memory on device
        cudaMalloc(&device_y, N_size);
        cudaMalloc(&device_Jacobian_function, N_size);
    }

    //! Destructor
    ~Leja_GPU()
    {
        //? Deallocate memory
        cudaFree(&device_y);
        cudaFree(&device_Jacobian_function);
    }

    //? Matrix exponential interpolated on real Leja points
    void real_Leja_exp(rhs& RHS,                                   //? RHS function
                       double* device_u,                           //? Input state variable(s)
                       double* device_polynomial,                  //? Output state variable(s)
                       vector<double>& Leja_X,                     //? Array of Leja points
                       double c,                                   //? Shifting factor
                       double Gamma,                               //? Scaling factor
                       double tol,                                 //? Tolerance (normalised desired accuracy)
                       double dt                                   //? Step size
                       );

    //? Phi function interpolated on real Leja points
    void real_Leja_phi_nl(rhs& RHS,                                //? RHS function
                          double* device_interp_vector,            //? Input vector multiplied to phi function
                          double* device_polynomial,               //? Output vector multiplied to phi function
                          double (* phi_function) (double),        //? Phi function (typically phi_1)
                          vector<double>& Leja_X,                  //? Array of Leja points
                          double c,                                //? Shifting factor
                          double Gamma,                            //? Scaling factor
                          double tol,                              //? Tolerance (normalised desired accuracy)
                          double dt                                //? Step size
                          );

    //? Phi function interpolated on real Leja points (Vertical implementation)
    void real_Leja_phi(rhs& RHS,                                   //? RHS function
                       double* device_u,                           //? Input state variable(s)
                       double* device_interp_vector,               //? Input vector multiplied to phi function
                       double* device_polynomial,                  //? Output vector multiplied to phi function
                       vector<double> integrator_coeffs,           //? Coefficients of the integrator
                       double (* phi_function) (double),           //? Phi function (typically phi_1)
                       vector<double>& Leja_X,                     //? Array of Leja points
                       double c,                                   //? Shifting factor
                       double Gamma,                               //? Scaling factor
                       double tol,                                 //? Tolerance (normalised desired accuracy)
                       double dt                                   //? Step size
                       );
};
