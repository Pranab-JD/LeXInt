#pragma once

#include <iostream>
#include <iomanip>
#include <vector>
#include <cmath>
#include <functional>

#include "functions.hpp"
#include "Divided_Differences.hpp"

//? CUDA 
#include "Kernels.hpp"
#include "cublas_v2.h"
#include "error_check.hpp"

using namespace std;

template <typename state, typename rhs>
struct Leja_GPU
{
    int N;
    size_t N_size;                                   //? Memory size = # elements * size of each element
    int threads_per_block = 128;
    int blocks_per_grid;

    state y;                                         //? To avoid changing 'u'
    state polynomial;                                //? Initialise the polynomial
    state Jacobian_function;                         //? Jacobian-vector product
    
    //? Allocate memory on device
    double *device_y;
    double *device_polynomial;
    double *device_Jacobian_function;

    cublasHandle_t &cublas_handle;                    //? CuBLAS handle

    //! Constructor
    Leja_GPU(int _N, cublasHandle_t &_cublas_handle) : 
                            N{_N}, 
                            N_size{N * sizeof(double)}, 
                            blocks_per_grid{(N/threads_per_block) + 1},
                            cublas_handle{_cublas_handle}

    {
        y.resize(N);
        polynomial.resize(N);
        Jacobian_function.resize(N);
        
        //? Allocate memory on device
        cudaMalloc(&device_y, N_size);
        cudaMalloc(&device_polynomial, N_size);
        cudaMalloc(&device_Jacobian_function, N_size);
    }

    //! Destructor
    ~Leja_GPU()
    {
        //? Deallocate memory
        cudaFree(device_y);
        cudaFree(device_polynomial);
        cudaFree(device_Jacobian_function);
    }

    //? Matrix exponential interpolated on real Leja points
    void real_Leja_exp(rhs& RHS,                                   //? RHS function
                        double* u,                                   //? State variable(s)
                        double* u_output,
                        vector<double>& Leja_X,                     //? Array of Leja points
                        double c,                                   //? Shifting factor
                        double Gamma,                               //? Scaling factor
                        double tol,                                 //? Tolerance (normalised desired accuracy)
                        double dt                                   //? Step size
                        );

    //? Phi function interpolated on real Leja points
    state real_Leja_phi_nl(rhs& RHS,                                //? RHS function
                           state& u,                                //? State variable(s)
                           state& interp_vector,                    //? Vector multiplied to phi function
                           double (* phi_function) (double),        //? Phi function (typically phi_1)
                           vector<double>& Leja_X,                  //? Array of Leja points
                           double c,                                //? Shifting factor
                           double Gamma,                            //? Scaling factor
                           double tol,                              //? Tolerance (normalised desired accuracy)
                           double dt                                //? Step size
                           );

    //? Phi function interpolated on real Leja points (Vertical implementation)
    vector<state> real_Leja_phi(rhs& RHS,                           //? RHS function
                                state& u,                           //? State variable(s)
                                state& interp_vector,               //? Vector multiplied to phi function
                                vector<double> integrator_coeffs,   //? Coefficients of the integrator
                                double (* phi_function) (double),   //? Phi function (typically phi_1)
                                vector<double>& Leja_X,             //? Array of Leja points
                                double c,                           //? Shifting factor
                                double Gamma,                       //? Scaling factor
                                double tol,                         //? Tolerance (normalised desired accuracy)
                                double dt                           //? Step size
                                );

};
