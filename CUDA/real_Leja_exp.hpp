#pragma once

#include <iostream>
#include <iomanip>
#include <vector>
#include <cmath>
#include <functional>

#include "functions.hpp"
#include "Divided_Differences.hpp"

//? CUDA 
#include "Kernels.h"
#include "cublas_v2.h"
#include "error_check.hpp"

using namespace std;

//? Matrix exponential interpolated on real Leja points
template <typename state, typename rhs>
state real_Leja_exp(rhs& RHS,                       //? RHS function
                    state& u,                       //? State variable(s)
                    int N,                          //? Number of grid points
                    vector<double>& Leja_X,         //? Array of Leja points
                    double c,                       //? Shifting factor
                    double Gamma,                   //? Scaling factor
                    double tol,                     //? Tolerance (normalised desired accuracy)
                    double dt,                      //? Step size
                    cublasHandle_t &cublas_handle   //? CuBLAS handle
                    )
{
    //* -------------------------------------------------------------------------

    //* Computes the polynomial interpolation of matrix exponential applied to 'u' at real Leja points.
    //*
    //*    Returns
    //*    ----------
    //*    polynomial              : state
    //*                                Polynomial interpolation of 'u' multiplied 
    //*                                by the matrix exponential at real Leja points

    //* -------------------------------------------------------------------------

    size_t N_size = N * sizeof(double);                 //? Memory size = # elements * size of each element
    int threads_per_block = 32;
    int blocks_per_grid = ceil(N/threads_per_block);
    
    double y_error;                                     //? Error incurred at every iteration
    double poly_norm;                                   //? Norm of the polynomial
    int max_Leja_pts = Leja_X.size();                   //? Max. # of Leja points
    
    state y(u);                                         //? To avoid changing 'u'
    state Jacobian_function(N);                         //? Jacobian-vector product
    state polynomial(N);                                //? Initialise the polynomial

    //? Allocate memory on device
    double *device_y; cudaMalloc(&device_y, N_size);
    double *device_polynomial; cudaMalloc(&device_polynomial, N_size);
    double *device_Jacobian_function; cudaMalloc(&device_Jacobian_function, N_size);
    
    //* Copy 'y' from host to device
    cudaMemcpy(device_y, &y[0], N_size, cudaMemcpyHostToDevice);

    //* Matrix exponential (scaled and shifted)
    vector<double> matrix_exponential(max_Leja_pts);

    for (int ii = 0; ii < max_Leja_pts; ii++)
    {
        matrix_exponential[ii] = exp(dt * (c + (Gamma * Leja_X[ii])));
    }

    //* Compute polynomial coefficients
    vector<double> coeffs = Divided_Differences(Leja_X, matrix_exponential);

    //* Form the polynomial (first term): polynomial = coeffs[0] * y
    axpby<<<blocks_per_grid, threads_per_block>>>(coeffs[0], device_y, device_polynomial, N);

    //? Iterate until converges
    for (int nn = 1; nn < max_Leja_pts - 1; nn++)
    {
        //! REMOVE! Get RHS fnction on GPU
        cudaMemcpy(&y[0], device_y, N_size, cudaMemcpyDeviceToHost);

        //* Compute numerical Jacobian (for linear eqs., this is the RHS evaluation at y)
        Jacobian_function = RHS(y);

        //! REMOVE! device_Jacobian_function is to be computed directly on GPU
        cudaMemcpy(device_Jacobian_function, &Jacobian_function[0], N_size, cudaMemcpyHostToDevice);

        //* y = y * ((z - c)/Gamma - Leja_X)
        axpby<<<blocks_per_grid, threads_per_block>>>(1./Gamma, device_Jacobian_function, (-c/Gamma - Leja_X[nn - 1]), device_y, device_y, N);

        //* Error estimate for 'y': y_error = |coeffs[nn]| ||y||
        cublasDnrm2(cublas_handle, N, device_y, 1, &y_error);
        y_error = abs(coeffs[nn]) * y_error;

        //* Add the new term to the polynomial (polynomial = polynomial + (coeffs[nn] * y))
        axpby<<<blocks_per_grid, threads_per_block>>>(coeffs[nn], device_y, 1.0, device_polynomial, device_polynomial, N);

        //* Compute norm of the polynomial
        cublasDnrm2(cublas_handle, N, device_polynomial, 1, &poly_norm);

        //? If new term to be added < tol, break loop; safety factor = 0.1
        if (y_error < 0.1*tol*poly_norm)
        {
            cout << "Converged! Iterations: " << nn << endl;
            break;
        }

        //! Warning flags
        if (nn == max_Leja_pts - 2)
        {
            cout << "Warning!! Max. # of Leja points reached without convergence!! Max. Leja points currently set to " << max_Leja_pts << endl;
            cout << "Try increasing the number of Leja points. Max available: 10000." << endl;
            break;
        }
    }

    return polynomial;
}