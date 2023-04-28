#pragma once

#include <iostream>
#include <iomanip>
#include <vector>
#include <cmath>
#include <functional>

#include "Divided_Differences.hpp"

//? CUDA 
#include "cublas_v2.h"
#include "error_check.hpp"
#include "Leja_GPU.hpp"

using namespace std;

//? Matrix exponential interpolated on real Leja points
template <typename rhs>
real_Leja_exp(rhs& RHS,                       //? RHS function
                                    double* device_u,               //? Input state variable(s)
                                    double* device_polynomial,      //? Output state variable(s)
                                    vector<double>& Leja_X,         //? Array of Leja points
                                    double c,                       //? Shifting factor
                                    double Gamma,                   //? Scaling factor
                                    double tol,                     //? Tolerance (normalised desired accuracy)
                                    double dt                       //? Step size
                                    )
{
    //* -------------------------------------------------------------------------

    //* Computes the polynomial interpolation of matrix exponential applied to 'u' at real Leja points.
    //*
    //*    Returns
    //*    ----------
    //*    device_polynomial        : double*
    //*                                     Polynomial interpolation of 'u' multiplied 
    //*                                     by the matrix exponential at real Leja points

    //* -------------------------------------------------------------------------
    
    double y_error;
    double poly_norm;                                               //? Norm of the polynomial
    int max_Leja_pts = Leja_X.size();                               //? Max. # of Leja points
    cudaMemcpy(device_y, device_u, N_size, cudaMemcpyDefault);      //? To avoid changing 'u'

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
        //* Compute numerical Jacobian (for linear eqs., this is the RHS evaluation at y)
        RHS(device_y, device_Jacobian_function);

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
        if (y_error < (0.1*tol*poly_norm) + tol)
        {
            cout << "Converged! Iterations: " << nn << endl;
            break;
        }

        //! Warning flags
        if (nn == max_Leja_pts - 2)
        {
            cout << "Warning!! Max. number of Leja points reached without convergence!! Max. Leja points currently set to " << max_Leja_pts << endl;
            cout << "Try increasing the number of Leja points. Max available: 10000." << endl;
            break;
        }
    }
}