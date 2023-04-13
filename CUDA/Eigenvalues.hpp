#pragma once

#include <iostream>
#include <vector>

#include "functions.hpp"

using namespace std;

//? -----------------------------------------------------------------
//?
//? Description:
//?     Use Gershgorin's disk theorem if matrix is explcitly avilable.
//?     Else, use power iterations.
//?
//?     NOTE: Largest real eigenvalue has to be NEGATIVE!
//?
//? -----------------------------------------------------------------

//! ======================================================================================== !//

//! Gershgorin's Disk Theorem
template <typename T>
double Gershgorin(T A,      //? N x N matrix A 
                  int N     //? No. of rows or columns
                  )
{
    vector<double> eigen_list(N);
        
    for (int ii = 0; ii < N; ii++)
    {
        double eigenvalue = 0;

        for (int jj = 0; jj < N; jj++)
        {
            eigenvalue = eigenvalue + abs(A[ii][jj]);
        }

        eigen_list[ii] = eigenvalue;
    }

    //! Returns the largest eigenvalue in magnitude
    return *max_element(begin(eigen_list), end(eigen_list));
}

//! ======================================================================================== !//

//! Power Iterations
template <typename rhs>
void Power_iterations(rhs& RHS,                     //? RHS function
                      double* u,                    //? State variable(s)
                      size_t N,                     //? Number of grid points
                      double& largest_eigenvalue,   //? Largest eigenvalue (output)
                      cublasHandle_t &cublas_handle
                      )
{
    double tol = 0.02;                              //? 2% tolerance
    double eigenvalue_ii = 0.0;                     //? Eigenvalue at ii
    double eigenvalue_ii_1 = 0.0;                   //? Eigenvalue at ii-1
    int niters = 1000;                              //? Max. number of iterations

    double* device_init_vector; cudaMalloc(&device_init_vector, N * sizeof(double));
    double* device_eigenvector; cudaMalloc(&device_eigenvector, N * sizeof(double));

    for (int ii = 0; ii < niters; ii++)
    {
        //? Compute new eigenvector
        Jacobian_vector(RHS, u, device_init_vector, device_eigenvector, N, cublas_handle);

        //? Norm of eigenvector = eigenvalue
        cublasDnrm2(cublas_handle, N, device_eigenvector, 1, &eigenvalue_ii);

        //? Normalize eigenvector to eigenvalue; new estimate of eigenvector
        axpby<<<(N/128) + 1, 128>>>(1.0/eigenvalue_ii, device_eigenvector, device_init_vector, N);

        //? Check convergence for eigenvalues (eigenvalues converge faster than eigenvectors)
        if (abs(eigenvalue_ii - eigenvalue_ii_1) <= tol * eigenvalue_ii)
        {
            //! Returns the largest eigenvalue in magnitude (needs to multiplied to a safety factor)
            largest_eigenvalue = eigenvalue_ii;
            break;
        }

        //? This value becomes the previous one
        eigenvalue_ii_1 = eigenvalue_ii;
    }

    cudaFree(device_init_vector);
    cudaFree(device_eigenvector);
}

//! ======================================================================================== !//
