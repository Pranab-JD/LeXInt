#pragma once

#include "Kernels_CUDA_Cpp.hpp"
#include "Jacobian_vector.hpp"

namespace LeXInt
{
    //! Power Iterations
    template <typename rhs>
    void Power_iterations(rhs& RHS,                     //? RHS function
                          double* u,                    //? Input state variable(s)
                          size_t N,                     //? Number of grid points
                          double& largest_eigenvalue,   //? Largest eigenvalue (output)
                          double* auxillary,            //? Internal auxillary variables (Jv)
                          bool GPU,                     //? false (0) --> CPU; true (1) --> GPU
                          GPU_handle& cublas_handle     //? CuBLAS handle
                          )
    {
        double tol = 0.02;                              //? 2% tolerance
        double eigenvalue_ii = 0.0;                     //? Eigenvalue at ii
        double eigenvalue_ii_1 = 0.0;                   //? Eigenvalue at ii-1
        int niters = 1000;                              //? Max. number of iterations

        //? Allocate memory for internal vectors
        double* init_vector = &auxillary[0];
        double* eigenvector = &auxillary[N];
        double* auxillary_Jv = &auxillary[2*N];

        //? Set initial estimate of eigenvector = 1.0
        ones(init_vector, N, GPU);

        //? Iterate untill convergence is reached
        for (int ii = 0; ii < niters; ii++)
        {
            //? Compute new eigenvector
            Jacobian_vector(RHS, u, init_vector, eigenvector, auxillary_Jv, N, GPU, cublas_handle);

            //? Norm of eigenvector = eigenvalue
            eigenvalue_ii = l2norm(eigenvector, N, GPU, cublas_handle);

            //? Normalize eigenvector to eigenvalue; new estimate of eigenvector
            axpby(1.0/eigenvalue_ii, eigenvector, init_vector, N, GPU);

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
    }
}