#pragma once

#include<vector>
#include <functional>

#include "Leja.hpp"
#include "Timer.hpp"
#include "Kernels_CUDA_Cpp.hpp"
#include "real_Leja_linear_exp.hpp"

namespace LeXInt
{
    //! Kernel for CUDA
    void v_dot_B(double* v, std::vector<double*> B, double* vB, int N)
    {
        for(int ii = 0; ii < N; ii++ )
        {
            for (int jj = 0; jj < B.size(); jj++)
            {
                vB[ii] = vB[ii] + B[jj][ii]*v[jj+N];
            }
        }
    }

    //? A~ . v; augmented matrix applied to augmented vector
    template <typename rhs>
    void RHS_aug(rhs& RHS,                              //? RHS function
                 std::vector<double*> interp_vector,    //? Vectors applied to phi_1, phi_2, phi_3, .... (N x p)
                 double* input,                         //? Augmented vector (N + p)
                 double* output,                        //? A~ . v, where A~ = [A B; 0 K] and K = [0 I; 0 0]
                 size_t N,                              //? Number of grid points
                 bool GPU                               //? false (0) --> CPU; true (1) --> GPU
                 )
    {
        std::vector<double*> B(interp_vector.rbegin(), interp_vector.rend()-1);
        int p = B.size();

        double* rhs_u = (double*)malloc(N*sizeof(double));
        double* vB = (double*)malloc(N*sizeof(double));

        RHS(input, rhs_u);
        v_dot_B(rhs_u, B, vB, N);
        
        //? output[0:N] = rhs_u + vB
        axpby(1.0, rhs_u, 1.0, vB, output, N, GPU);

        //? output[N:N+p] = input[N+1:N+p], 0.0
        for(int ii = N+1; ii < N+p; ii++)
        {
            output[ii-1] = input[ii];
        }
        output[N+p-1] = 0.0;
    }


    //? polynomial[0:n] = phi_0(A) u(:, 1) + phi_1(A) u(:, 2) + ... + phi_p(A) u(:, p+1)
    template <typename rhs>
    void linear_phi(rhs& RHS,                                //? RHS function (function that evaluates the augmented matrix)
                    std::vector<double*> interp_vector,      //? Vector to evaluated/interpolated
                    double* polynomial,                      //? Output vector multiplied to linear combinations of phi functions
                    double* auxiliary_Leja,                  //? Internal auxiliary variables (Leja)
                    size_t N,                                //? Number of grid points
                    double T_final,                          //? Step size
                    int substeps,                            //? Initial guess for substeps
                    double integrator_coeff,                 //? Coefficients of the integrator
                    std::vector<double>& Leja_X,             //? Array of Leja points
                    double c,                                //? Shifting factor
                    double Gamma,                            //? Scaling factor
                    double tol,                              //? Tolerance (normalised desired accuracy)
                    int& iters,                              //? # of iterations needed to converge (iteration variable)
                    bool GPU,                                //? false (0) --> CPU; true (1) --> GPU
                    GPU_handle& cublas_handle                //? CuBLAS handle
                    )
    {
        //* -------------------------------------------------------------------------

        //* Evaluates a linear combinaton of the phi functions as the exponential of an augmented matrix.
        //*
        //*    Output parameters
        //*    ----------
        //*    polynomial[0:N]          : double*
        //*                                 Polynomial interpolation of 'u' multiplied 
        //*                                 by the matrix exponential at real/imag Leja points
        //*    substeps                 : int
        //*                                  Number of substeps used
        //*    iters                    : int
        //*                                  Number of iterations needed to converge
        //*
        //*    Reference: 
        //*
        //*         R.B. Sidje, Expokit: A Software Package for Computing Matrix Exponentials, ACM Trans. Math. Softw. 24 (1) (1998) 130 - 156.
        //*         doi:10.1145/285861.285868

        //* -------------------------------------------------------------------------


        int m = interp_vector.size();
        int p = m - 1;

        std::vector<double*> B(interp_vector.rbegin(), interp_vector.rend()-1);

        auto RHS_aug_v = [&](double* input, double* output) 
        {
            RHS_aug(RHS, interp_vector, input, output, N, GPU);
        };

        double* v = (double*)malloc((N+p)*sizeof(double));
        for(int ii = 0; ii < N+p-1; ii++)
        {
            v[ii] = 0.0;
        }
        v[N+p-1] = 1.0;
        std::cout << sizeof(v)/sizeof(v[0]) << std::endl;

        real_Leja_linear_exp(RHS_aug_v, v, polynomial, auxiliary_Leja, N, T_final, substeps, integrator_coeff, Leja_X, c, Gamma, tol, iters, GPU, cublas_handle);


    }


}
