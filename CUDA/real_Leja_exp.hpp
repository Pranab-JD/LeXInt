#pragma once

#include "Divided_Differences.hpp"
#include "Timer.hpp"

//? CUDA
#include "error_check.hpp"
#include "Leja_GPU.hpp"

namespace LeXInt
{
    //? Matrix exponential interpolated on real Leja points
    template <typename rhs>
    void real_Leja_exp(rhs& RHS,                       //? RHS function
                       double* u,                      //? Input state variable(s)
                       double* polynomial,             //? Output matrix exponential multiplied by 'u'
                       double* auxiliary_Leja,         //? Internal auxiliary variables (Leja)
                       size_t N,                       //? Number of grid points
                       vector<double>& Leja_X,         //? Array of Leja points
                       double c,                       //? Shifting factor
                       double Gamma,                   //? Scaling factor
                       double tol,                     //? Tolerance (normalised desired accuracy)
                       double dt,                      //? Step size
                       int& iters,                     //? # of iterations needed to converge (iteration variable)
                       bool GPU,                       //? false (0) --> CPU; true (1) --> GPU
                       GPU_handle& cublas_handle       //? CuBLAS handle
                       )
    {
        //* -------------------------------------------------------------------------

        //* Computes the polynomial interpolation of matrix exponential applied to 'u' at real Leja points.
        //*
        //*    Returns
        //*    ----------
        //*    polynomial        : double*
        //*                             Polynomial interpolation of 'u' multiplied 
        //*                             by the matrix exponential at real Leja points

        //* -------------------------------------------------------------------------
        
        int max_Leja_pts = Leja_X.size();                               //? Max. # of Leja points
        double* Jac_vec = &auxiliary_Leja[0];                           //? auxiliary variable for Jacobian-vector product

        //* Matrix exponential (scaled and shifted)
        vector<double> matrix_exponential(max_Leja_pts);

        for (int ii = 0; ii < max_Leja_pts; ii++)
        {
            matrix_exponential[ii] = exp(dt * (c + (Gamma * Leja_X[ii])));
        }
        
        //* Compute polynomial coefficients
        vector<double> coeffs = Divided_Differences(Leja_X, matrix_exponential);

        //* Form the polynomial (first term): polynomial = coeffs[0] * u
        axpby(coeffs[0], u, polynomial, N, GPU);

        //? Iterate until converges
        for (iters = 1; iters < max_Leja_pts - 1; iters++)
        {
            //* Compute numerical Jacobian (for linear eqs., this is the RHS evaluation at u)
            RHS(u, Jac_vec);

            //* u = u * ((z - c)/Gamma - Leja_X)
            axpby(1./Gamma, Jac_vec, (-c/Gamma - Leja_X[iters - 1]), u, u, N, GPU);

            //* Add the new term to the polynomial (polynomial = polynomial + (coeffs[iters] * u))
            axpby(coeffs[iters], u, 1.0, polynomial, polynomial, N, GPU);

            //* Error estimate: poly_error = |coeffs[iters]| ||u|| at every iteration
            double poly_error = l2norm(u, N, GPU, cublas_handle);
            poly_error = abs(coeffs[iters]) * poly_error;

            //* Norm of the polynomial
            double poly_norm = l2norm(polynomial, N, GPU, cublas_handle);

            //? If new term to be added < tol, break loop
            if (poly_error < ((tol*poly_norm) + tol))
            {
                // ::std::cout << "Converged! Iterations: " << iters << ::std::endl;
                break;
            }

            //! Warning flags
            if (iters == max_Leja_pts - 2)
            {
                ::std::cout << "Warning!! Max. number of Leja points reached without convergence!!" << ::std::endl; 
                ::std::cout << "Max. Leja points currently set to " << max_Leja_pts << ::std::endl;
                ::std::cout << "Try increasing the number of Leja points. Max available: 10000." << ::std::endl;
                break;
            }
        }
    }
}