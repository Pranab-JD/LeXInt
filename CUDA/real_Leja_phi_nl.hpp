#pragma once

#include "Phi_functions.hpp"
#include "Divided_Differences.hpp"
#include "Timer.hpp"

//? CUDA
#include "error_check.hpp"
#include "Leja_GPU.hpp"

namespace LeXInt
{

    //? Phi function interpolated on real Leja points
    template <typename rhs>
    void real_Leja_phi_nl(rhs& RHS,                           //? RHS function
                          double* interp_vector,              //? Input vector multiplied to phi function
                          double* polynomial,                 //? Output vector multiplied to phi function
                          double* auxiliary_Leja,             //? Internal auxiliary variables (Leja)
                          size_t N,                           //? Number of grid points
                          double (* phi_function) (double),   //? Phi function
                          vector<double>& Leja_X,             //? Array of Leja points
                          double c,                           //? Shifting factor
                          double Gamma,                       //? Scaling factor
                          double tol,                         //? Tolerance (normalised desired accuracy)
                          double dt,                          //? Step size
                          int& iters,                         //? # of iterations needed to converge (iteration variable)
                          bool GPU,                           //? false (0) --> CPU; true (1) --> GPU
                          GPU_handle& cublas_handle           //? CuBLAS handle
                          )
    {
        //* -------------------------------------------------------------------------
        //*
        //* Computes the polynomial interpolation of phi function applied to 'interp_vector' at real Leja points.
        //*
        //*    Returns
        //*    ----------
        //*    polynomial          : double*
        //*                             Polynomial interpolation of 'interp_vector', applied to
        //*                             phi function, at real Leja points
        //*
        //* -------------------------------------------------------------------------

        int max_Leja_pts = Leja_X.size();                               //? Max. # of Leja points
        double* Jacobian_function = &auxiliary_Leja[0];                 //? auxiliary variable for Jacobian-vector product
        
        //* Phi function applied to 'interp_vector' (scaled and shifted)
        vector<double> phi_function_array(max_Leja_pts);
        
        for (int ii = 0; ii < max_Leja_pts; ii++)
        {
            phi_function_array[ii] = phi_function(dt * (c + (Gamma * Leja_X[ii])));
        }

        //* Compute polynomial coefficients
        vector<double> coeffs = Divided_Differences(Leja_X, phi_function_array);

        //* Form the polynomial (first term): polynomial = coeffs[0] * interp_vector
        axpby(coeffs[0], interp_vector, polynomial, N, GPU);

        //? Iterate until converges
        for (iters = 1; iters < max_Leja_pts - 1; iters++)
        {
            //* Compute numerical Jacobian (for linear eqs., this is the RHS evaluation at y)
            RHS(interp_vector, Jacobian_function);

            //* y = y * ((z - c)/Gamma - Leja_X)
            axpby(1./Gamma, Jacobian_function, (-c/Gamma - Leja_X[iters - 1]), interp_vector, interp_vector, N, GPU);

            //* Add the new term to the polynomial (polynomial = polynomial + (coeffs[iters] * y))
            axpby(coeffs[iters], interp_vector, 1.0, polynomial, polynomial, N, GPU);

            //* Error estimate: poly_error = |coeffs[iters]| ||interp_vector|| at every iteration
            double poly_error = l2norm(interp_vector, N, GPU, cublas_handle);
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