#pragma once

#include "Leja.hpp"
#include "Phi_functions.hpp"
#include "Divided_Differences.hpp"

namespace LeXInt
{
    //? Phi function interpolated on real Leja points
    template <typename rhs>
    void real_Leja_phi(rhs& RHS,                                //? RHS function
                       double* u,                               //? Input state variable(s)
                       double* interp_vector,                   //? Input vector multiplied to phi function
                       double* polynomial,                      //? Output vector multiplied to phi function
                       double* auxiliary_Leja,                  //? Internal auxiliary variables (Leja)
                       size_t N,                                //? Number of grid points
                       std::vector<double> integrator_coeffs,   //? Coefficients of the integrator
                       double (* phi_function) (double),        //? Phi function
                       std::vector<double>& Leja_X,             //? Array of Leja points
                       double c,                                //? Shifting factor
                       double Gamma,                            //? Scaling factor
                       double tol,                              //? Tolerance (normalised desired accuracy)
                       double dt,                               //? Step size
                       int& iters,                              //? # of iterations needed to converge (iteration variable)
                       bool GPU,                                //? false (0) --> CPU; true (1) --> GPU
                       GPU_handle& cublas_handle                //? CuBLAS handle
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
        //*                             phi function, at real Leja points, at the respective
        //*                             integrator coefficients.
        //*
        //* -------------------------------------------------------------------------

        int max_Leja_pts = Leja_X.size();                                     //? Max. # of Leja points
        int num_interpolations = integrator_coeffs.size();                    //? Number of interpolations in vertical

        double* Jac_vec = &auxiliary_Leja[0];                                 //? auxiliary variable for Jacobian-vector product
        double* auxiliary_Jv = &auxiliary_Leja[N];                            //? auxiliary variables for Jacobian-vector computation
        double* y = &auxiliary_Leja[3*N];                                     //? To avoid overwriting "interp_vector"
        copy(interp_vector, y, N, GPU);

        //* Phi function applied to 'y' (scaled and shifted)
        std::vector<std::vector<double> > phi_function_array(num_interpolations);

        //* Polynomial coefficients
        std::vector<std::vector<double> > coeffs(num_interpolations);
        
        //* Reshape vectors to "num_interpolations x max_Leja_pts"
        for (int ij = 0; ij < num_interpolations; ij++)
        {
        	phi_function_array[ij].resize(max_Leja_pts);
        	coeffs[ij].resize(max_Leja_pts);
        }

        //* Loop for vertical implementation
        for (int ij = 0; ij < num_interpolations; ij++)
        {
            for (int ii = 0; ii < max_Leja_pts; ii++)
            {
                //? Phi function applied to 'y' (scaled and shifted)
                phi_function_array[ij][ii] = phi_function(integrator_coeffs[ij] * dt * (c + (Gamma * Leja_X[ii])));
            }

            //? Compute polynomial coefficients
            coeffs[ij] = Divided_Differences(Leja_X, phi_function_array[ij]);

            //? Form the polynomial (first term): polynomial = coeffs[0] * y
            axpby(coeffs[ij][0], y, &polynomial[ij*N], N, GPU);
        }

        //? Iterate until converges
        for (iters = 1; iters < max_Leja_pts - 1; iters++)
        {
            //* Compute numerical Jacobian: J(u) * y = (F(u + epsilon*y) - F(u - epsilon*y))/(2*epsilon)
            Jacobian_vector(RHS, u, y, Jac_vec, auxiliary_Jv, N, GPU, cublas_handle);

            //* y = y * ((z - c)/Gamma - Leja_X)
            axpby(1./Gamma, Jac_vec, (-c/Gamma - Leja_X[iters - 1]), y, y, N, GPU);

            //* Add the new term to the polynomial
            for (int ij = 0; ij < num_interpolations; ij++)
            {
                //? polynomial = polynomial + coeffs[iters] * y
                axpby(coeffs[ij][iters], y, 1.0, &polynomial[ij*N], &polynomial[ij*N], N, GPU);
            }

            //* Error estimate for 'y': poly_error = |coeffs[iters]| ||y|| at every iteration
            double poly_error = l2norm(y, N, GPU, cublas_handle);
            poly_error = abs(coeffs[num_interpolations - 1][iters]) * poly_error;

            //* Norm of the (largest) polynomial
            double poly_norm = l2norm(&polynomial[(num_interpolations - 1)*N], N, GPU, cublas_handle);

            //? If new term to be added < tol, break loop
            if (poly_error < ((tol*poly_norm) + tol))
            {
                std::cout << "Converged! Iterations: " << iters << std::endl;
                break;
            }

            //! Warning flags
            if (iters == max_Leja_pts - 2)
            {
                
                std::cout << "Warning!! Max. number of Leja points reached without convergence!! Reduce dt." << std::endl;
                break;
            }
        }
    }
}