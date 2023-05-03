#pragma once

#include "Phi_functions.hpp"
#include "Divided_Differences.hpp"
#include "Timer.hpp"

//? CUDA
#include "error_check.hpp"
#include "Leja_GPU.hpp"

using namespace std;

//? Phi function interpolated on real Leja points
template <typename rhs>
void real_Leja_phi(rhs& RHS,                           //? RHS function
                   double* u,                          //? Input state variable(s)
                   double* interp_vector,              //? Input vector multiplied to phi function
                   double* polynomial,                 //? Output vector multiplied to phi function
                   double* auxillary_Leja,             //? Internal auxillary variables (Leja)
                   size_t N,                           //? Number of grid points
                   vector<double> integrator_coeffs,   //? Coefficients of the integrator
                   double (* phi_function) (double),   //? Phi function
                   vector<double>& Leja_X,             //? Array of Leja points
                   double c,                           //? Shifting factor
                   double Gamma,                       //? Scaling factor
                   double tol,                         //? Tolerance (normalised desired accuracy)
                   double dt,                          //? Step size
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
    //*                             phi function, at real Leja points, at the respective
    //*                             integrator coefficients.
    //*
    //* -------------------------------------------------------------------------

    int max_Leja_pts = Leja_X.size();                                     //? Max. # of Leja points
    int num_interpolations = integrator_coeffs.size();                    //? Number of interpolations in vertical

    double* Jacobian_function = &auxillary_Leja[0];                       //? Auxillary variable for Jacobian-vector product
    double* auxillary_Jv = &auxillary_Leja[N];                            //? Auxillary variables for Jacobian-vector computation

    //* Phi function applied to 'interp_vector' (scaled and shifted)
    vector<vector<double>> phi_function_array(num_interpolations);

    //* Polynomial coefficients
    vector<vector<double>> coeffs(num_interpolations);
    
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
            //? Phi function applied to 'interp_vector' (scaled and shifted)
            phi_function_array[ij][ii] = phi_function(integrator_coeffs[ij] * dt * (c + (Gamma * Leja_X[ii])));
        }

        //? Compute polynomial coefficients
        coeffs[ij] = Divided_Differences(Leja_X, phi_function_array[ij]);

        //? Form the polynomial (first term): polynomial = coeffs[0] * y
        axpby(coeffs[ij][0], interp_vector, &polynomial[ij*N], N, GPU);
    }

    timer t1;

    //? Iterate until converges
    for (int nn = 1; nn < max_Leja_pts - 1; nn++)
    {
        // t1.start();
        //* Compute numerical Jacobian: J(u) * y = (F(u + epsilon*y) - F(u - epsilon*y))/(2*epsilon)
        Jacobian_vector(RHS, u, interp_vector, Jacobian_function, auxillary_Jv, N, GPU, cublas_handle);

        // cudaDeviceSynchronize(); t1.start();
        //* y = y * ((z - c)/Gamma - Leja_X)
        axpby(1./Gamma, Jacobian_function, (-c/Gamma - Leja_X[nn - 1]), interp_vector, interp_vector, N, GPU);
        // cudaDeviceSynchronize(); t1.stop();

        //* Add the new term to the polynomial
        for (int ij = 0; ij < num_interpolations; ij++)
        {
            //? polynomial = polynomial + coeffs[nn] * y
            axpby(coeffs[ij][nn], interp_vector, 1.0, &polynomial[ij*N], &polynomial[ij*N], N, GPU);
        }

        //* Error estimate for 'y': poly_error = |coeffs[nn]| ||y|| at every iteration
        double poly_error = l2norm(interp_vector, N, GPU, cublas_handle);
        poly_error = abs(coeffs[num_interpolations - 1][nn]) * poly_error;

        //* Norm of the (largest) polynomial
        double poly_norm = l2norm(&polynomial[(num_interpolations - 1)*N], N, GPU, cublas_handle);

        //? If new term to be added < tol, break loop
        if (poly_error < ((tol*poly_norm) + tol))
        {
            cout << "Converged! Iterations: " << nn << endl;
            // t1.stop();
            // cout << "Leja " << t1.average() << endl;
            break;
        }

        //! Warning flags
        if (nn == max_Leja_pts - 2)
        {
            cout << "Warning!! Max. number of Leja points reached without convergence!! Reduce dt." << endl;
            break;
        }
    }
}