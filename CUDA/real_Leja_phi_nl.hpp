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
void real_Leja_phi_nl(rhs& RHS,                           //? RHS function
                      double* interp_vector,              //? Input vector multiplied to phi function
                      double* polynomial,                 //? Output vector multiplied to phi function
                      double* auxillary_Leja,             //? Internal auxillary variables (Leja)
                      size_t N,                           //? Number of grid points
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
    //*                             phi function, at real Leja points
    //*
    //* -------------------------------------------------------------------------

    int max_Leja_pts = Leja_X.size();                               //? Max. # of Leja points
    double* Jacobian_function = &auxillary_Leja[0];                 //? Auxillary variable for Jacobian-vector product
    
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
    for (int nn = 1; nn < max_Leja_pts - 1; nn++)
    {
        //* Compute numerical Jacobian (for linear eqs., this is the RHS evaluation at y)
        RHS(interp_vector, Jacobian_function);

        //* y = y * ((z - c)/Gamma - Leja_X)
        axpby(1./Gamma, Jacobian_function, (-c/Gamma - Leja_X[nn - 1]), interp_vector, interp_vector, N, GPU);

        //* Add the new term to the polynomial (polynomial = polynomial + (coeffs[nn] * y))
        axpby(coeffs[nn], interp_vector, 1.0, polynomial, polynomial, N, GPU);

        //* Error estimate: poly_error = |coeffs[nn]| ||interp_vector|| at every iteration
        double poly_error = l2norm(interp_vector, N, GPU, cublas_handle);
        poly_error = abs(coeffs[nn]) * poly_error;

        //? If new term to be added < tol, break loop
        if (poly_error < ((tol*poly_error) + tol))
        {
            cout << "Converged! Iterations: " << nn << endl;
            break;
        }

        //! Warning flags
        if (nn == max_Leja_pts - 2)
        {
            cout << "Warning!! Max. number of Leja points reached without convergence!!" << endl; 
            cout << "Max. Leja points currently set to " << max_Leja_pts << endl;
            cout << "Try increasing the number of Leja points. Max available: 10000." << endl;
            break;
        }
    }
}