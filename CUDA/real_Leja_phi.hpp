#pragma once

#include <iostream>
#include <iomanip>
#include <vector>
#include <cmath>
#include <functional>

#include "Phi_functions.hpp"
#include "Divided_Differences.hpp"

//? CUDA 
#include "cublas_v2.h"
#include "error_check.hpp"
#include "Leja_GPU.hpp"

using namespace std;

//? Phi function interpolated on real Leja points
template <typename state, typename rhs>
vector<state> Leja_GPU<state, rhs> :: real_Leja_phi(rhs& RHS,                           //? RHS function
                                                    state& u,                           //? State variable(s)
                                                    state& interp_vector,               //? Vector multiplied to phi function
                                                    vector<double> integrator_coeffs,   //? Coefficients of the integrator
                                                    double (* phi_function) (double),   //? Phi function (typically phi_1)
                                                    vector<double>& Leja_X,             //? Array of Leja points
                                                    double c,                           //? Shifting factor
                                                    double Gamma,                       //? Scaling factor
                                                    double tol,                         //? Tolerance (normalised desired accuracy)
                                                    double dt                           //? Step size
                                                    )
{
    //* -------------------------------------------------------------------------
    //*
    //* Computes the polynomial interpolation of phi function applied to 'interp_vector' at real Leja points.
    //*
    //*    Returns
    //*    ----------
    //*    polynomial              : vector<state>
    //*                                 Polynomial interpolation of 'interp_vector', applied to
    //*                                 phi function, at real Leja points
    //*
    //* -------------------------------------------------------------------------
    

    double y_error;
    double poly_norm;                                               //? Norm of the polynomial
    int max_Leja_pts = Leja_X.size();                               //? Max. # of Leja points
    int num_interpolations = integrator_coeffs.size();              //? Number of interpolations in vertical
    y = interp_vector;                                              //? To avoid changing 'interp_vector'

    
    vector<vector<double>> phi_function_array(num_interpolations);  //? Phi function applied to 'interp_vector'
    vector<vector<double>> coeffs(num_interpolations);              //? Polynomial coefficients
    
    //! Defined as state in Leja_GPU.hpp
    vector<state> polynomial(num_interpolations);                   //? Polynomial array
    
    for (int ij = 0; ij < num_interpolations; ij++)
    {
    	phi_function_array[ij].resize(max_Leja_pts);
    	coeffs[ij].resize(max_Leja_pts);
    	polynomial[ij].resize(N);
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
        polynomial[ij] = axpby(coeffs[ij][0], y, N);
    }

    //? Iterate until converges
    for (int nn = 1; nn < max_Leja_pts - 1; nn++)
    {
        //* Compute numerical Jacobian
        Jacobian_function = Jacobian_vector(RHS, u, y, N);

        //* y = y * ((z - c)/Gamma - Leja_X)
        y = axpby(1.0/Gamma, Jacobian_function, (-c/Gamma - Leja_X[nn - 1]), y, N);

        //* Error estimate; poly_error = |coeffs[nn]| ||y||
        poly_error = abs(coeffs[num_interpolations - 1][nn]) * l2norm(y, N);

        //* Add the new term to the polynomial
        for (int ij = 0; ij < num_interpolations; ij++)
        {
            //? polynomial = polynomial + coeffs[nn] * y
            polynomial[ij] = axpby(1.0, polynomial[ij], coeffs[ij][nn], y, N);
        }

        //? If new term to be added < tol, break loop
        if (poly_error < tol*l2norm(polynomial[num_interpolations - 1], N) + tol)
        {
            // cout << "Converged: " << nn << endl;
            break;
        }

        //! Warning flags
        if (nn == max_Leja_pts - 2)
        {
            cout << "Warning!! Max. # of Leja points reached without convergence!! Reduce dt. " << endl;
            break;
        }
    }
    
    return polynomial;
}
