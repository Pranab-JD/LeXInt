#pragma once

#include <iostream>
#include <iomanip>
#include <vector>
#include <cmath>
#include <functional>

#include "functions.hpp"
#include "Phi_functions.hpp"
#include "Divided_Differences.hpp"

using namespace std;

//? Phi function interpolated on real Leja points
template <typename state, typename rhs>
state real_Leja_phi_nl(rhs& RHS,                           //? RHS function
                       state& u,                           //? State variable(s)
                       state& interp_vector,               //? Vector multiplied to phi function
                       int N,                              //? Number of grid points
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
    //*    polynomial              : state
    //*                                 Polynomial interpolation of 'interp_vector', applied to
    //*                                 phi function, at real Leja points
    //*
    //* -------------------------------------------------------------------------

    int max_Leja_pts = Leja_X.size();                   //? Max. # of Leja points
    double poly_error;                                  //? Error incurred at every iteration

    state y(u);                                         //? To avoid changing 'u'
    state Jacobian_function(N);                         //? Jacobian-vector product
    state polynomial(N);                                //? Initialise the polynomial
    
    //* Matrix exponential (scaled and shifted)
    vector<double> phi_function_array(max_Leja_pts);

    for (int ii = 0; ii < max_Leja_pts; ii++)
    {
        phi_function_array[ii] = phi_function(dt * (c + (Gamma * Leja_X[ii])));
    }

    //* Compute polynomial coefficients
    vector<double> coeffs = Divided_Differences(Leja_X, phi_function_array);

    //* Form the polynomial: p_0 term
    polynomial = axpby(coeffs[0], y, N);

    //? Iterate until converges
    for (int nn = 1; nn < max_Leja_pts - 1; nn++)
    {
        //* Compute numerical Jacobian
        Jacobian_function = RHS(y);

        //* y = y * ((z - c)/Gamma - Leja_X)
        y = axpby(1.0/Gamma, Jacobian_function, (-c/Gamma - Leja_X[nn - 1]), y, N);

        //* Error estimate; poly_error = |coeffs[nn]| ||y||
        poly_error = abs(coeffs[nn]) * l2norm(y, N);

        //* Add the new term to the polynomial
        polynomial = axpby(1.0, polynomial, coeffs[nn], y, N);

        //? If new term to be added < tol, break loop
        if (poly_error < tol*l2norm(polynomial, N) + tol)
        {
            break;
        }

        //! Warning flags
        if (nn == max_Leja_pts - 2)
        {
            cout << "Warning!! Max. # of Leja points reached without convergence!! Max. Leja points currently set to " << max_Leja_pts << endl;
            cout << "Try increasing the number of Leja points. Max available: 10000." << endl;
            break;
        }
    }
    
    return polynomial;
}
