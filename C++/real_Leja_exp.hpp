#pragma once

#include <iostream>
#include <iomanip>
#include <vector>
#include <cmath>
#include <functional>

#include "functions.hpp"
#include "Divided_Differences.hpp"

using namespace std;

//? Matrix exponential interpolated on real Leja points
template <typename state, typename rhs>
state real_Leja_exp(rhs& RHS,                       //? RHS function
                    state& u,                       //? State variable(s)
                    int N,                          //? Number of grid points
                    vector<double>& Leja_X,         //? Array of Leja points
                    double c,                       //? Shifting factor
                    double Gamma,                   //? Scaling factor
                    double tol,                     //? Tolerance (normalised desired accuracy)
                    double dt                       //? Step size
                    )
{
    //* -------------------------------------------------------------------------
    //*
    //* Computes the polynomial interpolation of matrix exponential applied to 'u' at real Leja points.
    //*
    //*    Returns
    //*    ----------
    //*    polynomial              : state
    //*                                 Polynomial interpolation of 'u', applied to
    //*                                 matrix exponential, at real Leja points
    //*
    //* -------------------------------------------------------------------------
    
    int max_Leja_pts = Leja_X.size();                   //? Max. # of Leja points
    double poly_error;                                  //? Error incurred at every iteration

    state y(u);                                         //? To avoid changing 'u'
    state Jacobian_function(N);                         //? Jacobian-vector product
    state polynomial(N);                                //? Initialise the polynomial
    
    //* Matrix exponential (scaled and shifted)
    vector<double> matrix_exponential(max_Leja_pts);

    for (int ii = 0; ii < max_Leja_pts; ii++)
    {
        matrix_exponential[ii] = exp(dt * (c + (Gamma * Leja_X[ii])));
    }

    //* Compute polynomial coefficients
    vector<double> coeffs = Divided_Differences(Leja_X, matrix_exponential);

    //* Form the polynomial: p_0 term
    polynomial = axpby(coeffs[0], y, N);

    //? Iterate until converges
    for (int nn = 1; nn < max_Leja_pts - 1; nn++)
    {
        //* Compute numerical Jacobian (for linear eqs., this is the RHS evaluation at y)
        Jacobian_function = RHS(y);

        //* y = y * ((z - c)/Gamma - Leja_X)
        y = axpby(1.0/Gamma, Jacobian_function, (-c/Gamma - Leja_X[nn - 1]), y, N);

        //* Error estimate
        poly_error = abs(coeffs[nn]) * l2norm(y, N);

        //* Add the new term to the polynomial
        polynomial = axpby(1.0, polynomial, coeffs[nn], y, N);

        //? If new term to be added < tol, break loop; safety factor = 0.1
        if (poly_error < 0.1*tol*l2norm(polynomial, N) + tol)
        {
            cout << "Converged! Iterations: " << nn << endl;
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