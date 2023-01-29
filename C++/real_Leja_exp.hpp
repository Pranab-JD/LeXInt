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
state real_Leja_exp(rhs& RHS, state& u, int N, vector<double>& Leja_X, double c, double Gamma, double tol, double dt)
{
    //* -------------------------------------------------------------------------

    //* Computes the polynomial interpolation of matrix exponential applied to 'u' at real Leja points.
    //*
    //*    Parameters
    //*    ----------
    //*
    //*    Leja_X                  : vector <double>
    //*                                Set of Leja points
    //*
    //*    c                       : double
    //*                                Shifting factor
    //*
    //*    Gamma                   : double
    //*                                Scaling factor
    //*
    //*    tol                     : double
    //*                                Accuracy of the polynomial so formed
    //*
    //*    dt                      : double
    //*                                Step size
    //*
    //*    Returns
    //*    ----------
    //*    polynomial              : 
    //*                                Polynomial interpolation of 'u' multiplied 
    //*                                by the matrix exponential at real Leja points

    //* -------------------------------------------------------------------------
    
    int max_Leja_pts = Leja_X.size();               //? Max. # of Leja points
    state y(u);                                         //? To avoid changing 'u'
    state Jacobian_function(N);                         //? Jacobian-vector product
    state polynomial(N);                                //? Initialise the polynomial
    double poly_error;                              //? Error incurred at every iteration

    //* Matrix exponential (scaled and shifted)
    vector<double> matrix_exponential(max_Leja_pts);

    for (int ii = 0; ii < max_Leja_pts; ii++)
    {
        matrix_exponential[ii] = exp(dt * (c + (Gamma * Leja_X[ii])));
    }

    //* Compute polynomial coefficients
    vector<double> coeffs = Divided_Differences(Leja_X, matrix_exponential);

    //* Form the polynomial: p_0 term
    for (int ii = 0; ii < N; ii++)
    {
        polynomial[ii] = coeffs[0] * y[ii];
    }

    //? Iterate until converges
    for (int nn = 1; nn < max_Leja_pts - 1; nn++)
    {
        //* Compute numerical Jacobian (for linear eqs., this is the RHS evaluation at y)
        Jacobian_function = RHS(y);

        //* y = y * ((z - c)/Gamma - Leja_X)
        for (int ii = 0; ii < N; ii++)
        {
            y[ii] = Jacobian_function[ii]/Gamma + (y[ii] * (-c/Gamma - Leja_X[nn - 1]));
        }

        //* Error estimate
        poly_error = abs(coeffs[nn]) * l2norm(y, N);

        //* Add the new term to the polynomial
        for (int ii = 0; ii < N; ii++)
        {
            polynomial[ii] = polynomial[ii] + (coeffs[nn] * y[ii]);
        }

        //? If new term to be added < tol, break loop; safety factor = 0.25
        if (poly_error < 0.25*tol*l2norm(polynomial, N))
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