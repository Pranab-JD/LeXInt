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

//? Phi functions interpolated on real Leja points
template <typename state, typename rhs>
state real_Leja_phi(rhs& RHS, state& u, state& interp_vector, int N,  double (* phi_function) (double), vector<double>& Leja_X, double c, double Gamma, double tol, double dt)
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
    
    int max_Leja_pts = Leja_X.size();                   //? Max. # of Leja points
    double poly_error;                                  //? Error incurred at every iteration
    state y(interp_vector);                             //? To avoid changing 'u'
    state Jacobian_function(N);                         //? Jacobian-vector product
    state polynomial(N);                                //? Initialise the polynomial

    //* Matrix exponential (scaled and shifted)
    vector<double> phi_function_array(max_Leja_pts);

    for (int ii = 0; ii < max_Leja_pts; ii++)
    {
        //? Call phi function
        phi_function_array[ii] = phi_function(dt * (c + (Gamma * Leja_X[ii])));
    }

    //* Compute polynomial coefficients
    vector<double> coeffs = Divided_Differences(Leja_X, phi_function_array);

    //* Form the polynomial: p_0 term
    for (int ii = 0; ii < N; ii++)
    {
        polynomial[ii] = coeffs[0] * y[ii];
    }

    //? Iterate until converges
    for (int nn = 1; nn < max_Leja_pts - 1; nn++)
    {
        //* Compute numerical Jacobian
        Jacobian_function = Jacobian_vector(RHS, u, y, N);

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

        //? If new term to be added < tol, break loop
        if (poly_error < tol*l2norm(polynomial, N) + tol)
        {
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
