#pragma once
#include <vector>

vector<double> Divided_Differences(vector<double> X, vector<double> diffs)
{
    //* -------------------------------------------------------------------------
    //* Compute the coefficients for polynomial interpolation.
    //*
    //* Parameters
    //* -----------
    //* X                     : vector <double>
    //*                           Set of Leja points
    //* 
    //* diffs                 : vector <double>
    //*                           Vector of which coeffs are to be computed
    //*
    //* Returns
    //* ----------
    //* coeffs                : vector <double>
    //*                           Coefficients
    //* -------------------------------------------------------------------------

    //* # of interpolation (Leja) points
    int N = X.size();

    //* Return coefficients of the vector 'diffs'
    vector<double> coeffs(diffs);

    //* Compute the divided differences
    for (int ii = 1; ii < N; ii++)
    {
        for (int jj = 0; jj < ii; jj++)
        {
            coeffs[ii] = (coeffs[ii] - coeffs[jj])/(X[ii] - X[jj]);
        }
    }
    
    return coeffs;
}