#pragma once

#include <vector>

namespace LeXInt
{
    std::vector<double> Divided_Differences(const std::vector<double>& X, std::vector<double> coeffs)
    {
        //* -------------------------------------------------------------------------
        //* Compute the coefficients for polynomial interpolation.
        //*
        //* Parameters
        //* -----------
        //* X                     : vector <double>
        //*                           Set of Leja points
        //* 
        //* coeffs                 : vector <double>
        //*                           Vector of which coeffs are to be computed
        //*
        //* Returns
        //* ----------
        //* coeffs                : vector <double>
        //*                           Coefficients
        //* -------------------------------------------------------------------------

        //* Number of interpolation (Leja) points
        int N = X.size();

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
}