#pragma once

#include <iostream>
#include <vector>

#include "functions.hpp"

using namespace std;

//? -----------------------------------------------------------------
//?
//? Description:
//?     Use Gershgorin's disk theorem if matrix is explcitly avilable.
//?     Else, use power iterations.
//?
//?     NOTE: Largest real eigenvalue has to be NEGATIVE!
//?
//? -----------------------------------------------------------------

//! ======================================================================================== !//

//! Gershgorin's Disk Theorem
template <typename T>
double Gershgorin(T A,      //? N x N matrix A 
                  int N     //? No. of rows or columns
                  )
{
    vector<double> eigen_list(N);
        
    for (int ii = 0; ii < N; ii++)
    {
        double eigenvalue = 0;

        for (int jj = 0; jj < N; jj++)
        {
            eigenvalue = eigenvalue + abs(A[ii][jj]);
        }

        eigen_list[ii] = eigenvalue;
    }

    //! Returns the largest eigenvalue in magnitude
    return *max_element(begin(eigen_list), end(eigen_list));
}

//! ======================================================================================== !//

//! Power Iterations
template <typename rhs, typename T>
double Power_iterations(rhs& RHS,       //? RHS function
                        T u,            //? State variable(s)
                        int N           //? Number of grid points
                        )
{
    double tol = 0.1;                                   //? 10% tolerance
    int niters = 1000;                                  //? Max. number of iterations
    double eigen_max, eigen_min;
    double largest_eigenvalue;                          //? Largest eigenvalue
    
    vector<double> eigenvalue(niters);                  //? Array of largest eigenvalue at each iteration
    vector<double> init_vector(N); init_vector[5] = 1;  //? Initial estimate of eigenvector
    vector<double> eigenvector(N);                      //? Iniliatise eigenvector

    for (int ii = 0; ii < niters; ii++)
    {
        //? Compute new eigenvector
        eigenvector = Jacobian_vector(RHS, u, init_vector, N);

        //? Max of eigenvector = eigenvalue
        eigen_max = *max_element(begin(eigenvector), end(eigenvector));
        eigen_min = *min_element(begin(eigenvector), end(eigenvector));
        eigenvalue[ii] = max(eigen_min, eigen_max);

        //? Normalize eigenvector to eigenvalue
        eigenvector = axpby(1.0/eigenvalue[ii], eigenvector, N);

        //? New estimate of eigenvector
        init_vector = axpby(1.0, eigenvector, N);

        //? Check convergence for eigenvalues (eigenvalues converge faster than eigenvectors)
        if (abs(eigenvalue[ii] - eigenvalue[ii - 1]) <= tol * eigenvalue[ii])
        {
            cout << ii << endl;
            largest_eigenvalue = eigenvalue[ii];
            break;
        }
    }

    //! Returns the largest eigenvalue in magnitude (needs to multiplied to a safety factor)
    return largest_eigenvalue;

}

//! ======================================================================================== !//
