"""
Created on Thu Aug 8 20:22 2022

@author: Pranab JD

Description: -
        Functions to determine the largest eigenvalue of a 
        matrix/related matrix.
        
        Gershgorin's disks can be used only if the matrix is 
        explicitly available. For matrix-free implementation, 
        choose power iterations.
"""

import numpy as np

def Gershgorin(A):
    """
    Parameters
    ----------
    A        : N x N matrix

    Returns
    -------
    eig_real : Largest real eigenvalue (negative magnitude)
    eig_imag : Largest imaginary eigenvalue

    """

    ### Divide matrix 'A' into Hermitian and skew-Hermitian
    A_Herm = (A + A.T.conj())/2
    A_SkewHerm = (A - A.T.conj())/2

    eig_real = - np.max(np.sum(abs(A_Herm), 1))       # Has to be NEGATIVE
    eig_imag = np.max(np.sum(abs(A_SkewHerm), 1))

    return eig_real, eig_imag

def Power_iteration(u, RHS_function):
    """
    Parameters
    ----------
    u                       : 1D vector u (input)
    RHS_function	        : RHS function

    Returns
    -------
    largest_eigen_value     : Largest eigenvalue (within 10% accuracy)
    2*ii                    : # of RHS calls

    """

    tol = 0.01                                  # 1% tolerance
    niters = 1000                               # Max. # of iterations
    epsilon = 1e-7
    eigen_value = np.zeros(niters)              # Array of max. eigen value at each iteration
    vector = np.zeros(len(u)); vector[0] = 1    # Initial estimate of eigen vector

    for ii in range(niters):

        ### Compute new eigenvector
        eigen_vector = (RHS_function(u + (epsilon * vector)) - RHS_function(u - (epsilon * vector)))/(2*epsilon)

        ### Max of eigenvector = eigenvalue
        eigen_value[ii] = np.max(abs(eigen_vector))
        
        ### Normalize eigenvector to eigenvalue
        eigen_vector = eigen_vector/eigen_value[ii]
        
        ### New estimate of eigenvector
        vector = eigen_vector.copy()

        ### Eigenvalues converge faster than eigenvectors, check convergence of eigenvalues
        if (abs(eigen_value[ii] - eigen_value[ii - 1]) <= tol * eigen_value[ii]):
            largest_eigen_value = eigen_value[ii]
            break

    return largest_eigen_value, 2*ii