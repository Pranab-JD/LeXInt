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

import sys
import numpy as np

sys.path.insert(1, "./LeXInt/Python/")
from Jacobian import Jacobian

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

    ###? Divide matrix 'A' into Hermitian and skew-Hermitian
    A_Herm = (A + A.T.conj())/2
    A_SkewHerm = (A - A.T.conj())/2

    eig_real = - np.max(np.sum(abs(A_Herm), 1))       # Has to be NEGATIVE
    eig_imag = np.max(np.sum(abs(A_SkewHerm), 1))

    return eig_real, eig_imag

def Power_iteration(u, RHS_function):
    """
    Parameters
    ----------
    u                       : Input state variable(s)
    RHS_function	        : RHS function

    Returns
    -------
    largest_eigen_value     : Largest eigenvalue (within 2% accuracy)
    3*ii                    : Number of RHS calls

    """

    tol = 0.02                                  #? 2% tolerance
    niters = 1000                               #? Max. number of iterations                    
    eigenvalue_ii_1 = 0                         #? Eigenvalue at ii-1
    vector = np.ones(np.size(u))                #? Initial estimate of eigenvector

    for ii in range(niters):

        ###? Compute new eigenvector
        eigenvector = Jacobian(RHS_function, u, vector)

        ###? Norm of eigenvector = eigenvalue
        eigenvalue = np.linalg.norm(eigenvector)
        
        ###? Normalize eigenvector to eigenvalue; new estimate of eigenvector
        vector = eigenvector/eigenvalue

        ###? Check convergence for eigenvalues (eigenvalues converge faster than eigenvectors)
        if (abs(eigenvalue - eigenvalue_ii_1) <= tol * eigenvalue):
            largest_eigen_value = eigenvalue
            break
        
        ###? This value becomes the previous one
        eigenvalue_ii_1 = eigenvalue

    return largest_eigen_value, 3*ii