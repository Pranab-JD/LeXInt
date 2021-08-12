"""
Created on Thu Aug 12 19:15:17 2021

@author: Pranab JD

Description: -
        Contains embedded explicit integrators

"""

################################################################################################

def RKF45(u, dt, *A):
    """
    Parameters
    ----------
    u       : 1D vector u (input)
    dt      : Step size
    *A		: N x N matrix A, power to which u is raised

    Returns
    -------
    u_rkf4  : 1D vector u (output) after time dt (4th order)
    5, 10   : # of matrix-vector products for u_rkf4
    u_rkf5  : 1D vector u (output) after time dt (5th order)
    6, 12   : # of matrix-vector products for u_rkf5

    Note    : |u_rkf5 - u_rkf4| gives 4th order error estimate 

    """

    if len(A) == 2:

        # A[0] - matrix, A[1] - index of 'u'
        A = A[0]; m = A[1]                  

        k1 = dt * (A.dot(u**m))
        k2 = dt * (A.dot((u + k1/4)**m))
        k3 = dt * (A.dot((u + 3./32.*k1 + 9./32.*k2)**m))
        k4 = dt * (A.dot((u + 1932./2197.*k1 - 7200./2197.*k2 + 7296./2197.*k3)**m))
        k5 = dt * (A.dot((u + 439./216.*k1 - 8*k2 + 3680./513.*k3 - 845./4104.*k4)**m))
        k6 = dt * (A.dot((u - 8./27.*k1 + 2*k2 - 3544./2565.*k3 + 1859./4140.*k4 - 11./40.*k5)**m))

        ### Solution
        u_rkf4 = u + (25./216.*k1 + 1408./2565.*k3 + 2197./4101.*k4 - 1./5.*k5)
        u_rkf5 = u + (16./135.*k1 + 6656./12825.*k3 + 28561./56430.*k4 - 9./50.*k5 + 2./55.*k6)

        return u_rkf4, 5, u_rkf5, 6

    elif len(A) == 4:

        # A[0] - matrix 1, A[1] - index of 'u1', A[0] - matrix 2, A[1] - index of 'u2'
        A1 = A[0]; m1 = A[1]; A2 = A[2]; m2 = A[3]

        k1 = dt * (A1.dot(u**m1) + A2.dot(u**m2))
        k2 = dt * (A1.dot((u + k1/4)**m1) + A2.dot((u + k1/4)**m2))
        k3 = dt * (A1.dot((u + 3./32.*k1 + 9./32.*k2)**m1) + A2.dot((u + 3./32.*k1 + 9./32.*k2)**m2))
        k4 = dt * (A1.dot((u + 1932./2197.*k1 - 7200./2197.*k2 + 7296./2197.*k3)**m1) \
                + A2.dot((u + 1932./2197.*k1 - 7200./2197.*k2 + 7296./2197.*k3)**m2))
        k5 = dt * (A1.dot((u + 439./216.*k1 - 8*k2 + 3680./513.*k3 - 845./4104.*k4)**m1) \
                + A2.dot((u + 439./216.*k1 - 8*k2 + 3680./513.*k3 - 845./4104.*k4)**m2))
        k6 = dt * (A1.dot((u - 8./27.*k1 + 2*k2 - 3544./2565.*k3 + 1859./4140.*k4 - 11./40.*k5)**m1) \
                + A2.dot((u - 8./27.*k1 + 2*k2 - 3544./2565.*k3 + 1859./4140.*k4 - 11./40.*k5)**m2))

        ### Solution
        u_rkf4 = u + (25./216.*k1 + 1408./2565.*k3 + 2197./4101.*k4 - 1./5.*k5)
        u_rkf5 = u + (16./135.*k1 + 6656./12825.*k3 + 28561./56430.*k4 - 9./50.*k5 + 2./55.*k6)

        return u_rkf4, 10, u_rkf5, 12
    
    else:
        print("Error!!! Check number of input matrices!!")


################################################################################################
