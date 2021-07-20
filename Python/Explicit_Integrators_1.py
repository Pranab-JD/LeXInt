"""
Created on Wed Sep 04 16:15:14 2020

@author: Pranab JD

Description: -
        Contains explicit integrators for 1-matrix equations 
        (du/dt = A.u^m)

"""

################################################################################################

### Explicit Integrators ###

def RK2(A, m, u, dt):
    """
    Parameters
    ----------
    A       : N x N matrix (A)
    m       : Index of u (u^m)
    u       : 1D vector u (input)
    dt      : Step size

    Returns
    -------
    u_rk2   : 1D vector u (output) after time dt (2nd order)
    2       : # of matrix-vector products

    """

    k1 = dt * (A.dot(u**m))
    k2 = dt * (A.dot((u + k1)**m))

    ## Solution
    u_rk2 = u + 1./2. * (k1 + k2)

    return u_rk2, 2

##############################################################################

def RK4(A, m, u, dt):
    """
    Parameters
    ----------
    A       : N x N matrix (A)
    m       : Index of u (u^m)
    u       : 1D vector u (input)
    dt      : Step size

    Returns
    -------
    u_rk4   : 1D vector u (output) after time dt (4th order)
    2       : # of matrix-vector products

    """

    k1 = dt * (A.dot(u**m))
    k2 = dt * (A.dot((u + k1/2)**m))
    k3 = dt * (A.dot((u + k2/2)**m))
    k4 = dt * (A.dot((u + k3)**m))

    ## Solution
    u_rk4 = u + 1./6.*(k1 + 2*k2 + 2*k3 + k4)

    return u_rk4, 4

##############################################################################

def RKF45(A, m, u, dt):
    """
    Parameters
    ----------
    A       : N x N matrix (A)
    m       : Index of u (u^m)
    u       : 1D vector u (input)
    dt      : Step size

    Returns
    -------
    u_rkf4  : 1D vector u (output) after time dt (4th order)
    5       : # of matrix-vector products for u_rkf4
    u_rkf5  : 1D vector u (output) after time dt (5th order)
    6       : # of matrix-vector products for u_rkf5

    """

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

################################################################################################