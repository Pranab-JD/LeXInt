"""
Created on Wed Aug 19 16:17:29 2020

@author: Pranab JD

Description: -
        Contains explicit integrators for 2-matrix equations
        (du/dt = A1.u^m1 + A2.u^m2)

"""

################################################################################################

### Explicit Integrators ###

def RK2(A1, m1, A2, m2, u, dt):
    """
    Parameters
    ----------
    A1      : N x N matrix 1
    m1      : Index of u for A1
    A2      : N x N matrix 2
    m2      : Index of u for A2
    u       : 1D vector u (input)
    dt      : Step size

    Returns
    -------
    u_rk2   : 1D vector u (output) after time dt (2nd order)
    4       : # of matrix-vector products

    """

    k1 = dt * (A1.dot(u**m1) + A2.dot(u**m2))
    k2 = dt * (A1.dot((u + k1)**m1) + A2.dot((u + k1)**m2))

    ## Solution
    u_rk2 = u + 1./2. * (k1 + k2)

    return u_rk2, 2

##############################################################################

def RK4(A1, m1, A2, m2, u, dt):
    """
    Parameters
    ----------
    A1      : N x N matrix 1
    m1      : Index of u for A1
    A2      : N x N matrix 2
    m2      : Index of u for A2
    u       : 1D vector u (input)
    dt      : Step size

    Returns
    -------
    u_rk4   : 1D vector u (output) after time dt (4th order)
    8       : # of matrix-vector products

    """

    k1 = dt * (A1.dot(u**m1) + A2.dot(u**m2))
    k2 = dt * (A1.dot((u + k1/2)**m1) + A2.dot((u + k1/2)**m2))
    k3 = dt * (A1.dot((u + k2/2)**m1) + A2.dot((u + k1/2)**m2))
    k4 = dt * (A1.dot((u + k3)**m1) + A2.dot((u + k1/2)**m2))

    ## Solution
    u_rk4 = u + 1./6.*(k1 + 2*k2 + 2*k3 + k4)

    return u_rk4, 8

##############################################################################

def RKF45(A1, m1, A2, m2, u, dt):
    """
    Parameters
    ----------
    A1      : N x N matrix 1
    m1      : Index of u for A1
    A2      : N x N matrix 2
    m2      : Index of u for A2
    u       : 1D vector u (input)
    dt      : Step size

    Returns
    -------
    u_rkf4  : 1D vector u (output) after time dt (4th order)
    10      : # of matrix-vector products for u_rkf4
    u_rkf5  : 1D vector u (output) after time dt (5th order)
    12      : # of matrix-vector products for u_rkf5

    Note: |u_rkf5 - u_rkf4| gives 4th order error estimate 

    """

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

################################################################################################