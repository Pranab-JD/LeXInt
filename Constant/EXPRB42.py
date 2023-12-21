import sys
sys.path.insert(1, "../")

import numpy as np

###! LeXInt functions
from Jacobian import Jacobian
from linear_phi import linear_phi

################################################################################################

def EXPRB42(u, T_final, substeps, RHS_function, c, Gamma, Leja_X, tol, Real_Imag):
    """
    Parameters
    ----------
    u                       : numpy array
                                State variable(s)
    T_final                 : double
                                Step size
    RHS_function            : user-defined function 
                                RHS function
    c                       : double
                                Shifting factor
    Gamma                   : double
                                Scaling factor
    Leja_X                  : numpy array
                                Array of Leja points
    tol                     : double
                                Accuracy of the polynomial so formed
    Real_Imag               : int
                                0 - Real, 1 - Imaginary

    Returns
    -------
    u_exprb4                : numpy array
                                Output state variable(s) after time dt (4th order)
    num_rhs_calls           : int
                                # of RHS calls
    
    Reference:
    
        V. T. Luan
        Fourth-order two-stage explicit exponential integrators for time-dependent PDEs, Appl. Numer. Math. 112 (2017) 91-103
        doi:10.1016/j.apnum.2016.10.008

    """

    ###? RHS evaluated at 'u'
    rhs_u = RHS_function(u)

    ###? Array of zeros vectors
    zero_vec = np.zeros(np.shape(u))
    
    ###? dt * J(u).z
    Jac_vec = lambda z: T_final * Jacobian(RHS_function, u, z, rhs_u)
    
    ###? J(u) . u
    Jacobian_u = Jacobian(RHS_function, u, u, rhs_u)
    
    ###? Interpolation of RHS(u) at 3/4; 3/4 phi_1(3/4 J(u) dt) f(u) dt
    u_flux_1, rhs_calls_1, substeps = linear_phi([zero_vec, rhs_u*T_final], T_final, substeps, Jac_vec, 3/4, c, Gamma, Leja_X, tol)

    ###? Internal stage 1; a = u + 3/4 phi_1(3/4 J(u) dt) f(u) dt
    a = u + u_flux_1

    ###? Difference of nonlinear remainders at a
    R_a = (RHS_function(a) - Jacobian(RHS_function, u, a, rhs_u)) - (rhs_u - Jacobian_u)

    ###? Interpolation 2: phi_1(J(u) dt) f(u) dt + phi_3(J(u) dt) R(a) dt
    u_flux, rhs_calls_2, substeps = linear_phi([zero_vec, rhs_u*T_final, zero_vec, 32/9*R_a*T_final], T_final, substeps, Jac_vec, 1, c, Gamma, Leja_X, tol)
    
    ###? 3rd order solution; u_4 = u + phi_1(J(u) dt) f(u) dt + 32/9 phi_3(J(u) dt) R(a) dt
    u_exprb4 = u + u_flux

    ###? Proxy of computational cost
    num_rhs_calls = rhs_calls_1 + rhs_calls_2 + 4

    return u_exprb4, num_rhs_calls, substeps