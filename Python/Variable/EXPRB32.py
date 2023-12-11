import numpy as np

###! LeXInt functions
from Jacobian import Jacobian
from linear_phi import linear_phi

################################################################################################

def EXPRB32(u, T_final, RHS_function, c, Gamma, Leja_X, tol, Real_Imag):
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
    u_exprb2                : numpy array
                                Output state variable(s) after time dt (2nd order)
    u_exprb3                : numpy array
                                Output state variable(s) after time dt (3rd order)
    num_rhs_calls           : int
                                # of RHS calls
    
    Reference:
    
        M. Hochbruck, A. Ostermann, Exponential Integrators, Acta Numer. 19 (2010) 209-286. 
        doi:10.1017/S0962492910000048

    """
    
    ###? RHS evaluated at 'u'
    rhs_u = RHS_function(u)

    ###? Array of zeros vectors
    zero_vec = np.zeros(np.shape(u))
    
    ###? dt * J(u).z
    Jac_vec = lambda z: T_final * Jacobian(RHS_function, u, z, rhs_u)
    
    ###? Interpolation 1; phi_1(J(u) dt) f(u) dt
    u_flux, rhs_calls_1 = linear_phi([zero_vec, rhs_u*T_final], T_final, Jac_vec, 1, c, Gamma, Leja_X, tol)

    ###? Internal stage; 2nd order solution; u_2 = u + phi_1(J(u) dt) f(u) dt
    u_exprb2 = u + u_flux

    ###? Difference of nonlinear remainders at u_exprb2
    R_a = (RHS_function(u_exprb2) - Jacobian(RHS_function, u, u_exprb2, rhs_u)) - (rhs_u - Jacobian(RHS_function, u, u, rhs_u))

    ###? Interpolation 2; phi_3(J(u) dt) R(a) dt
    u_nl, rhs_calls_2 = linear_phi([zero_vec, zero_vec, zero_vec, 2*R_a*T_final], T_final, Jac_vec, 1, c, Gamma, Leja_X, tol)
    
    ###? 3rd order solution; u_3 = u_2 + 2 phi_3(J(u) dt) R(a) dt
    u_exprb3 = u_exprb2 + u_nl

    ###? Proxy of computational cost
    num_rhs_calls = rhs_calls_1 + rhs_calls_2 + 5

    return u_exprb2, u_exprb3, num_rhs_calls