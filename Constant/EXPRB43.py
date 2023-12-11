import numpy as np

###! LeXInt functions
from Jacobian import Jacobian
from linear_phi import linear_phi

################################################################################################

def EXPRB43(u, T_final, RHS_function, c, Gamma, Leja_X, tol, Real_Imag):
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
    
        M. Hochbruck and A. Ostermann
        Exponential Integrators, Acta Numer. 19 (2010) 209-286
        doi:10.1017/S0962492910000048

    """
    
    ###? RHS evaluated at 'u'
    rhs_u = RHS_function(u)

    ###? Array of zeros vectors
    zero_vec = np.zeros(np.shape(u))
    
    ###? dt * J(u).z
    Jac_vec = lambda z: T_final * Jacobian(RHS_function, u, z, rhs_u)
    
    ###? J(u) . u
    Jacobian_u = Jacobian(RHS_function, u, u, rhs_u)
    
    ###? Interpolation 1; 1/2 phi_1(1/2 J(u) dt) f(u) dt
    u_flux_1, rhs_calls_1 = linear_phi([zero_vec, rhs_u*T_final], T_final, Jac_vec, 1/2, c, Gamma, Leja_X, tol)

    ###? Internal stage 1; a = u + 1/2 phi_1(1/2 J(u) dt) f(u) dt
    a = u + u_flux_1

    ###? Difference of nonlinear remainder at a
    R_a = (RHS_function(a) - Jacobian(RHS_function, u, a, rhs_u)) - (rhs_u - Jacobian_u)

    ###? Interpolation 2; phi_1(J(u) dt) f(u) dt + phi_1(J(u) dt) R(a) dt
    u_flux_2, rhs_calls_2 = linear_phi([zero_vec, (rhs_u + R_a)*T_final], T_final, Jac_vec, 1, c, Gamma, Leja_X, tol)

    ###? Internal stage 2; b = u + phi_1(J(u) dt) f(u) dt + phi_1(J(u) dt) R(a) dt
    b = u + u_flux_2

    ###? Nonlinear remainder at b
    R_b = (RHS_function(b) - Jacobian(RHS_function, u, b, rhs_u)) - (rhs_u - Jacobian_u)

    ###? Interpolation 3; phi_1(J(u) dt) f(u) dt + phi_3(J(u) dt) (16R(a) - 2R(b)) dt + phi_4(J(u) dt) (-48R(a) + 12R(b)) dt
    u_flux, rhs_calls_3 = linear_phi([zero_vec, rhs_u*T_final, zero_vec, (16*R_a-2*R_b)*T_final, (-48*R_a+12*R_b)*T_final], T_final, Jac_vec, 1, c, Gamma, Leja_X, tol)
    
    ###? 4th order solution; u_4 = u + phi_1(J(u) dt) f(u) dt + phi_3(J(u) dt) (16R(a) - 2R(b)) dt + phi_4(J(u) dt) (-48R(a) + 12R(b)) dt
    u_exprb4 = u + u_flux

    ###? Proxy of computational cost
    num_rhs_calls = rhs_calls_1 + rhs_calls_2 + rhs_calls_3 + 6

    return u_exprb4, num_rhs_calls