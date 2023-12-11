import numpy as np

###! LeXInt functions
from Jacobian import Jacobian
from linear_phi import linear_phi

################################################################################################

def EPIRK4s3A(u, T_final, RHS_function, c, Gamma, Leja_X, tol, Real_Imag):
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
    u_epirk4                : numpy array
                                Output state variable(s) after time dt (4th order)
    num_rhs_calls           : int
                                # of RHS calls
    
    Reference: 
    
        G. Rainwater and M. Tokman
        A new approach to constructing efficient stiffly accurate EPIRK methods, J. Comput. Phys. 323 (2016) 283-309
        doi:10.1016/j.jcp.2016.07.026

    """
    
    ###? RHS evaluated at 'u'
    rhs_u = RHS_function(u)

    ###? Array of zeros vectors
    zero_vec = np.zeros(np.shape(u))
    
    ###? dt * J(u).z
    Jac_vec = lambda z: T_final * Jacobian(RHS_function, u, z, rhs_u)
    
    ###? J(u) . u
    Jacobian_u = Jacobian(RHS_function, u, u, rhs_u)

    ###? Interpolations 1 & 2; {1/2, 2/3} phi_1({1/2, 2/3} J(u) dt) f(u) dt
    u_flux_1, rhs_calls_1 = linear_phi([zero_vec, rhs_u*T_final], T_final, Jac_vec, 1/2, c, Gamma, Leja_X, tol)
    u_flux_2, rhs_calls_2 = linear_phi([zero_vec, rhs_u*T_final], T_final, Jac_vec, 2/3, c, Gamma, Leja_X, tol)

    ###? Internal stage 1; a = u + 1/2 phi_1(1/2 J(u) dt) f(u) dt
    a = u + u_flux_1

    ###? Internal stage 2; b = u + 2/3 phi_1(2/3 J(u) dt) f(u) dt
    b = u + u_flux_2

    ###? Difference of nonlinear remainders at a and b
    R_a = (RHS_function(a) - Jacobian(RHS_function, u, a, rhs_u)) - (rhs_u - Jacobian_u)
    R_b = (RHS_function(b) - Jacobian(RHS_function, u, b, rhs_u)) - (rhs_u - Jacobian_u)

    ###? Interpolation 2; phi_1(J(u) dt) f(u) dt + phi_3(J(u) dt) (32*R(a) - 27/2*R(b)) dt + phi_4(J(u) dt) (-144*R(a) + 81*R(b)) dt
    u_flux, rhs_calls_3 = linear_phi([zero_vec, rhs_u*T_final, zero_vec, (32*R_a-27/2*R_b)*T_final, (-144*R_a+81*R_b)*T_final], T_final, Jac_vec, 1, c, Gamma, Leja_X, tol)
    
    ###? 4th order solution; u_4 = u + phi_1(J(u) dt) f(u) dt + phi_3(J(u) dt) (32R(a) - (27/2)R(b)) dt + phi_4(J(u) dt) (-144R(a) + 81R(b)) dt
    u_epirk4 = u + u_flux

    ###? Proxy of computational cost
    num_rhs_calls = rhs_calls_1 + rhs_calls_2 + rhs_calls_3 + 6

    return u_epirk4, num_rhs_calls