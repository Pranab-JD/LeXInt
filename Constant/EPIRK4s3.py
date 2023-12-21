import sys
sys.path.insert(1, "../")

import numpy as np

###! LeXInt functions
from Jacobian import Jacobian
from linear_phi import linear_phi

################################################################################################

def EPIRK4s3(u, T_final, substeps, RHS_function, c, Gamma, Leja_X, tol, Real_Imag):
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
    
    References:
    
        1. D. L. Michels, V. T. Luan, M. Tokman
        A stiffly accurate integrator for elastodynamic problems, ACM Trans. Graph. 36 (4) (2017)
        doi:10.1145/3072959.3073706.
        
        2. G. Rainwater and M. Tokman
        Designing efficient exponential integrators with EPIRK framework,
        in: International Conference of Numerical Analysis and Applied Mathematics (ICNAAM 2016), Vol. 1863 of American Institute of Physics Conference Series, 2017, p. 020007
        doi:10.1063/1.4992153

    """
    
    ###? RHS evaluated at 'u'
    rhs_u = RHS_function(u)

    ###? Array of zeros vectors
    zero_vec = np.zeros(np.shape(u))
    
    ###? dt * J(u).z
    Jac_vec = lambda z: T_final * Jacobian(RHS_function, u, z, rhs_u)
    
    ###? J(u) . u
    Jacobian_u = Jacobian(RHS_function, u, u, rhs_u)
    
    ###? Interpolations 1 & 2; {1/8, 1/9} phi_1({1/8, 1/9} J(u) dt) f(u) dt
    u_flux_1, rhs_calls_1, substeps = linear_phi([zero_vec, rhs_u*T_final], T_final, substeps, Jac_vec, 1/8, c, Gamma, Leja_X, tol)
    u_flux_2, rhs_calls_2, substeps = linear_phi([zero_vec, rhs_u*T_final], T_final, substeps, Jac_vec, 1/9, c, Gamma, Leja_X, tol)

    ###? Internal stage 1; a = u + 1/8 phi_1(1/8 J(u) dt) f(u) dt
    a = u + u_flux_1
    
    ###? Internal stage 2; b = u + 1/9 phi_1(1/9 J(u) dt) f(u) dt
    b = u + u_flux_2

    ###? Difference of nonlinear remainders at a and b
    R_a = (RHS_function(a) - Jacobian(RHS_function, u, a, rhs_u)) - (rhs_u - Jacobian_u)
    R_b = (RHS_function(b) - Jacobian(RHS_function, u, b, rhs_u)) - (rhs_u - Jacobian_u)

    ###? phi_1(J(u) dt) f(u) dt + phi_3(J(u) dt) (1892*R(a) + 1458*(R(b) - 2*R(a))) dt + phi_4(J(u) dt) (-42336*R(a) - 34992*(R(b) - 2*R(a))) dt
    u_flux, rhs_calls_3, substeps = linear_phi([zero_vec, rhs_u*T_final, zero_vec, (1892*R_a + 1458*(R_b - 2*R_a))*T_final, (-42336*R_a - 34992*(R_b - 2*R_a))*T_final], T_final, substeps, Jac_vec, 1, c, Gamma, Leja_X, tol)
    
    ###? 4th order solution; u_4 = u + phi_1(J(u) dt) f(u) dt + phi_3(J(u) dt) (-1024R(a) + 1458R(b)) dt + phi_4(J(u) dt) (27648R(a) - 34992R(b)) dt
    u_epirk4 = u + u_flux

    ###? Proxy of computational cost
    num_rhs_calls = rhs_calls_1 + rhs_calls_2 + rhs_calls_3 + 6

    return u_epirk4, num_rhs_calls, substeps