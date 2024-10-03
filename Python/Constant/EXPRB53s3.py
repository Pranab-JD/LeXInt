import sys
sys.path.insert(1, "../")

import numpy as np

###! LeXInt functions
from Jacobian import Jacobian
from linear_phi import linear_phi

################################################################################################

def EXPRB53s3(u, T_final, substeps, RHS_function, c, Gamma, Leja_X, tol, Real_Imag):
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
    u_exprb5                : numpy array
                                Output state variable(s) after time dt (5th order)
    num_rhs_calls           : int
                                # of RHS calls
    
    Reference:
    
        V. T. Luan and A. Ostermann
        Exponential Rosenbrock methods of order five - construction, analysis and numerical comparisons, J. Comput. Appl. Math. 255 (2014) 417-431
        doi:10.1016/j.cam.2013.04.041

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
    u_flux_1, rhs_calls_1, substeps = linear_phi([zero_vec, rhs_u*T_final], T_final, substeps, Jac_vec, 1/2, c, Gamma, Leja_X, tol)

    ###? Internal stage 1; a = u + 1/2 phi_1(1/2 J(u) dt) f(u) dt
    a = u + u_flux_1
    
    ###? Difference of nonlinear remainder at a
    R_a = (RHS_function(a) - Jacobian(RHS_function, u, a, rhs_u)) - (rhs_u - Jacobian_u)
    
    ###? Interpolation 2a; 9/10 phi_1(9/10 J(u) dt) f(u) dt + 729/125 phi_3(9/10 J(u) dt)) R(a) dt
    u_flux_2a, rhs_calls_2a, substeps = linear_phi([zero_vec, rhs_u*T_final, zero_vec, 10/9*729/125*R_a*T_final], T_final, substeps, Jac_vec, 9/10, c, Gamma, Leja_X, tol)
    
    ###? Interpolation 2b; 27/25 phi_3(1/2 J(u) dt
    u_flux_2b, rhs_calls_2b, substeps = linear_phi([zero_vec, zero_vec, zero_vec, 2*27/25*R_a*T_final], T_final, substeps, Jac_vec, 1/2, c, Gamma, Leja_X, tol)

    ###? b = u + 9/10 phi_1(9/10 J(u) dt) f(u) dt + 27/25 phi_3(1/2 J(u) dt + 729/125 phi_3(9/10 J(u) dt)) R(a) dt
    b = u + u_flux_2a + u_flux_2b
    
    ###? Nonlinear remainder at b
    R_b = (RHS_function(b) - Jacobian(RHS_function, u, b, rhs_u)) - (rhs_u - Jacobian_u)
    
    ###? Interpolation 3; phi_1(J(u) dt) f(u) dt + phi_3(J(u) dt) (18R(a) - (250/81)R(b)) dt + phi_4(J(u) dt) (-60R(a) + (500/27)R(b)) dt
    u_flux, rhs_calls_3, substeps = linear_phi([zero_vec, rhs_u*T_final, zero_vec, (18*R_a - (250/81)*R_b)*T_final, (-60*R_a + (500/27)*R_b)*T_final], T_final, substeps, Jac_vec, 1, c, Gamma, Leja_X, tol)
    
    ###? 5th order solution; u_5 = u + phi_1(J(u) dt) f(u) dt + phi_3(J(u) dt) (18R(a) - (250/81)R(b)) dt + phi_4(J(u) dt) (-60R(a) + (500/27)R(b)) dt
    u_exprb5 = u + u_flux

    ###? Proxy of computational cost
    num_rhs_calls = rhs_calls_1 + rhs_calls_2a + rhs_calls_2b + rhs_calls_3 + 6

    return u_exprb5, num_rhs_calls, substeps