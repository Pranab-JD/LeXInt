import sys
sys.path.insert(1, "../")

from Jacobian import *
from Phi_functions import *
from real_Leja_phi import *
from imag_Leja_phi import *

################################################################################################

def EPIRK4s3(u, dt, RHS_function, c, Gamma, Leja_X, tol, Real_Imag):
    """
    Parameters
    ----------
    u                       : numpy array
                                State variable(s)
    dt                      : double
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
    u_epirk3                : numpy array
                                Output state variable(s) after time dt (3rd order)
    u_epirk4                : numpy array
                                Output state variable(s) after time dt (4th order)
    num_rhs_calls           : int
                                # of RHS calls
    
    References:
    
        1. D. L. Michels, V. T. Luan, M. Tokman, A stiffly accurate integrator for elastodynamic problems, ACM Trans. Graph. 36 (4) (2017). 
        doi:10.1145/3072959.3073706.
        
        2. G. Rainwater, M. Tokman, Designing efficient exponential integrators with EPIRK framework, in: International Conference of Numerical
        Analysis and Applied Mathematics (ICNAAM 2016), Vol. 1863 of American Institute of Physics Conference Series, 2017, p. 020007.
        doi:10.1063/1.4992153.

    """
    
    ############## --------------------- ##############

    ###? Interpolate on either real Leja or imaginary Leja points
    if Real_Imag == 0:
        Leja_phi = real_Leja_phi
    elif Real_Imag == 1:
        Leja_phi = imag_Leja_phi
    else:
        print("Error!! Choose 0 for real or 1 for imaginary Leja points.")
    
    ############## --------------------- ##############

	###? Vertical interpolation of f(u) at 1/8, 1/9, and 1; phi_1({1/8, 1/9, 1} J(u) dt) f(u) dt
    u_flux, rhs_calls_1, convergence = Leja_phi(u, dt, RHS_function, RHS_function(u)*dt, [1/8, 1/9, 1], c, Gamma, Leja_X, phi_1, tol)

    ###! If it does not converge, return (try with smaller dt)
    if convergence == 0:
        return u, 2.1*u, rhs_calls_1

    ###? Internal stage 1; a = u + 1/8 phi_1(1/8 J(u) dt) f(u) dt
    a = u + (1/8 * u_flux[:, 0])
    
    ###? Internal stage 2; b = u + 1/9 phi_1(1/9 J(u) dt) f(u) dt
    b = u + (1/9 * u_flux[:, 1])

    ###? Nonlinear remainder at u, a, and b
    Nonlinear_u = RHS_function(u) - Jacobian(RHS_function, u, u)
    Nonlinear_a = RHS_function(a) - Jacobian(RHS_function, u, a)
    Nonlinear_b = RHS_function(b) - Jacobian(RHS_function, u, b)
    
    R_a = Nonlinear_a - Nonlinear_u
    R_b = Nonlinear_b - Nonlinear_u
    
    ###? phi_3(J(u) dt) (1892*R(a) + 1458*(R(b) - 2*R(a))) dt
    u_nl_3, rhs_calls_2, _ = Leja_phi(u, dt, RHS_function, (1892*R_a + 1458*(R_b - 2*R_a))*dt, [1], c, Gamma, Leja_X, phi_3, tol)
    
    ###? phi_4(J(u) dt) (-42336*R(a) - 34992*(R(b) - 2*R(a))) dt
    u_nl_4, rhs_calls_3, _ = Leja_phi(u, dt, RHS_function, (-42336*R_a - 34992*(R_b - 2*R_a))*dt, [1], c, Gamma, Leja_X, phi_4, tol)
 
    ###? 3rd order solution; u_3 = u + phi_1(J(u) dt) f(u) dt + phi_3(J(u) dt) (-1024R(a) + 1458R(b)) dt
    u_epirk3 = u + u_flux[:, 2] + u_nl_3[:, 0]
    
    ###? 4th order solution; u_4 = u_3 + phi_4(J(u) dt) (27648R(a) - 34992R(b)) dt
    u_epirk4 = u_epirk3 + u_nl_4[:, 0]

    ###? Proxy of computational cost
    num_rhs_calls = rhs_calls_1 + rhs_calls_2 + rhs_calls_3 + 13

    return u_epirk3, u_epirk4, num_rhs_calls