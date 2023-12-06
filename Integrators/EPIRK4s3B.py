import sys
sys.path.insert(1, "../")

from Jacobian import *
from Phi_functions import *
from real_Leja_phi import *
from imag_Leja_phi import *

################################################################################################

def EPIRK4s3B(u, dt, RHS_function, c, Gamma, Leja_X, tol, Real_Imag):
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
    u_epirk4                : numpy array
                                Output state variable(s) after time dt (4th order)
    num_rhs_calls           : int
                                # of RHS calls
    
    Reference:
    
        G. Rainwater, M. Tokman, A new approach to constructing efficient stiffly accurate EPIRK methods, J. Comput. Phys. 323 (2016) 283-309.
        doi:10.1016/j.jcp.2016.07.026.

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
    
    ###? f(u)
    rhs_u = RHS_function(u)

	###? Vertical interpolation of f(u) at 1/2 and 3/4; phi_2({1/2, 3/4} J(u) dt) f(u) dt
    u_flux, rhs_calls_1, convergence = Leja_phi(u, dt, RHS_function, rhs_u*dt, [1/2, 3/4], c, Gamma, Leja_X, phi_2, tol)

    ###? Internal stage 1; a = u + 2/3 phi_2(1/2 J(u) dt) f(u) dt
    a = u + (2/3 * u_flux[:, 0])
    
    ###? Internal stage 2; b = u + phi_2(3/4 J(u) dt) f(u) dt
    b = u + u_flux[:, 1]

    ###? Nonlinear remainder at u, a, and b
    Nonlinear_u = rhs_u - Jacobian(RHS_function, u, u)
    Nonlinear_a = RHS_function(a) - Jacobian(RHS_function, u, a)
    Nonlinear_b = RHS_function(b) - Jacobian(RHS_function, u, b)
    
    R_a = Nonlinear_a - Nonlinear_u
    R_b = Nonlinear_b - Nonlinear_u
    
    ###? phi_1(J(u) dt) f(u) dt
    u_1, rhs_calls_2, _    = Leja_phi(u, dt, RHS_function, rhs_u*dt, [1], c, Gamma, Leja_X, phi_1, tol)
    
    ###? phi_3(J(u) dt) (54*R(a) - 16*R(b)) dt
    u_nl_3, rhs_calls_3, _ = Leja_phi(u, dt, RHS_function, (54*R_a - 16*R_b)*dt, [1], c, Gamma, Leja_X, phi_3, tol)
    
    ###? phi_4(J(u) dt) (-324*R(a) + 144*R(b)) dt
    u_nl_4, rhs_calls_4, _ = Leja_phi(u, dt, RHS_function, (-324*R_a + 144*R_b)*dt, [1], c, Gamma, Leja_X, phi_4, tol)
 
    ###? 4th order solution; u_4 = u + phi_1(J(u) dt) f(u) dt + phi_3(J(u) dt) (54R(a) - 16R(b)) dt + phi_4(J(u) dt) (-324R(a) + 144R(b)) dt
    u_epirk4 = u + u_1[:, 0] + u_nl_3[:, 0] + u_nl_4[:, 0]

    ###? Proxy of computational cost
    num_rhs_calls = rhs_calls_1 + rhs_calls_2 + rhs_calls_3 + rhs_calls_4 + 9

    return u_epirk4, num_rhs_calls