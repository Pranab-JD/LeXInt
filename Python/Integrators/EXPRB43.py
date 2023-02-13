import sys
sys.path.insert(1, "../")

from Jacobian import *
from Phi_functions import *
from real_Leja_phi import *
from imag_Leja_phi import *

################################################################################################

def EXPRB43(u, dt, RHS_function, c, Gamma, Leja_X, tol, Real_Imag):
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
    u_exprb3                : numpy array
                                Output state variable(s) after time dt (3rd order)
    u_exprb4                : numpy array
                                Output state variable(s) after time dt (4th order)
    num_rhs_calls           : int
                                # of RHS calls
    
    Reference:
    
        M. Hochbruck, A. Ostermann, Exponential Integrators, Acta Numer. 19 (2010) 209-286. 
        doi:10.1017/S0962492910000048.

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

	###? Vertical interpolation of f(u) at 1/2 and 1; phi_1({1/2, 1} J(u) dt) f(u) dt
    u_flux, rhs_calls_1, convergence = Leja_phi(u, dt, RHS_function, RHS_function(u)*dt, [1/2, 1], c, Gamma, Leja_X, phi_1, tol)

    ###? If it does not converge, return (try with smaller dt)
    if convergence == 0:
        return u, 2.1*u, rhs_calls_1

    ###? Internal stage 1; a = u + 1/2 phi_1(1/2 J(u) dt) f(u) dt
    a = u + (1/2 * u_flux[:, 0])

    ###? Nonlinear remainder at u and a
    Nonlinear_u = RHS_function(u) - Jacobian(RHS_function, u, u)
    Nonlinear_a = RHS_function(a) - Jacobian(RHS_function, u, a)
    R_a = Nonlinear_a - Nonlinear_u

    ###? phi_1(J(u) dt) R(a) dt
    b_n_nl, rhs_calls_2, _ = Leja_phi(u, dt, RHS_function, R_a*dt, [1], c, Gamma, Leja_X, phi_1, tol)

    ###? Internal stage 2; b = u + phi_1(J(u) dt) f(u) dt + phi_1(J(u) dt) R(a) dt
    b = u + u_flux[:, 1] + b_n_nl[:, 0]

    ###? Nonlinear remainder at b
    Nonlinear_b = RHS_function(b) - Jacobian(RHS_function, u, b)
    R_b = Nonlinear_b - Nonlinear_u

    ###? phi_3(J(u) dt) (16R(a) - 2R(b)) dt
    u_nl_3, rhs_calls_3, _ = Leja_phi(u, dt, RHS_function, (16*R_a - 2*R_b)*dt, [1], c, Gamma, Leja_X, phi_3, tol)
    
    ###? phi_4(J(u) dt) (-48R(a) + 12R(b)) dt
    u_nl_4, rhs_calls_4, _ = Leja_phi(u, dt, RHS_function, (-48*R_a + 12*R_b)*dt, [1], c, Gamma, Leja_X, phi_4, tol)

    ###? 3rd order solution; u_3 = u + phi_1(J(u) dt) f(u) dt + phi_3(J(u) dt) (16R(a) - 2R(b)) dt
    u_exprb3 = u + u_flux[:, 1] + u_nl_3[:, 0]
    
    ###? 4th order solution; u_4 = u_3 + phi_4(J(u) dt) (-48R(a) + 12R(b)) dt
    u_exprb4 = u_exprb3 + u_nl_4[:, 0]

    ###? Proxy of computational cost
    num_rhs_calls = rhs_calls_1 + rhs_calls_2 + rhs_calls_3 + rhs_calls_4 + 13

    return u_exprb3, u_exprb4, num_rhs_calls