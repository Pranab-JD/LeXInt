import sys
sys.path.insert(1, "../")

from Jacobian import *
from Phi_functions import *
from real_Leja_phi import *
from imag_Leja_phi import *

################################################################################################

def EXPRB42(u, dt, RHS_function, c, Gamma, Leja_X, tol, Real_Imag):
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
    u_exprb4                : numpy array
                                Output state variable(s) after time dt (4th order)
    num_rhs_calls           : int
                                # of RHS calls
    
    Reference:
    
        V. T. Luan, Fourth-order two-stage explicit exponential integrators for time-dependent PDEs, Appl. Numer. Math. 112 (2017) 91-103. 
        doi:10.1016/j.apnum.2016.10.008.

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

    ###? Vertical interpolation of f(u) at 3/4 and 1; phi_1({3/4, 1} J(u) dt) f(u) dt
    u_flux, rhs_calls_1, convergence = Leja_phi(u, dt, RHS_function, RHS_function(u)*dt, [3/4, 1], c, Gamma, Leja_X, phi_1, tol)

    ###? Internal stage 1; a = u + 3/4 phi_1(3/4 J(u) dt) f(u) dt
    a = u + (3/4 * u_flux[:, 0])

    ###? Difference of nonlinear remainders at a
    R_a = (RHS_function(a) - Jacobian(RHS_function, u, a)) - (RHS_function(u) - Jacobian(RHS_function, u, u))

    ###? phi_3(J(u) dt) R(a) dt
    u_nl_3, rhs_calls_2, _ = Leja_phi(u, dt, RHS_function, R_a*dt, [1], c, Gamma, Leja_X, phi_3, tol)
    
    ###? 3rd order solution; u_4 = u + phi_1(J(u) dt) f(u) dt + 32/9 phi_3(J(u) dt) R(a) dt
    u_exprb4 = u + u_flux[:, 1] + (32/9 * u_nl_3[:, 0])

    ###? Proxy of computational cost
    num_rhs_calls = rhs_calls_1 + rhs_calls_2 + 9

    return u_exprb4, num_rhs_calls