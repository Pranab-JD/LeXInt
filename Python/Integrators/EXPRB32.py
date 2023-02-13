import sys
sys.path.insert(1, "../")

from Jacobian import *
from Phi_functions import *
from real_Leja_phi import *
from imag_Leja_phi import *

################################################################################################

def EXPRB32(u, dt, RHS_function, c, Gamma, Leja_X, tol, Real_Imag):
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
    u_exprb2                : numpy array
                                Output state variable(s) after time dt (2nd order)
    u_exprb3                : numpy array
                                Output state variable(s) after time dt (3rd order)
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

    ###? Interpolation of f(u) at 1
    u_flux, rhs_calls_1, convergence = Leja_phi(u, dt, RHS_function, RHS_function(u)*dt, [1], c, Gamma, Leja_X, phi_1, tol)
    
    ## If it does not converge, return (try with smaller dt)
    if convergence == 0:
        return u, 2.1*u, rhs_calls_1

    ###? Internal stage; 2nd order solution; u_2 = u + phi_1(J(u) dt) f(u) dt
    u_exprb2 = u + u_flux[:, 0]

    ###? Difference of nonlinear remainders at a
    R_a = (RHS_function(u_exprb2) - Jacobian(RHS_function, u, u_exprb2)) - (RHS_function(u) - Jacobian(RHS_function, u, u))

    ###? phi_3(J(u) dt) R(a) dt
    u_nl_3, rhs_calls_2, _ = Leja_phi(u, dt, RHS_function, R_a*dt, [1], c, Gamma, Leja_X, phi_3, tol)
    
    ###? 3rd order solution; u_3 = u_2 + 2 phi_3(J(u) dt) R(a) dt
    u_exprb3 = u_exprb2 + (2*u_nl_3[:, 0])

    ###? Proxy of computational cost
    num_rhs_calls = rhs_calls_1 + rhs_calls_2 + 9

    return u_exprb2, u_exprb3, num_rhs_calls