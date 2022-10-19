import sys
sys.path.insert(1, "../")

from Phi_functions import *
from real_Leja_phi import *
from imag_Leja_phi import *

################################################################################################

def Rosenbrock_Euler(u, dt, RHS_function, c, Gamma, Leja_X, tol, Real_Imag):
    """
    Parameters
    ----------
    u               : 1D vector u (input)
    dt              : Step size
    RHS_function    : RHS function
    c               : Shifting factor
    Gamma           : Scaling factor
    Leja_X          : Array of Leja points
    tol             : Accuracy of the polynomial so formed
    Real_Imag       : 0 - Real, 1 - Imaginary

    Returns
    -------
    u_roseu         : 1D vector u (output) after time dt (2nd order)
    num_rhs_calls   : # of RHS calls
    
    Reference:
        D. A. Pope, An exponential method of numerical integration of ordinary differential equations, Commun. ACM 6 (8) (1963) 491-493.
        doi:10.1145/366707.367592.

    """

    ### Interpolate on either real Leja or imaginary Leja points.
    if Real_Imag == 0:
        Leja_phi = real_Leja_phi
    elif Real_Imag == 1:
        Leja_phi = imag_Leja_phi
    else:
        print("Error!! Choose 0 for real or 1 for imaginary Leja points.")

    ### Stage 1; interpolation of RHS_function(u) at 1
    u_flux, rhs_calls_1, convergence = Leja_phi(u, dt, RHS_function, RHS_function(u)*dt, [1], c, Gamma, Leja_X, phi_1, tol)

    if convergence == 0:
        print("Error! Step size too large!!")

    ### 2nd order solution; u_2 = u + phi_1(J(u) dt) f(u) dt
    u_roseu = u + u_flux[:, 0]

    ### Proxy of computational cost
    num_rhs_calls = rhs_calls_1 + 1

    return u_roseu, num_rhs_calls