import sys

sys.path.insert(1, "../../Constant/")

from Leja_Interpolation import *

################################################################################################

def Rosenbrock_Euler(u, dt, RHS_func, c, Gamma, rel_tol, Real_Imag_Leja):
    """
    Parameters
    ----------
	u               : 1D vector u (input)
	dt              : Step size
	RHS_func	    : RHS function
    c               : Shifting factor
    Gamma           : Scaling factor
    rel_tol         : Accuracy of the polynomial so formed
    Real_Imag_Leja  : 0 - Real, 1 - Imag

    Returns
    -------
    u_etd           : 1D vector u (output) after time dt (1st/2nd order)
    num_rhs_calls   : # of RHS calls
    
    """

    ### Interpolate on either real Leja or imaginary Leja points.
    if Real_Imag_Leja == 0:
        Leja_phi = real_Leja_phi
    elif Real_Imag_Leja == 1:
        Leja_phi = imag_Leja_phi
    else:
        print("Error!! Choose 0 for real or 1 for imag Leja points.")
    
    ### RHS of PDE at u
    f_u = RHS_func(u)
    
    ### Solution
    u_flux, num_rhs_calls = Leja_phi(u, dt, RHS_func, f_u, c, Gamma, phi_1, rel_tol)
    u_roseu = u + (u_flux * dt)

    return u_roseu, num_rhs_calls