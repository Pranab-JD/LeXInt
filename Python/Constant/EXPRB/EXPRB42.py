import sys

sys.path.insert(1, "../../Constant/")

from Leja_Interpolation import *

################################################################################################

def EXPRB42(u, dt, RHS_func, c, Gamma, rel_tol, Real_Imag_Leja):
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
    u_exprb42       : 1D vector u (output) after time dt (4th order)
    num_rhs_calls   : # of RHS calls
    
    """

    ### Interpolate on either real Leja or imaginary Leja points.
    if Real_Imag_Leja == 0:
        Leja_phi = real_Leja_phi
    elif Real_Imag_Leja == 1:
        Leja_phi = imag_Leja_phi
    else:
        print("Error!! Choose 0 for real or 1 for imag Leja points.")

    epsilon = 1e-7

    ### RHS of PDE at u
    f_u = RHS_func(u)

    ############## --------------------- ##############
    
    ### Internal stage 1
    a_n_f, rhs_calls_1 = Leja_phi(u, 3*dt/4, RHS_func, f_u, c, Gamma, phi_1, rel_tol)
    a_n = u + (a_n_f * 3*dt/4)

    ############## --------------------- ##############
    
    ### J(u) * u
    Linear_u = (RHS_func(u + (epsilon * u)) - f_u)/epsilon

    ### F(u) = f(u) - (J(u) * u)
    Nonlin_u = f_u - Linear_u

    ### J(u) * a
    Linear_a = (RHS_func(u + (epsilon * a_n)) - f_u)/epsilon

    ### F(a) = f(a) - (J(u) * a)
    Nonlin_a = RHS_func(a_n) - Linear_a

    ############## --------------------- ##############

    u_flux, rhs_calls_2 = Leja_phi(u, dt, RHS_func, f_u, c, Gamma, phi_1, rel_tol)
    u_nl_3, rhs_calls_3 = Leja_phi(u, dt, RHS_func, (Nonlin_a - Nonlin_u), c, Gamma, phi_3, rel_tol)

    ### 4th order solution
    u_exprb42 = u + (u_flux * dt) + (u_nl_3 * 32*dt/9)

    ## Proxy of computational cost
    num_rhs_calls = rhs_calls_1 + rhs_calls_2 + rhs_calls_3 + 4

    return u_exprb42, num_rhs_calls