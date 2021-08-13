"""
Created on Fri Aug 13 15:31:46 2021

@author: Pranab JD

Description: -
        Contains embedded exponential Rosenbrock (EXPRB) schemes

"""

from Leja_Interpolation import *

################################################################################################

def EXPRB32(u, dt, RHS_func, c, Gamma, rel_tol, Real_Imag_Leja):
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
    u_exprb2        : 1D vector u (output) after time dt (2nd order)
    u_exprb3        : 1D vector u (output) after time dt (3rd order)
    num_rhs_calls   : # of RHS calls
    
    """

    ## Use either Real Leja or imaginary Leja
    if Real_Imag_Leja == 0:
        Leja_phi = real_Leja_phi
    elif Real_Imag_Leja == 1:
        Leja_phi = imag_Leja_phi
    else:
        print('Error in choosing Real/Imag Leja!!')

    epsilon = 1e-7
    
    ### RHS of PDE at u
    f_u = RHS_func(u)

    ############## --------------------- ##############

    ### Internal stage 1 
    a_n_f, rhs_calls_1 = Leja_phi(u, dt, RHS_func, f_u, c, Gamma, phi_1, rel_tol)
    a_n = u + (a_n_f * dt)

    ### 2nd order solution
    u_exprb2 = a_n

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

    ### 3rd order solution
    u_3, rhs_calls_2 = Leja_phi(u, dt, RHS_func, 2*(Nonlin_a - Nonlin_u), c, Gamma, phi_3, rel_tol)
    u_exprb3 = u_exprb2 + (u_3 * dt)

    ## Proxy of computational cost
    num_rhs_calls = rhs_calls_1 + rhs_calls_2 + 4

    return u_exprb2, u_exprb3, num_rhs_calls

################################################################################################

def EXPRB43(u, dt, RHS_func, c, Gamma, rel_tol, Real_Imag_Leja):
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
    u_exprb3        : 1D vector u (output) after time dt (3rd order)
    u_exprb4        : 1D vector u (output) after time dt (4th order)
    num_rhs_calls   : # of RHS calls
    
    """

    ## Use either Real Leja or imaginary Leja
    if Real_Imag_Leja == 0:
        Leja_phi = real_Leja_phi
    elif Real_Imag_Leja == 1:
        Leja_phi = imag_Leja_phi
    else:
        print('Error in choosing Real/Imag Leja!!')

    epsilon = 1e-7

    ### RHS of PDE at u
    f_u = RHS_func(u)

    ############## --------------------- ##############

    ### Internal stage 1
    a_n_f, rhs_calls_1 = Leja_phi(u, dt/2, RHS_func, f_u, c, Gamma, phi_1, rel_tol)
    a_n = u + (a_n_f * dt/2)

    ############## --------------------- ##############

    ### J(u) * u
    Linear_u = (RHS_func(u + (epsilon * u)) - f_u)/epsilon

    ### F(u) = f(u) - (J(u) * u)
    Nonlin_u = f_u - Linear_u

    ### J(u) * a
    Linear_a = (RHS_func(u + (epsilon * a_n)) - f_u)/epsilon

    ### F(a) = f(a) - (J(u) * a)
    Nonlin_a = RHS_func(a_n) - Linear_a

    ############# --------------------- ##############

    ### Internal stage 2
    b_n_f, rhs_calls_2 = Leja_phi(u, dt, RHS_func, f_u, c, Gamma, phi_1, rel_tol)
    b_n_nl, rhs_calls_3 = Leja_phi(u, dt, RHS_func, (Nonlin_a - Nonlin_u), c, Gamma, phi_1, rel_tol)

    b_n = u + (b_n_f * dt) + (b_n_nl * dt)

    ############# --------------------- ##############

    ### J(u) * b
    Linear_b = (RHS_func(u + (epsilon * b_n)) - f_u)/epsilon

    ### F(b) = f(b) - (J(u) * b)
    Nonlin_b = RHS_func(b_n) - Linear_b

    ############# --------------------- ##############

    ### 3rd and 4th order solutions
    u_flux = b_n_f
    u_nl_3, rhs_calls_4 = Leja_phi(u, dt, RHS_func, (-14*Nonlin_u + 16*Nonlin_a - 2*Nonlin_b), c, Gamma, phi_3, rel_tol)
    u_nl_4, rhs_calls_5 = Leja_phi(u, dt, RHS_func, (36*Nonlin_u - 48*Nonlin_a + 12*Nonlin_b), c, Gamma, phi_4, rel_tol)

    u_exprb3 = u + (u_flux * dt) + (u_nl_3 * dt)
    u_exprb4 = u_exprb3 + (u_nl_4 * dt)

    ## Proxy of computational cost
    num_rhs_calls = rhs_calls_1 + rhs_calls_2 + rhs_calls_3 + rhs_calls_4 + rhs_calls_5 + 6

    return u_exprb3, u_exprb4, num_rhs_calls

################################################################################################