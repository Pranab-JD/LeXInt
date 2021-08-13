"""
Created on Fri Aug 13 15:31:46 2021

@author: Pranab JD

Description: -
        Contains embedded exponential propagation
        iterative schemes of Runge-Kutta type (EPIRK)

"""

from Leja_Interpolation import *

################################################################################################

def EPIKR5P1(u, dt, RHS_func, c, Gamma, rel_tol, Real_Imag_Leja):
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
    u_epirk4        : 1D vector u (output) after time dt (4th order)
    u_epirk5        : 1D vector u (output) after time dt (5th order)
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
    
    ### Parameters of EPIRK5P1 (5th order)
    a11 = 0.35129592695058193092;
    a21 = 0.84405472011657126298;
    a22 = 1.6905891609568963624;

    b1  = 1.0;
    b2  = 1.2727127317356892397;
    b3  = 2.2714599265422622275;

    g11 = 0.35129592695058193092;
    g21 = 0.84405472011657126298;
    g22 = 0.5;
    g31 = 1.0;
    g32 = 0.71111095364366870359;
    g33 = 0.62378111953371494809;
    
    ### 4th order
    g32_4 = 0.5;
    g33_4 = 1.0;

    ### RHS of PDE at u
    f_u = RHS_func(u)

    ############## --------------------- ##############

    ### Internal stage 1
    a_n_f, rhs_calls_1 = Leja_phi(u, g11 * dt, RHS_func, f_u, c, Gamma, phi_1, rel_tol)
    a_n = u + (a_n_f * a11 * dt)

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
    b_n_f, rhs_calls_2  = Leja_phi(u, g21 * dt, RHS_func, f_u, c, Gamma, phi_1, rel_tol)
    b_n_nl, rhs_calls_3 = Leja_phi(u, g22 * dt, RHS_func, (Nonlin_a - Nonlin_u), c, Gamma, phi_1, rel_tol)

    b_n = u + (b_n_f * a21 * dt) + (b_n_nl * a22 * dt)

    ############# --------------------- ##############

    ### J(u) * b
    Linear_b = (RHS_func(u + (epsilon * b_n)) - f_u)/epsilon

    ### F(b) = f(b) - (J(u) * b)
    Nonlin_b = RHS_func(b_n) - Linear_b

    ############# --------------------- ##############
    
    ### Final stage
    u_flux, rhs_calls_4 = Leja_phi(u, g31 * dt, RHS_func, f_u, c, Gamma, phi_1, rel_tol)
    
    ### Nonlinear remainders for 5th order solution
    u_nl_1_5, rhs_calls_5 = Leja_phi(u, g32 * dt, RHS_func, (-Nonlin_u + Nonlin_a), c, Gamma, phi_1, rel_tol)
    u_nl_3_5, rhs_calls_6 = Leja_phi(u, g33 * dt, RHS_func, (Nonlin_u - 2*Nonlin_a + Nonlin_b), c, Gamma, phi_3, rel_tol)
    
    ### Nonlinear remainders for 5th order solution
    u_nl_1_4, rhs_calls_7 = Leja_phi(u, g32_4 * dt, RHS_func, (-Nonlin_u + Nonlin_a), c, Gamma, phi_1, rel_tol)
    u_nl_3_4, rhs_calls_8 = Leja_phi(u, g33_4 * dt, RHS_func, (Nonlin_u - 2*Nonlin_a + Nonlin_b), c, Gamma, phi_3, rel_tol)
 
    u_epirk4 = u + (u_flux * b1 * dt) + (u_nl_1_4 * b2 * dt) + (u_nl_3_4 * b3 * dt)
    u_epirk5 = u + (u_flux * b1 * dt) + (u_nl_1_5 * b2 * dt) + (u_nl_3_5 * b3 * dt)

    ## Proxy of computational cost
    num_rhs_calls = rhs_calls_1 + rhs_calls_2 + rhs_calls_3 + rhs_calls_4 + rhs_calls_5 + \
                    rhs_calls_6 + rhs_calls_7 + rhs_calls_8 + 6

    return u_epirk4, u_epirk5, num_rhs_calls

##############################################################################
