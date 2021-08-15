import sys

sys.path.insert(1, "../../Adaptive/")

from Leja_Interpolation import *

################################################################################################

def EXPRB54s4(u, dt, RHS_func, c, Gamma, rel_tol, Real_Imag_Leja):
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
    u_exprb4        : 1D vector u (output) after time dt (4th order)
    u_exprb5        : 1D vector u (output) after time dt (5th order)
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
    
    ### Check if iterations converge for largest value of "f_u" and "dt".
    ### If this stage doesn't converge, we have to go for a smaller 
    ### step size. "u_flux" is used in the final stage.
    
    u_flux, rhs_calls_1 = Leja_phi(u, dt, RHS_func, f_u, c, Gamma, phi_1, rel_tol)
    
    ############## --------------------- ##############

    ### Internal stage 1
    a_n_f, rhs_calls_2 = Leja_phi(u, dt/4, RHS_func, f_u, c, Gamma, phi_1, rel_tol)
    a_n = u + (a_n_f * dt/4)

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
    b_n_f,  rhs_calls_3 = Leja_phi(u, dt/2, RHS_func, f_u, c, Gamma, phi_1, rel_tol)
    b_n_nl, rhs_calls_4 = Leja_phi(u, dt/2, RHS_func, (Nonlin_a - Nonlin_u), c, Gamma, phi_3, rel_tol)

    b_n = u + (1/2 * b_n_f * dt) + (4 * b_n_nl * dt)

    ############# --------------------- ##############

    ### J(u) * b
    Linear_b = (RHS_func(u + (epsilon * b_n)) - f_u)/epsilon

    ### F(b) = f(b) - (J(u) * b)
    Nonlin_b = RHS_func(b_n) - Linear_b

    ############# --------------------- ##############
    
    ### Internal stage 3
    c_n_f,  rhs_calls_5 = Leja_phi(u, 9*dt/10, RHS_func, f_u, c, Gamma, phi_1, rel_tol)
    c_n_nl, rhs_calls_6 = Leja_phi(u, 9*dt/10, RHS_func, (Nonlin_b - Nonlin_u), c, Gamma, phi_3, rel_tol)

    c_n = u + (1/2 * c_n_f * dt) + (4 * c_n_nl * dt)

    ############# --------------------- ##############

    ### J(u) * c
    Linear_c = (RHS_func(u + (epsilon * c_n)) - f_u)/epsilon

    ### F(c) = f(c) - (J(u) * c)
    Nonlin_c = RHS_func(c_n) - Linear_c

    ############# --------------------- ##############
    
    ### Interpolation of nonlinear remainders for final solutions
    u_nl_4_3, rhs_calls_7  = Leja_phi(u, dt, RHS_func, (-56*Nonlin_u + 64*Nonlin_a - 8*Nonlin_b), c, Gamma, phi_3, rel_tol)
    u_nl_4_4, rhs_calls_8  = Leja_phi(u, dt, RHS_func, (80*Nonlin_u - 60*Nonlin_a - (285/8)*Nonlin_b + (125/8)*Nonlin_c), c, Gamma, phi_4, rel_tol)
    u_nl_5_3, rhs_calls_9  = Leja_phi(u, dt, RHS_func, (-(1208/81)*Nonlin_u + 18*Nonlin_b - (250/81)*Nonlin_c), c, Gamma, phi_3, rel_tol)
    u_nl_5_4, rhs_calls_10 = Leja_phi(u, dt, RHS_func, ((1120/27)*Nonlin_u - 60*Nonlin_b + (500/27)*Nonlin_c), c, Gamma, phi_4, rel_tol)

    ### 4th and 5th order solutions
    u_exprb4 = u + (u_flux * dt) + (u_nl_4_3 * dt) + (u_nl_4_4 * dt)
    u_exprb5 = u + (u_flux * dt) + (u_nl_5_3 * dt) + (u_nl_5_4 * dt)

    ## Proxy of computational cost
    num_rhs_calls = rhs_calls_1 + rhs_calls_2 + rhs_calls_3 + rhs_calls_4 + rhs_calls_5 + \
                    rhs_calls_6 + rhs_calls_7 + rhs_calls_8 + rhs_calls_9 + rhs_calls_10 + 8

    return u_exprb4, u_exprb5, num_rhs_calls