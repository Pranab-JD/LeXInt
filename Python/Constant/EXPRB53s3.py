import sys
sys.path.insert(1, "../")

from real_Leja_phi import *
from Phi_functions import *

################################################################################################

def EXPRB53s3(u, dt, RHS_function, c, Gamma, Leja_X, tol, Real_Imag):
    """
    Parameters
    ----------
    u               : 1D vector u (input)
    dt              : Step size
    RHS_function	: RHS function
    c               : Shifting factor
    Gamma           : Scaling factor
    Leja_X          : Array of Leja points
    tol             : Accuracy of the polynomial so formed
    Real_Imag       : 0 - Real, 1 - Imaginary

    Returns
    -------
    u_exprb3        : 1D vector u (output) after time dt (3rd order)
    u_exprb5        : 1D vector u (output) after time dt (5th order)
    num_rhs_calls   : # of RHS calls

    """

    ### Interpolate on either real Leja or imaginary Leja points.
    if Real_Imag == 0:
        Leja_phi = real_Leja_phi
    elif Real_Imag == 1:
        Leja_phi = imag_Leja_phi
    else:
        print("Error!! Choose 0 for real or 1 for imaginary Leja points.")
    
    ### RHS of PDE at u
    f_u = RHS_function(u)
    
    ### Function to compute the nonlinear remainder at stage 'y'
    def Nonlinear_remainder(y):
        
        epsilon = 1e-7
        
        ### J(u) * y
        Linear_y = (RHS_function(u + (epsilon * y)) - f_u)/epsilon

        ### F(y) = f(y) - (J(u) * y)
        Nonlinear_y = RHS_function(y) - Linear_y
        
        return Nonlinear_y
    
    ############## --------------------- ##############
    
    ### Vertical interpolation of f_u at 1/2, 9/10, and 1
    u_flux, rhs_calls_1, convergence = Leja_phi(u, dt, RHS_function, f_u*dt, [1/2, 9/10, 1], c, Gamma, Leja_X, phi_1, tol)

    ### If it does not converge, return (try with smaller dt)
    if convergence == 0:
        return u, 2.1*u, rhs_calls_1

    ### Internal stage 1; a = u + 1/2 phi_1(1/2 J(u) dt) f(u) dt
    a = u + (1/2 * u_flux[:, 0])

    ############## --------------------- ##############
    
    ### Nonlinear remainder at u
    Nonlinear_u = Nonlinear_remainder(u)

    ### Nonlinear remainder at a
    Nonlinear_a = Nonlinear_remainder(a)

    ############# --------------------- ##############

    ### Vertical interpolation of (N(a) - N(u)) at 1/2 and 9/10
    b_n_nl, rhs_calls_2, convergence = Leja_phi(u, dt, RHS_function, (Nonlinear_a - Nonlinear_u)*dt, [1/2, 9/10], c, Gamma, Leja_X, phi_3, tol)

    ### b = u + 9/10 phi_1(9/10 J(u) dt) f(u) dt + (27/25 phi_3(1/2 J(u) dt) + 729/125 phi_3(9/10 J(u) dt)) (N(a) - N(u)) dt
    b = u + (9/10 * u_flux[:, 1]) + (27/25 * b_n_nl[:, 0]) + (729/125 * b_n_nl[:, 1])
    
    ### Nonlinear remainder at b
    Nonlinear_b = Nonlinear_remainder(b)

    ############# --------------------- ##############
    
    ### Final nonlinear stages
    u_nl_4_3, rhs_calls_3, convergence = Leja_phi(u, dt, RHS_function, (-(312/81)*Nonlinear_u + 2*Nonlinear_a + (150/81)*Nonlinear_b)*dt,   [1], c, Gamma, Leja_X, phi_3, tol)
    u_nl_5_3, rhs_calls_4, convergence = Leja_phi(u, dt, RHS_function, (-(1208/81)*Nonlinear_u + 18*Nonlinear_a - (250/81)*Nonlinear_b)*dt, [1], c, Gamma, Leja_X, phi_3, tol)
    u_nl_5_4, rhs_calls_5, convergence = Leja_phi(u, dt, RHS_function, ((1120/27)*Nonlinear_u - 60*Nonlinear_a + (500/27)*Nonlinear_b)*dt,  [1], c, Gamma, Leja_X, phi_4, tol)

    ### 3rd order solution; u_3 = u + phi_1(J(u) dt) f(u) dt + phi_3(J(u) dt) (-(312/81)N(u) + 2N(a) + (150/81)N(b)) dt
    u_exprb4 = u + u_flux[:, 2] + u_nl_4_3[:, 0]
    
    ### 5th order solution; u_5 = u + phi_1(J(u) dt) f(u) dt + phi_3(J(u) dt) (-(1208/81)N(u) + 18N(a) - (250/81)N(b)) dt + phi_4(J(u) dt) (-(1120/27)N(u) - 60N(a) + (500/27)N(b)) dt
    u_exprb5 = u + u_flux[:, 2] + u_nl_5_3[:, 0] + u_nl_5_4[:, 0]

    ### Proxy of computational cost
    num_rhs_calls = rhs_calls_1 + rhs_calls_2 + rhs_calls_3 + rhs_calls_4 + rhs_calls_5 + 7

    return u_exprb4, u_exprb5, num_rhs_calls