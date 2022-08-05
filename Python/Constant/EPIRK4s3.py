import sys
sys.path.insert(1, "../")

from Phi_functions import *
from real_Leja_phi_constant import *
from imag_Leja_phi_constant import *

################################################################################################

def EPIRK4s3(u, dt, RHS_function, c, Gamma, Leja_X, tol, Real_Imag):
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
    u_epirk4        : 1D vector u (output) after time dt (4th order)
    num_rhs_calls   : # of RHS calls

    """

    ### Interpolate on either real Leja or imaginary Leja points
    if Real_Imag == 0:
        Leja_phi = real_Leja_phi
    elif Real_Imag == 1:
        Leja_phi = imag_Leja_phi
    else:
        print("Error!! Choose 0 for real or 1 for imaginary Leja points.")
    
    ### Function to compute the nonlinear remainder at stage 'y'
    def Nonlinear_remainder(y):
        
        epsilon = 1e-4
        
        ### J(u) * y
        Linear_y = (RHS_function(u + (epsilon * y)) - RHS_function(u - (epsilon * y)))/(2*epsilon)

        ### F(y) = f(y) - (J(u) * y)
        Nonlinear_y = RHS_function(y) - Linear_y
        
        return Nonlinear_y
    
    ############## --------------------- ##############

    ### Vertical interpolation of RHS_function(u) at 1/8, 1/9, and 1
    u_flux, rhs_calls_1 = Leja_phi(u, dt, RHS_function, RHS_function(u)*dt, [1/8, 1/9, 1], c, Gamma, Leja_X, phi_1, tol)

    ### Internal stage 1; a = u + 1/8 phi_1(1/8 J(u) dt) f(u) dt
    a = u + (1/8 * u_flux[:, 0])
    
    ### Internal stage 2; b = u + 1/9 phi_1(1/9 J(u) dt) f(u) dt
    b = u + (1/9 * u_flux[:, 1])

    ############# --------------------- ##############

    ### Nonlinear remainder at u, a, and b
    Nonlinear_u = Nonlinear_remainder(u)
    Nonlinear_a = Nonlinear_remainder(a)
    Nonlinear_b = Nonlinear_remainder(b)
    
    R_a = Nonlinear_a - Nonlinear_u
    R_b = Nonlinear_b - Nonlinear_u

    ############# --------------------- ##############
    
    ### Final nonlinear stages
    u_nl_3, rhs_calls_2 = Leja_phi(u, dt, RHS_function, (1892*R_a + 1458*(R_b - 2*R_a))*dt,    [1], c, Gamma, Leja_X, phi_3, tol)
    u_nl_4, rhs_calls_3 = Leja_phi(u, dt, RHS_function, (-42336*R_a - 34992*(R_b - 2*R_a))*dt, [1], c, Gamma, Leja_X, phi_4, tol)

    ### 4th order solution; u_4 = u + phi_1(J(u) dt) f(u) dt + phi_3(J(u) dt) (-1024R(a) + 1458R(b)) dt + phi_4(J(u) dt) (27648R(a) - 34992R(b)) dt
    u_epirk4 = u + u_flux[:, 2] + u_nl_3[:, 0] + u_nl_4[:, 0]

    ### Proxy of computational cost
    num_rhs_calls = rhs_calls_1 + rhs_calls_2 + rhs_calls_3 + 9

    return u_epirk4, num_rhs_calls