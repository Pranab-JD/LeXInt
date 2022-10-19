import sys
sys.path.insert(1, "../")

from Phi_functions import *
from real_Leja_phi import *
from imag_Leja_phi import *

################################################################################################

def EPIRK4s3B(u, dt, RHS_function, c, Gamma, Leja_X, tol, Real_Imag):
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
    
    Reference: 
        G. Rainwater, M. Tokman, A new approach to constructing efficient stiffly accurate EPIRK methods, J. Comput. Phys. 323 (2016) 283-309.
        doi:10.1016/j.jcp.2016.07.026.

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
        
        epsilon = 1e-7
        
        ### J(u) * y
        Linear_y = (RHS_function(u + (epsilon * y)) - RHS_function(u - (epsilon * y)))/(2*epsilon)

        ### F(y) = f(y) - (J(u) * y)
        Nonlinear_y = RHS_function(y) - Linear_y
        
        return Nonlinear_y
    
    ############## --------------------- ##############

	### Vertical interpolation of RHS_function(u) at 1/2 and 3/4
    u_flux, rhs_calls_1, convergence = Leja_phi(u, dt, RHS_function, RHS_function(u)*dt, [1/2, 3/4], c, Gamma, Leja_X, phi_2, tol)

    if convergence == 0:
        print("Error! Step size too large!!")

    ### Internal stage 1; a = u + 2/3 phi_2(1/2 J(u) dt) f(u) dt
    a = u + (2/3 * u_flux[:, 0])
    
    ### Internal stage 2; b = u + phi_2(3/4 J(u) dt) f(u) dt
    b = u + u_flux[:, 1]

    ############# --------------------- ##############

    ### Nonlinear remainder at u, a, and b
    Nonlinear_u = Nonlinear_remainder(u)
    Nonlinear_a = Nonlinear_remainder(a)
    Nonlinear_b = Nonlinear_remainder(b)
    
    R_a = Nonlinear_a - Nonlinear_u
    R_b = Nonlinear_b - Nonlinear_u

    ############# --------------------- ##############
    
    ### Final stages
    u_1, rhs_calls_2, convergence    = Leja_phi(u, dt, RHS_function, f_u*dt, [1], c, Gamma, Leja_X, phi_1, tol)
    u_nl_3, rhs_calls_3, convergence = Leja_phi(u, dt, RHS_function, (54*R_a - 16*R_b)*dt, [1], c, Gamma, Leja_X, phi_3, tol)
    u_nl_4, rhs_calls_4, convergence = Leja_phi(u, dt, RHS_function, (-324*R_a + 144*R_b)*dt, [1], c, Gamma, Leja_X, phi_4, tol)
 
    ### 4th order solution; u_4 = u + phi_1(J(u) dt) f(u) dt + phi_3(J(u) dt) (54R(a) - 16R(b)) dt + phi_4(J(u) dt) (-324R(a) + 144R(b)) dt
    u_epirk4 = u + u_1[:, 0] + u_nl_3[:, 0] + u_nl_4[:, 0]

    ### Proxy of computational cost
    num_rhs_calls = rhs_calls_1 + rhs_calls_2 + rhs_calls_3 + rhs_calls_4 + 9

    return u_epirk4, num_rhs_calls