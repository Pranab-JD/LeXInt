import sys
sys.path.insert(1, "../")

from Phi_functions import *
from real_Leja_phi import *
from imag_Leja_phi import *

################################################################################################

def EXPRB54s4(u, dt, RHS_function, c, Gamma, Leja_X, tol, Real_Imag):
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
    u_exprb4        : 1D vector u (output) after time dt (4th order)
    u_exprb5        : 1D vector u (output) after time dt (5th order)
    num_rhs_calls   : # of RHS calls
    
    Reference:
        V. T. Luan, A. Ostermann, Exponential rosenbrock methods of order five — construction, analysis and numerical comparisons, J. Comput. Appl. Math. 255 (2014) 417-431. 
        doi:10.1016/j.cam.2013.04.041.

    """

    ### Interpolate on either real Leja or imaginary Leja points.
    if Real_Imag == 0:
        Leja_phi = real_Leja_phi
    elif Real_Imag == 1:
        Leja_phi = imag_Leja_phi
    else:
        print("Error!! Choose 0 for real or 1 for imaginary Leja points.")
    
    ### Function to compute the nonlinear remainder at stage 'y'
    def Nonlinear_remainder(y):
        
        epsilon = 1e-5
        
        ### J(u) * y
        Linear_y = (RHS_function(u + (epsilon * y)) - RHS_function(u - (epsilon * y)))/(2*epsilon)

        ### F(y) = f(y) - (J(u) * y)
        Nonlinear_y = RHS_function(y) - Linear_y
        
        return Nonlinear_y
    
    ############## --------------------- ##############
    
    ### Vertical interpolation of f_u at 1/4, 1/2, 9/10, and 1
    u_flux, rhs_calls_1, convergence = Leja_phi(u, dt, RHS_function, RHS_function(u)*dt, [1/4, 1/2, 9/10, 1], c, Gamma, Leja_X, phi_1, tol)

    ### If it does not converge, return (try with smaller dt)
    if convergence == 0:
        return u, 2.1*u, rhs_calls_1

    ### Internal stage 1; a = u + (1/4) phi_1(1/4 J(u) dt) f(u) dt
    a_n = u + (1/4 * u_flux[:, 0])
    
    ############## --------------------- ##############
    
    ### Nonlinear remainder at u
    Nonlinear_u = Nonlinear_remainder(u)

    ### Nonlinear remainder at a
    Nonlinear_a = Nonlinear_remainder(a_n)
    
    R_a = Nonlinear_a - Nonlinear_u

    ############# --------------------- ##############

    ### Internal stage 2; interpolation of (N(a) - N(u)) at 1
    b_n_nl, rhs_calls_2, convergence = Leja_phi(u, dt, RHS_function, R_a*dt, [1/2], c, Gamma, Leja_X, phi_3, tol)

    ### b = u + (1/2) phi_1(1/2 J(u) dt) f(u) dt + 4 phi_3(1/2 J(u) dt) R(a) dt
    b_n = u + (1/2 * u_flux[:, 1]) + (4 * b_n_nl[:, 0])
    
    ### Nonlinear remainder at b
    Nonlinear_b = Nonlinear_remainder(b_n)
    
    R_b = Nonlinear_b - Nonlinear_u

    ############# --------------------- ##############
    
    ### Internal stage 3; interpolation of (N(b) - N(u)) at 1
    c_n_nl, rhs_calls_3, convergence = Leja_phi(u, dt, RHS_function, R_b*dt, [9/10], c, Gamma, Leja_X, phi_3, tol)

    ### c = u + (9/10) phi_1(9/10 J(u) dt) f(u) dt + (729/125) phi_3(9/10 J(u) dt) R(b) dt
    c_n = u + (9/10 * u_flux[:, 2]) + (729/125 * c_n_nl[:, 0])
    
    ### Nonlinear remainder at c
    Nonlinear_c = Nonlinear_remainder(c_n)
    
    R_c = Nonlinear_c - Nonlinear_u

    ############# --------------------- ##############
    
    ### Final nonlinear stages
    u_nl_4_3, rhs_calls_4, convergence = Leja_phi(u, dt, RHS_function, (64*R_a - 8*R_b)*dt, [1], c, Gamma, Leja_X, phi_3, tol)
    u_nl_4_4, rhs_calls_5, convergence = Leja_phi(u, dt, RHS_function, (-60*R_a - (285/8)*R_b + (125/8)*R_c)*dt, [1], c, Gamma, Leja_X, phi_4, tol)
    u_nl_5_3, rhs_calls_6, convergence = Leja_phi(u, dt, RHS_function, (18*R_b - (250/81)*R_c)*dt, [1], c, Gamma, Leja_X, phi_3, tol)
    u_nl_5_4, rhs_calls_7, convergence = Leja_phi(u, dt, RHS_function, (-60*R_b + (500/27)*R_c)*dt, [1], c, Gamma, Leja_X, phi_4, tol)

    ### 4th order solution; u_4 = u + phi_1(J(u) dt) f(u) dt + phi_3(J(u) dt) (64R(a) - 8R(b)) dt + phi_4(J(u) dt) (-60R(a) - (285/8)R(b) + (125/8)R(c)) dt
    u_exprb4 = u + u_flux[:, 3] + u_nl_4_3[:, 0] + u_nl_4_4[:, 0]
    
    ### 5th order solution; u_5 = u + phi_1(J(u) dt) f(u) dt + phi_3(J(u) dt) (18R(a) - (250/81)R(b)) dt + phi_4(J(u) dt) (-60R(a) + (500/27)R(b)) dt
    u_exprb5 = u + u_flux[:, 3] + u_nl_5_3[:, 0] + u_nl_5_4[:, 0]

    ### Proxy of computational cost
    num_rhs_calls = rhs_calls_1 + rhs_calls_2 + rhs_calls_3 + rhs_calls_4 + rhs_calls_5 + rhs_calls_6 + rhs_calls_7 + 12

    return u_exprb4, u_exprb5, num_rhs_calls