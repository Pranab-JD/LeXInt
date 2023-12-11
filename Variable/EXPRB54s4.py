import sys
sys.path.insert(1, "../")

from Jacobian import *
from Phi_functions import *
from real_Leja_phi import *
from imag_Leja_phi import *

################################################################################################

def EXPRB54s4(u, dt, RHS_function, c, Gamma, Leja_X, tol, Real_Imag):
    """
    Parameters
    ----------
    u                       : numpy array
                                State variable(s)
    dt                      : double
                                Step size
    RHS_function            : user-defined function 
                                RHS function
    c                       : double
                                Shifting factor
    Gamma                   : double
                                Scaling factor
    Leja_X                  : numpy array
                                Array of Leja points
    tol                     : double
                                Accuracy of the polynomial so formed
    Real_Imag               : int
                                0 - Real, 1 - Imaginary

    Returns
    -------
    u_exprb4                : numpy array
                                Output state variable(s) after time dt (4th order)
    u_exprb5                : numpy array
                                Output state variable(s) after time dt (5th order)
    num_rhs_calls           : int
                                # of RHS calls
    
    Reference:
    
        V. T. Luan, A. Ostermann, Exponential Rosenbrock methods of order five - construction, analysis and numerical comparisons, J. Comput. Appl. Math. 255 (2014) 417-431. 
        doi:10.1016/j.cam.2013.04.041.

    """

    ############## --------------------- ##############

    ###? Interpolate on either real Leja or imaginary Leja points
    if Real_Imag == 0:
        Leja_phi = real_Leja_phi
    elif Real_Imag == 1:
        Leja_phi = imag_Leja_phi
    else:
        print("Error!! Choose 0 for real or 1 for imaginary Leja points.")
    
    ############## --------------------- ##############
    
    ###? Vertical interpolation of f(u) at 1/4, 1/2, 9/10, and 1; phi_1({1/4, 1/2, 9/10, 1} J(u) dt) f(u) dt
    u_flux, rhs_calls_1, convergence = Leja_phi(u, dt, RHS_function, RHS_function(u)*dt, [1/4, 1/2, 9/10, 1], c, Gamma, Leja_X, phi_1, tol)

    ###? If it does not converge, return (try with smaller dt)
    if convergence == 0:
        return u, 2.1*u, rhs_calls_1

    ###? Internal stage 1; a = u + (1/4) phi_1(1/4 J(u) dt) f(u) dt
    a_n = u + (1/4 * u_flux[:, 0])
    
    ###? Nonlinear remainder at u and a
    Nonlinear_u = RHS_function(u) - Jacobian(RHS_function, u, u)
    Nonlinear_a = RHS_function(a_n) - Jacobian(RHS_function, u, a_n)
    R_a = Nonlinear_a - Nonlinear_u

    ###? phi_3(1/2 J(u) dt) R(a) dt
    b_n_nl, rhs_calls_2, _ = Leja_phi(u, dt, RHS_function, R_a*dt, [1/2], c, Gamma, Leja_X, phi_3, tol)

    ###? Internal stage 2; b = u + (1/2) phi_1(1/2 J(u) dt) f(u) dt + 4 phi_3(1/2 J(u) dt) R(a) dt
    b_n = u + (1/2 * u_flux[:, 1]) + (4 * b_n_nl[:, 0])
    
    ###? Nonlinear remainder at b
    Nonlinear_b = RHS_function(b_n) - Jacobian(RHS_function, u, b_n)
    R_b = Nonlinear_b - Nonlinear_u
    
    ###? phi_3(9/10 J(u) dt) R(b) dt
    c_n_nl, rhs_calls_3, _ = Leja_phi(u, dt, RHS_function, R_b*dt, [9/10], c, Gamma, Leja_X, phi_3, tol)

    ###? Internal stage 3; c = u + (9/10) phi_1(9/10 J(u) dt) f(u) dt + (729/125) phi_3(9/10 J(u) dt) R(b) dt
    c_n = u + (9/10 * u_flux[:, 2]) + (729/125 * c_n_nl[:, 0])
    
    ###? Nonlinear remainder at c
    Nonlinear_c = RHS_function(c_n) - Jacobian(RHS_function, u, c_n)
    R_b = Nonlinear_c - Nonlinear_u
    
    ###? phi_3(J(u) dt) (64R(a) - 8R(b)) dt
    u_nl_4_3, rhs_calls_4, _ = Leja_phi(u, dt, RHS_function, (64*R_a - 8*R_b)*dt, [1], c, Gamma, Leja_X, phi_3, tol)
    
    ###? phi_4(J(u) dt) (-60R(a) - (285/8)R(b) + (125/8)R(c)) dt
    u_nl_4_4, rhs_calls_5, _ = Leja_phi(u, dt, RHS_function, (-60*R_a - (285/8)*R_b + (125/8)*R_c)*dt, [1], c, Gamma, Leja_X, phi_4, tol)
    
    ###? phi_3(J(u) dt) (18R(a) - (250/81)R(b)) dt
    u_nl_5_3, rhs_calls_6, _ = Leja_phi(u, dt, RHS_function, (18*R_b - (250/81)*R_c)*dt, [1], c, Gamma, Leja_X, phi_3, tol)
    
    ###? phi_4(J(u) dt) (-60R(a) + (500/27)R(b)) dt
    u_nl_5_4, rhs_calls_7, _ = Leja_phi(u, dt, RHS_function, (-60*R_b + (500/27)*R_c)*dt, [1], c, Gamma, Leja_X, phi_4, tol)

    ###? 4th order solution; u_4 = u + phi_1(J(u) dt) f(u) dt + phi_3(J(u) dt) (64R(a) - 8R(b)) dt + phi_4(J(u) dt) (-60R(a) - (285/8)R(b) + (125/8)R(c)) dt
    u_exprb4 = u + u_flux[:, 3] + u_nl_4_3[:, 0] + u_nl_4_4[:, 0]
    
    ###? 5th order solution; u_5 = u + phi_1(J(u) dt) f(u) dt + phi_3(J(u) dt) (18R(a) - (250/81)R(b)) dt + phi_4(J(u) dt) (-60R(a) + (500/27)R(b)) dt
    u_exprb5 = u + u_flux[:, 3] + u_nl_5_3[:, 0] + u_nl_5_4[:, 0]

    ###? Proxy of computational cost
    num_rhs_calls = rhs_calls_1 + rhs_calls_2 + rhs_calls_3 + rhs_calls_4 + rhs_calls_5 + rhs_calls_6 + rhs_calls_7 + 17

    return u_exprb4, u_exprb5, num_rhs_calls