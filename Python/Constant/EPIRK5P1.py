import sys
sys.path.insert(1, "../")

from Jacobian import *
from Phi_functions import *
from real_Leja_phi import *
from imag_Leja_phi import *

################################################################################################

def EPIRK5P1(u, dt, RHS_function, c, Gamma, Leja_X, tol, Real_Imag):
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
    u_epirk4                : numpy array
                                Output state variable(s) after time dt (4th order)
    u_epirk5                : numpy array
                                Output state variable(s) after time dt (5th order)
    num_rhs_calls           : int
                                # of RHS calls
    
    Reference:
    
        M. Tokman, J. Loffeld, and P. Tranquilli
        New Adaptive Exponential Propagation Iterative Methods of Runge-Kutta Type, SIAM J. Sci. Comput. 34 (5) (2012) A2650-A2669
        doi:10.1137/110849961

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
    
    ###! Parameters of EPIRK5P1 (5th order)
    a11 = 0.35129592695058193092
    a21 = 0.84405472011657126298
    a22 = 1.6905891609568963624

    b1  = 1.0
    b2  = 1.2727127317356892397
    b3  = 2.2714599265422622275

    g11 = 0.35129592695058193092
    g21 = 0.84405472011657126298
    g22 = 1.0
    g31 = 1.0
    g32 = 0.71111095364366870359
    g33 = 0.62378111953371494809
    
    ###! 4th order
    g32_4 = 0.5
    g33_4 = 1.0
    
    ############## --------------------- ##############

    ###? Vertical interpolation of RHS_function(u) at g11, g21, and g31; phi_1({g11, g21, g31} J(u) dt) f(u) dt
    u_flux, rhs_calls_1, convergence = Leja_phi(u, dt, RHS_function, RHS_function(u)*dt, [g11, g21, g31], c, Gamma, Leja_X, phi_1, tol)
    
    ###? If it does not converge, return (try with smaller dt)
    if convergence == 0:
        return u, 2.1*u, rhs_calls_1

    ###? Internal stage 1; a = u + a11 phi_1(g11 J(u) dt) f(u) dt
    a = u + (a11 * u_flux[:, 0])
    
    ###? Nonlinear remainder at u and a
    Nonlinear_u = RHS_function(u) - Jacobian(RHS_function, u, u)
    Nonlinear_a = RHS_function(a) - Jacobian(RHS_function, u, a)
    R_a = Nonlinear_a - Nonlinear_u

    ###? Vertical interpolation of R_a at g32_4, g32, and g22; phi_1({g11, g21, g31} J(u) dt) R(a) dt
    u_nl_1, rhs_calls_2, _ = Leja_phi(u, dt, RHS_function, R_a*dt, [g32_4, g32, g22], c, Gamma, Leja_X, phi_1, tol)

    ###? b = u + a21 phi_1(g21 J(u) dt) f(u) dt + a22 phi_1(g22 J(u) dt) R_a dt
    b = u + (a21 * u_flux[:, 1]) + (a22 * u_nl_1[:, 2])

    ###? Nonlinear remainder at b
    Nonlinear_b = RHS_function(b) - Jacobian(RHS_function, u, b)
    R_b = Nonlinear_b - Nonlinear_u
    
    ###? Vertical interpolation of (-2*R(a) + R(b)) at g33 and g33_4; phi_3({g33, g33_4} J(u) dt) (-2*R(a) + R(b)) dt
    u_nl_2, rhs_calls_3, _ = Leja_phi(u, dt, RHS_function, (-2*R_a + R_b)*dt, [g33, g33_4], c, Gamma, Leja_X, phi_3, tol)
 
    ###? 4th order solution; u_4 = u + b1 phi_1(g31 J(u) dt) f(u) dt + b2 phi_1(g32 J(u) dt) R(a) dt + b3 phi_3(g33 J(u) dt) (-2*R(a) + R(b)) dt
    u_epirk4 = u + u_flux[:, 2] + (b2 * u_nl_1[:, 0]) + (b3 * u_nl_2[:, 1])
    
    ###? 5th order solution; u_5 = u + b1 phi_1(g31 J(u) dt) f(u) dt + b2 phi_1(g32_4 J(u) dt) R(a) dt + b3 phi_3(g33_4 J(u) dt) (-2*R(a) + R(b)) dt
    u_epirk5 = u + u_flux[:, 2] + (b2 * u_nl_1[:, 1]) + (b3 * u_nl_2[:, 0])

    ###? Proxy of computational cost
    num_rhs_calls = rhs_calls_1 + rhs_calls_2 + rhs_calls_3 + 13

    return u_epirk4, u_epirk5, num_rhs_calls