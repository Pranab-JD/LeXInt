import numpy as np

###! LeXInt functions
from Jacobian import Jacobian
from linear_phi import linear_phi

################################################################################################

def EPI4(u, u_prev, T_final, RHS_function, c, Gamma, Leja_X, tol, Real_Imag):
    """
    Parameters
    ----------
    u                       : numpy array
                                State variable(s) at the current time step (n)
    u_prev                  : numpy array
                                State variable(s) at the 2 previous time steps (n - 1, n - 2)
    T_final                 : double
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
    u_epi4                  : numpy array
                                Output state variable(s) after time T_final (4th order)
    num_rhs_calls           : int
                                # of RHS calls
    
    Reference:
    
        S. Gaudreault, M. Charron, V. Dallerit, and M. Tokman
        High-order numerical solutions to the shallow-water equations on the rotated cubed-sphere grid, J. Comput. Phys. 449 (2022) 110792. 
        doi:10.1016/j.jcp.2021.110792

    """
    
    ###? RHS evaluated at 'u'
    rhs_u = RHS_function(u)

    ###? Array of zeros vectors
    zero_vec = np.zeros(np.shape(u))
    
    ###? dt * J(u).z
    Jac_vec = lambda z: T_final * Jacobian(RHS_function, u, z, rhs_u)
    
    ###? J(u) . u
    Jacobian_u = Jacobian(RHS_function, u, u, rhs_u)
    
    ###? EPI4 coefficients
    a21 = -3/10; a22 = 3/40
    a31 = 32/5;  a32 = -11/10
    
    ###? Difference of nonlinear remainders at u^{n-1} and u^{n-2}
    R_1 = (RHS_function(u_prev[:, 0]) - Jacobian(RHS_function, u, u_prev[:, 0], rhs_u)) - (rhs_u - Jacobian_u)
    R_2 = (RHS_function(u_prev[:, 1]) - Jacobian(RHS_function, u, u_prev[:, 1], rhs_u)) - (rhs_u - Jacobian_u)
    
    ###? Interpolation 1; phi_1(J(u) dt) f(u) dt + phi_2(J(u) dt) (a21 R(u^{n-1}) + a22 R(u^{n-2})) dt + phi_3(J(u) dt) (a31 R(u^{n-1}) + a32 R(u^{n-2})) dt
    u_flux, rhs_calls = linear_phi([zero_vec, rhs_u*T_final, (a21*R_1+a22*R_2)*T_final, (a31*R_1+a32*R_2)*T_final], T_final, Jac_vec, 1, c, Gamma, Leja_X, tol)
    # u_flux, rhs_calls = linear_phi([zero_vec, rhs_u*T_final], T_final, Jac_vec, 1, c, Gamma, Leja_X, tol)

    ###? Internal stage; 4th order solution; u_3 = u + phi_1(J(u) dt) f(u) dt + phi_2(J(u) dt) (a21 R(u^{n-1}) + a22 R(u^{n-2})) dt + phi_3(J(u) dt) (a31 R(u^{n-1}) + a32 R(u^{n-2})) dt
    u_epi4 = u + u_flux

    ###? Proxy of computational cost
    num_rhs_calls = rhs_calls + 6

    return u_epi4, num_rhs_calls