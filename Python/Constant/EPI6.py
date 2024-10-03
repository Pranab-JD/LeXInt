import sys
sys.path.insert(1, "../")

import numpy as np

###! LeXInt functions
from Jacobian import Jacobian
from linear_phi import linear_phi

################################################################################################

def EPI6(u, u_prev, T_final, substeps, RHS_function, c, Gamma, Leja_X, tol, Real_Imag):
    """
    Parameters
    ----------
    u                       : numpy array
                                State variable(s) at the current time step (n)
    u_prev                  : numpy array
                                State variable(s) at the 2 previous time steps (n - 1, n - 2, n - 3, n - 4)
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
    u_epi6                  : numpy array
                                Output state variable(s) after time T_final (6th order)
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
    
    ###? EPI5 coefficients
    a21 = -49/60; a22 = 351/560; a23 = -359/1260; a24 = 367/6720
    a31 =   92/7; a32 =  -99/14; a33 =    176/63; a34 =     -1/2
    a41 = 485/21; a42 = -151/14; a43 =      23/9; a44 =  -31/168
    
    ###? Difference of nonlinear remainders at u^{n-1}, u^{n-2}, u^{n-3}, and u^{n-4}
    R_1 = (RHS_function(u_prev[:, 0]) - Jacobian(RHS_function, u, u_prev[:, 0], rhs_u)) - (rhs_u - Jacobian_u)
    R_2 = (RHS_function(u_prev[:, 1]) - Jacobian(RHS_function, u, u_prev[:, 1], rhs_u)) - (rhs_u - Jacobian_u)
    R_3 = (RHS_function(u_prev[:, 2]) - Jacobian(RHS_function, u, u_prev[:, 2], rhs_u)) - (rhs_u - Jacobian_u)
    R_4 = (RHS_function(u_prev[:, 3]) - Jacobian(RHS_function, u, u_prev[:, 3], rhs_u)) - (rhs_u - Jacobian_u)
    
    ###? Interpolation 1; phi_1(J(u) dt) f(u) dt + phi_2(J(u) dt) (a21 R(u^{n-1}) + a22 R(u^{n-2}) + a23 R(u^{n-3}) + a24 R(u^{n-4})) dt
    ###?                + phi_3(J(u) dt) (a31 R(u^{n-1}) + a32 R(u^{n-2}) + a33 R(u^{n-3}) + a34 R(u^{n-4})) dt + phi_4(J(u) dt) (a41 R(u^{n-1}) + a42 R(u^{n-2}) + a43 R(u^{n-3}) + a44 R(u^{n-4})) dt
    u_flux, rhs_calls, substeps = linear_phi([zero_vec, rhs_u*T_final, (a21*R_1+a22*R_2+a23*R_3+a24*R_4)*T_final, (a31*R_1+a32*R_2+a33*R_3+a34*R_4)*T_final,\
                                                                       (a41*R_1+a42*R_2+a43*R_3+a44*R_4)*T_final], T_final, substeps, Jac_vec, 1, c, Gamma, Leja_X, tol)

    ###? Internal stage; 6th order solution; u_6 = u + phi_1(J(u) dt) f(u) dt + phi_2(J(u) dt) (a21 R(u^{n-1}) + a22 R(u^{n-2}) + a32 R(u^{n-3})) dt
    ###?                                             + phi_3(J(u) dt) (a31 R(u^{n-1}) + a32 R(u^{n-2}) + a33 R(u^{n-3})) dt + phi_4(J(u) dt) (a41 R(u^{n-1}) + a42 R(u^{n-2}) + a43 R(u^{n-3})) dt
    u_epi6 = u + u_flux

    ###? Proxy of computational cost
    num_rhs_calls = rhs_calls + 10

    return u_epi6, num_rhs_calls, substeps