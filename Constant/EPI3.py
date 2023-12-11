import numpy as np

###! LeXInt functions
from Jacobian import Jacobian
from linear_phi import linear_phi

################################################################################################

def EPI3(u, u_prev, T_final, RHS_function, c, Gamma, Leja_X, tol, Real_Imag):
    """
    Parameters
    ----------
    u                       : numpy array
                                State variable(s) at the current time step (n)
    u_prev                  : numpy array
                                State variable(s) at the previous time step (n - 1)
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
    u_epi3                  : numpy array
                                Output state variable(s) after time T_final (3rd order)
    num_rhs_calls           : int
                                # of RHS calls
    
    Reference:
    
        M. Tokman
        Eﬃcient integration of large stiff systems of ODEs with exponential propagation iterative (EPI) methods, J. Comput. Phys. 213 (2) (2006) 748-776
        doi:10.1016/j.jcp.2005.08.032

    """
    
    ###? RHS evaluated at 'u'
    rhs_u = RHS_function(u)

    ###? Array of zeros vectors
    zero_vec = np.zeros(np.shape(u))
    
    ###? dt * J(u).z
    Jac_vec = lambda z: T_final * Jacobian(RHS_function, u, z, rhs_u)
    
    ###? J(u) . u
    Jacobian_u = Jacobian(RHS_function, u, u, rhs_u)
    
    ###? Difference of nonlinear remainders at u^{n-1}
    R_1 = (RHS_function(u_prev) - Jacobian(RHS_function, u, u_prev, rhs_u)) - (rhs_u - Jacobian_u)
    
    ###? Interpolation 1; phi_1(J(u) dt) f(u) dt + 2/3 phi_2(J(u) dt) R(u^{n-1}) dt
    u_flux, rhs_calls = linear_phi([zero_vec, rhs_u*T_final, 2/3*R_1*T_final], T_final, Jac_vec, 1, c, Gamma, Leja_X, tol)

    ###? Internal stage; 3rd order solution; u_3 = u + phi_1(J(u) dt) f(u) dt + 2/3 phi_2(J(u) dt) R(u^{n-1}) dt
    u_epi3 = u + u_flux

    ###? Proxy of computational cost
    num_rhs_calls = rhs_calls + 4

    return u_epi3, num_rhs_calls