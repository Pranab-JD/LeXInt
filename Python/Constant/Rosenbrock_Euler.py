import numpy as np

###! LeXInt functions
from Jacobian import Jacobian
from linear_phi import linear_phi

################################################################################################

def Rosenbrock_Euler(u, T_final, substeps, RHS_function, c, Gamma, Leja_X, tol, Real_Imag):
    """
    Parameters
    ----------
    u                       : numpy array
                                State variable(s)
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
    u_roseu                 : numpy array
                                Output state variable(s) after time dt (2nd order)
    num_rhs_calls           : int
                                # of RHS calls
    
    Reference:
    
        D. A. Pope 
        An exponential method of numerical integration of ordinary differential equations, Commun. ACM 6 (8) (1963) 491-493.
        doi:10.1145/366707.367592

    """
    
    ###? RHS evaluated at 'u'
    rhs_u = RHS_function(u)
    
    ###? Array of zeros vectors
    zero_vec = np.zeros(np.shape(u))
    
    ###? dt * J(u).z
    Jac_vec = lambda z: T_final * Jacobian(RHS_function, u, z, rhs_u)
    
    ###? Interpolation of RHS(u) at 1
    u_flux, rhs_calls, substeps = linear_phi([zero_vec, rhs_u*T_final], T_final, substeps, Jac_vec, 1, c, Gamma, Leja_X, tol)

    ###? 2nd order solution; u_roseu = u + phi_1(J(u) dt) f(u) dt
    u_roseu = u + u_flux

    ###? Proxy of computational cost
    num_rhs_calls = rhs_calls + 2

    return u_roseu, num_rhs_calls, substeps