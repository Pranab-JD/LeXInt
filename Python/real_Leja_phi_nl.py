import numpy as np
from Divided_Difference import Divided_Difference

def real_Leja_phi_nl(u, dt, RHS_function, interp_function, c, Gamma, Leja_X, phi_function, tol):
    """
    Computes the polynomial interpolation of 'phi_function' applied to 'interp_vector' at real Leja points.
    To be used when computation of Jacobian is not needed, i.e. "interp_function" is (or explicitly treated
    as) a nonlinear remainder. 


        Parameters
        ----------
        u                       : numpy array
                                    State variable(s)
        dt                      : double
                                    Step size
        RHS_function            : user-defined function 
                                    RHS function
        interp_vector           : numpy array
                                    Vector to be interpolated
        c                       : double
                                    Shifting factor
        Gamma                   : double
                                    Scaling factor
        Leja_X                  : numpy array
                                    Array of Leja points
        phi_function            : function
                                    phi function
        tol                     : double
                                    Accuracy of the polynomial so formed

        Returns
        ----------
        polynomial_array        : numpy array(s)
                                    Polynomial interpolation of 'interp_vector' 
                                    multiplied by 'phi_function' at real Leja points
        ii                      : int
                                    # of RHS calls
        convergence             : int
                                    0 -> did not converge, 1 -> converged

    """

    ### Initialize paramters and arrays
    convergence = 0                                                             # 0 -> did not converge, 1 -> converged
    max_Leja_pts = len(Leja_X)                                                  # Max number of Leja points  
    y = interp_function.copy()                                                  # To avoid changing 'interp_function'
        
    ### Phi function applied to 'interp_function' (scaled and shifted)
    phi_function_array = phi_function((c + Gamma*Leja_X)*dt)
    
    ### Compute polynomial coefficients
    poly_coeffs = Divided_Difference(Leja_X, phi_function_array) 
    
    ### p_0 term
    polynomial = interp_function * poly_coeffs[0]
    
    ### p_1, p_2, ...., p_n terms; iterate until converges
    for ii in range(1, max_Leja_pts):

        ### y = y * ((z - c)/Gamma - Leja_X)
        y = (RHS_function(y)/Gamma) + (y * (-c/Gamma - Leja_X[ii - 1]))

        ### Error estimate
        poly_error = np.linalg.norm(y) * abs(poly_coeffs[ii])
        
        ### To prevent diverging, restart simulations with smaller dt
        if poly_error > 1e17:
            convergence = 0
            print("Step size too large!! Did not converge.")
            return u, ii, convergence

        ### Add the new term to the polynomial
        polynomial = polynomial + (poly_coeffs[ii] * y)
        
        ### If new term to be added < tol, break loop; safety factor = 0.25
        if  poly_error < 0.25*tol*np.linalg.norm(polynomial):
            convergence = 1
            # print("# Leja points (phi): ", ii)
            break
        
        ### Warning flags
        if ii == max_Leja_pts - 1:
            print("Warning!! Max. # of Leja points reached without convergence!! Try increasing the number of Leja points. Max available: 10000.")
            break

    return polynomial, ii, convergence
