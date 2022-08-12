import numpy as np
from Divided_Difference import Divided_Difference

def imag_Leja_exp(u, dt, RHS_function, c, Gamma, Leja_X, tol):
    """
    Parameters
    ----------
    u                       : 1D vector u (input)
    dt                      : Step size
    RHS_function	        : RHS function
    c                       : Shifting factor
    Gamma                   : Scaling factor
    Leja_X                  : Array of Leja points
    tol                     : Accuracy of the polynomial so formed
    
    Returns
    ----------
    polynomial              : Polynomial interpolation of the matrix exponential multiplied
                              to 'u' at imaginary Leja points
    ii                      : # of RHS calls

    """

    ### Matrix exponential (scaled and shifted); c & Gamma (largest eigenvalue) scaled by dt
    matrix_exponential = np.exp(dt * (c + Gamma*Leja_X) * 1j)

    ### Compute the polynomial coefficients
    coeffs = Divided_Difference(Leja_X, matrix_exponential) 

    ### ------------------------------------------------------------------- ###

    ### Form the polynomial

    ## p_0 term
    polynomial = coeffs[0] * u + 0*1j

    ### p_1, p_2, ...., p_n terms
    max_Leja_pts = 100                                      # Max # of Leja points    
    y = u.copy() + 0*1j                                     # To avoid changing input vector 'u'

    ### Iterate until convergence is reached
    for ii in range(1, max_Leja_pts):
        
        ### Compute numerical Jacobian (for linear eqs., this is the RHS evaluation at y)
        Jacobian_function = RHS_function(y)

        ### Re-scale and re-shift
        y = y * (-c/Gamma - Leja_X[ii - 1])
        y = y + (Jacobian_function/Gamma) * (-1j)

        ### Error estimate
        poly_error = np.mean(abs(y)) * abs(coeffs[ii])
        
        ### Add the new term to the polynomial
        polynomial = polynomial + (coeffs[ii] * y)

        ### If new term to be added < tol, break loop; safety factor = 0.7
        if  poly_error < 0.7*tol:
            break

        if ii == max_Leja_pts - 1:
            print("Warning!! Max. # of Leja points reached!!")
            break

    return np.real(polynomial), ii
