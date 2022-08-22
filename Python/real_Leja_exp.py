import numpy as np
from Divided_Difference import Divided_Difference

def real_Leja_exp(u, dt, RHS_function, c, Gamma, Leja_X, tol):
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
                              by 'u' at real Leja points
    rhs_calls               : # of RHS calls

    """
    
    ### Max # of Leja points
    max_Leja_pts = len(Leja_X)  
    
    ### Matrix exponential (scaled and shifted); scaling down of c and Gamma (i.e. largest and smallest eigenvalue) by dt 
    matrix_exponential = np.exp((c + Gamma*Leja_X) * dt)
    
    ### Compute the polynomial coefficients
    coeffs = Divided_Difference(Leja_X, matrix_exponential) 
    
    ### ------------------------------------------------------------------- ###
    
     ### Form the polynomial, p_0 term
    y = u.copy()
    polynomial = coeffs[0] * u      

    ### p_1, p_2, ...., p_n terms; iterate until polynomial converges
    for ii in range(1, max_Leja_pts):

        ### y = y * ((z - c)/Gamma - Leja_X)
        y = (RHS_function(y)/Gamma) + (y * (-c/Gamma - Leja_X[ii - 1]))

        ### Error estimate
        poly_error = np.norm(y) * abs(coeffs[ii])

        ### Add new term to the polynomial
        polynomial = polynomial + (coeffs[ii] * y)

        ### If new term to be added < tol, break loop; safety factor = 0.1
        if  poly_error < 0.1*tol:
            print("# Leja points: ", ii)
            break

        if ii == max_Leja_pts - 1:
            print("Warning!! Max. # of Leja points reached!!")
            break

    return polynomial, rhs_calls
