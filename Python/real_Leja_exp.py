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
    
    s = 1                       # Number of substeps
    rhs_calls = 0               # Counter for # of Leja points used/RHS calls
    max_Leja_pts = len(Leja_X)  # Max # of Leja points
    
    ### Matrix exponential (scaled and shifted); scaling down of c and Gamma (i.e. largest and smallest eigenvalue) by dt 
    matrix_exponential = np.exp((c + Gamma*Leja_X) * dt/s)
    
    ### Compute the polynomial coefficients
    coeffs = Divided_Difference(Leja_X, matrix_exponential) 
    
    ### ------------------------------------------------------------------- ###
    
    ### Form the polynomial
    for substeps in range(0, s):                        
                
        ### p_0 term
        if substeps > 0:
            y = polynomial.copy()
            polynomial = coeffs[0] * polynomial 
        else:
            y = u.copy()
            polynomial = coeffs[0] * u      

        ### p_1, p_2, ...., p_n terms; iterate until polynomial converges
        for ii in range(1, max_Leja_pts):

            ### y = y * ((z - c)/Gamma - Leja_X)
            y = (RHS_function(y)/Gamma) + (y * (-c/Gamma - Leja_X[ii - 1]))

            ### Error estimate
            poly_error = np.mean(abs(y)) * abs(coeffs[ii])
            
            ### Add new term to the polynomial
            polynomial = polynomial + (coeffs[ii] * y)

            ### If new term to be added < tol, break loop; safety factor = 0.1
            if  poly_error < 0.1*tol:
                print("# Leja points: ", ii)
                rhs_calls = rhs_calls + ii
                break

            if ii == max_Leja_pts - 1:
                print("Warning!! Max. # of Leja points reached!!")
                break

    return polynomial, rhs_calls
