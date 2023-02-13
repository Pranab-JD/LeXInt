import numpy as np
from Divided_Difference import Divided_Difference

def real_Leja_exp(u, dt, RHS_function, c, Gamma, Leja_X, tol):
    """
    Computes the polynomial interpolation of matrix exponential applied to 'u' at real Leja points.


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
    
        Returns
        ----------
        polynomial              : numpy array
                                    Polynomial interpolation of 'u' multiplied 
                                    by the matrix exponential at real Leja points
        ii+1                    : int
                                    # of RHS calls

    """
    
    ###? Initialize parameters and arrays
    max_Leja_pts = len(Leja_X)                                    #* Max number of Leja points  
    y = u.copy()                                                  #* To avoid changing 'interp_function'

    ###? Matrix exponential (scaled and shifted)
    matrix_exponential = np.exp(dt * (c + Gamma*Leja_X))

    ###? Compute polynomial coefficients
    poly_coeffs = Divided_Difference(Leja_X, matrix_exponential) 

    ###? Form the polynomial: p_0 term
    polynomial = poly_coeffs[0] * u

    ###? p_1, p_2, ...., p_n terms; iterate until converges
    for ii in range(1, max_Leja_pts):
        
        ###? y = y * ((z - c)/Gamma - Leja_X)
        y = (RHS_function(y)/Gamma) + (y * (-c/Gamma - Leja_X[ii - 1]))

        ###? Error estimate; poly_error = |coeffs[nn]| ||y||
        poly_error = np.linalg.norm(y) * abs(poly_coeffs[ii])
        
        ###? Add the new term to the polynomial
        polynomial = polynomial + (poly_coeffs[ii] * y)

        ###? If new term to be added < tol, break loop; safety factor = 0.1
        if  poly_error < 0.1*tol*np.linalg.norm(polynomial) + tol:
            break

        ###! Warning flags
        if ii == max_Leja_pts - 1:
            print("Warning!! Max. # of Leja points reached without convergence!!")
            print("Max. Leja points currently set to", max_Leja_pts)
            print("Try increasing the number of Leja points. Max available: 10000.\n")
            break

    return polynomial, ii