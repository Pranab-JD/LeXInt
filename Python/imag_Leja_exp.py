import numpy as np
from Divided_Difference import Divided_Difference

def imag_Leja_exp(u, dt, RHS_function, c, Gamma, Leja_X, tol):
    """
    Computes the polynomial interpolation of matrix exponential applied to 'u' at imaginary Leja points.


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
                                    by the matrix exponential at imaginary Leja points
        ii+1                    : int
                                    # of RHS calls

    """

    ### Matrix exponential (scaled and shifted)
    matrix_exponential = np.exp(dt * (c + Gamma*Leja_X) * 1j)

    ### Compute polynomial coefficients
    coeffs = Divided_Difference(Leja_X, matrix_exponential) 

    ### Form the polynomial: p_0 term
    polynomial = coeffs[0] * u + 0*1j

    ### p_1, p_2, ...., p_n terms
    max_Leja_pts = len(Leja_X)                              # Max # of Leja points    
    y = u.copy() + 0*1j                                     # To avoid changing 'u'

    ### Iterate until convergence is reached
    for ii in range(1, max_Leja_pts):
        
        ### Compute numerical Jacobian (for linear eqs., this is the RHS evaluation at y)
        Jacobian_function = RHS_function(y)
        
        ### y = y * ((z - c)/Gamma - Leja_X)
        y = (-1j * Jacobian_function/Gamma) + (y * (-c/Gamma - Leja_X[ii - 1]))

        ### Error estimate
        poly_error = np.linalg.norm(y) * abs(coeffs[ii])
        
        ### Add the new term to the polynomial
        polynomial = polynomial + (coeffs[ii] * y)

        ### If new term to be added < tol, break loop; safety factor = 0.1
        if  poly_error < 0.1*tol:
            break

        ### Warning flags
        if ii == max_Leja_pts - 1:
            print("Warning!! Max. # of Leja points reached without convergence!!")
            break

    return np.real(polynomial), ii