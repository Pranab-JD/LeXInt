import numpy as np
from Divided_Difference import Divided_Difference

def real_Leja_phi_nl(u, dt, RHS_function, c, Gamma, Leja_X, phi_function, tol):
    """
    Computes the polynomial interpolation of phi_function applied to 'u' at real Leja points.
    
    
        Parameters
        ----------
        u                       : numpy array
                                    Vector multiplied to phi function
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
        phi_function            : function
                                    Phi function (typically phi_1)
        tol                     : double
                                    Accuracy of the polynomial so formed

        Returns
        ----------
        polynomial              : numpy array
                                    Polynomial interpolation of 'u' multiplied
                                    to phi_function at real Leja points
        ii+1                    : int
                                    Number of RHS calls

    """

    ###? Initialize parameters and arrays
    max_Leja_pts = len(Leja_X)                                    #* Max number of Leja points  
    y = u.copy()                                                  #* To avoid changing 'interp_function'
        
    ###? Phi function applied to 'interp_function' (scaled and shifted)
    phi_function_array = phi_function(dt * (c + Gamma*Leja_X))
    
    ###? Compute polynomial coefficients
    poly_coeffs = Divided_Difference(Leja_X, phi_function_array) 

    ###? Form the polynomial: 1st term (p_0)
    polynomial = poly_coeffs[0] * u
    
    ###? p_1, p_2, ...., p_n terms; iterate until converges
    for ii in range(1, max_Leja_pts):

        ###? y = y * ((z - c)/Gamma - Leja_X)
        y = (RHS_function(y)/Gamma) + (y * (-c/Gamma - Leja_X[ii - 1]))

        ###? Error estimate; poly_error = |coeffs[nn]| ||y||
        poly_error = np.linalg.norm(y) * abs(poly_coeffs[ii])

        ###? Add the new term to the polynomial
        polynomial = polynomial + (poly_coeffs[ii] * y)
        
        ###? If new term to be added < tol, break loop
        if  poly_error < tol*np.linalg.norm(polynomial):
            # print("Converged! # of Leja points used (phi nl): ", ii)
            break
        
        ###! Warning flags
        if ii == max_Leja_pts - 1:
            print("Warning!! Max. # of Leja points reached without convergence!!")
            print("Max. Leja points currently set to", max_Leja_pts)
            print("Try increasing the number of Leja points. Max available: 10000.\n")
            break

    return polynomial, ii