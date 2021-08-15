"""
Created on Thu Jul 23 15:54:52 2020

@author: Pranab JD

Description: -
        Interpolation at Leja points
"""

import os
import numpy as np

################################################################################################

### Phi Functions ('z' is assumed to be a double)

def phi_1(z):
    if abs(z) <= 1e-6:
        return 1 + z * (1./2. + z * (1./6. + z * (1./24. + 1./120. * z)))
    else:
        return (np.exp(z) - 1)/z

def phi_2(z):
    if abs(z) <= 1e-5:
        return 1./2. + z * (1./6. + z * (1./24. + z * (1./120. + 1./720. * z)));
    else:
        return (np.exp(z) - z - 1)/z**2

def phi_3(z):
    if abs(z) <= 1e-5:
        return 1./6. + z * (1./24. + z * (1./120. + z * (1./720. + 1./5040. * z)));
    else:
        return (np.exp(z) - z**2/2 - z - 1)/z**3

def phi_4(z):
    if abs(z) <= 1e-4:
        return 1./24. + z * (1./120. + z * (1./720. + z * (1./5040. + 1./40320. * z)));
    else:
        return (np.exp(z) - z**3/6 - z**2/2 - z - 1)/z**4

################################################################################################

def Leja_Points():
    """
    Load Leja points

    """
    dt = np.dtype("f8")
    
    return np.fromfile('real_leja_d.bin', dtype = dt)


def Divided_Difference(X, func):
    """
    Parameters
    ----------
    X       : Leja points
    func    : func(X)

    Returns
    -------
    div_diff : Polynomial coefficients

    """

    div_diff = func(X)

    for ii in range(1, int(len(X))):
        for jj in range(ii):

            div_diff[ii] = (div_diff[ii] - div_diff[jj])/(X[ii] - X[jj])

    return div_diff

################################################################################################

def real_Leja_exp(u, dt, RHS_func, c, Gamma, rel_tol):
    """
    Parameters
    ----------
    u                       : 1D vector u (input)
    dt                      : Step size
    RHS_func	            : RHS function
    c                       : Shifting factor
    Gamma                   : Scaling factor
    rel_tol                 : Accuracy of the polynomial so formed
    
    Returns
    ----------
    polynomial              : Polynomial interpolation of 
                              matrix exponential of 'u'
                              at real Leja points
    ii                      : # of RHS calls

    """

    def func(xx):
        return np.exp(dt * (c + Gamma*xx))

    Leja_X = Leja_Points()                                 # Leja Points
    coeffs = Divided_Difference(Leja_X, func)              # Polynomial Coefficients

    ### ------------------------------------------------------------------- ###

    ## a_0 term
    poly = u.copy()
    poly = coeffs[0] * poly

    ## a_1, a_2 .... a_n terms
    max_Leja_pts = 100                                     # Max number of Leja points
    y = u.copy()                                           # x values of the polynomial
    poly_vals = np.zeros(max_Leja_pts)                     # Array for error incurred
    y_val = np.zeros((max_Leja_pts, len(u)))               # Stores x values till polynomial converges
    scale_fact = 1/Gamma                                   # Re-scaling factor

    ### ------------------------------------------------------------------- ###

    ## Iterate until convergence is reached
    for ii in range(1, max_Leja_pts):

        shift_fact = -c * scale_fact - Leja_X[ii - 1]      # Re-shifting factor

        ## function: function to be multiplied to the matrix exponential of the Jacobian
        function = y.copy()
        
        ## Compute the numerical Jacobian
        Jacobian_function = RHS_func(function)

        ## Re-scale and re-shift
        y = y * shift_fact
        y = y + scale_fact * Jacobian_function

        ## Error incurred
        poly_vals[ii] = (sum(abs(y)**2)/len(y))**0.5 * abs(coeffs[ii])

        ## If new number (next order) to be added < tol, ignore it
        if  poly_vals[ii] < rel_tol:
            # print('No. of Leja points used (real phi) = ', ii)
            # print('----------Tolerance reached---------------')
            y_val[ii, :] = coeffs[ii] * y
            break

        ## To stop diverging
        elif poly_vals[ii] > 1e13:
            return 5 * u, ii

        else:
            y_val[ii, :] = coeffs[ii] * y

        if ii >= max_Leja_pts - 1:
            print("Error!! Max. # of Leja points reached!!")
            return 5 * u, ii

    ### ------------------------------------------------------------------- ###

    ### Choose polynomial terms up to the smallest term, ignore the rest
    if np.argmin(poly_vals[np.nonzero(poly_vals)]) + 1 == 0:               # Tolerance reached
        min_poly_val_x = np.argmin(poly_vals[np.nonzero(poly_vals)])

    else:
        min_poly_val_x = np.argmin(poly_vals[np.nonzero(poly_vals)]) + 1   # Starts to diverge

    ### Form the polynomial
    for jj in range(1, min_poly_val_x + 1):
        poly = poly + y_val[jj, :]

    ## Solution
    polynomial = poly.copy()

    return polynomial, ii

################################################################################################

def real_Leja_phi(u, dt, RHS_func, interp_func, c, Gamma, phi_func, rel_tol):
    """
    Parameters
    ----------
    u                       : 1D vector u (input)
    dt                      : Step size
    RHS_func	            : RHS function
    interp_func             : function to be multiplied to phi function
    c                       : Shifting factor
    Gamma                   : Scaling factor
    phi_func                : phi function
    rel_tol                 : Accuracy of the polynomial so formed

    Returns
    ----------
    polynomial              : Polynomial interpolation of 'interp_func' 
                              multiplied by 'phi_func'
                              at real Leja points
    ii * 2                  : # of RHS calls

    """

    def func(xx):

        np.seterr(divide = 'ignore', invalid = 'ignore')

        zz = (dt * (c + Gamma*xx))
        var = phi_func(zz)

        if phi_func != phi_1 or phi_func != phi_2 or phi_func != phi_3 or phi_func != phi_4:
            print('Error: Phi function not defined!!')

        return var

    ### ------------------------------------------------------------------- ###
    
    print("Curretn directory: ", os.getcwd())

    Leja_X = Leja_Points()                                  # Leja Points
    coeffs = Divided_Difference(Leja_X, func)               # Polynomial Coefficients

    ### ------------------------------------------------------------------- ###

    ## a_0 term
    poly = interp_func.copy()
    poly = coeffs[0] * poly

    ### ------------------------------------------------------------------- ###

    ## a_1, a_2 .... a_n terms
    y = interp_func.copy()                                  # x values of the polynomial
    max_Leja_pts = len(coeffs)                              # Max number of Leja points
    poly_vals = np.zeros(max_Leja_pts)                      # Array for error incurred
    epsilon = 1e-7
    y_val = np.zeros((max_Leja_pts, len(u)))                # Stores x values till polynomial converges
    scale_fact = 1/Gamma                                    # Re-scaling factor

    ### ------------------------------------------------------------------- ###

    ## Iterate until convergence is reached
    for ii in range(1, max_Leja_pts):

        shift_fact = -c * scale_fact - Leja_X[ii - 1]       # Re-shifting factor

        ## function: function to be multiplied to the phi function applied to Jacobian
        function = y.copy()
        
        ## Compute the numerical Jacobian
        Jacobian_function = (RHS_func(u + (epsilon * function)) - RHS_func(u))/epsilon

        ## Re-scale and re-shift
        y = y * shift_fact
        y = y + scale_fact * Jacobian_function

        ## Error incurred
        poly_vals[ii] = (sum(abs(y)**2)/len(y))**0.5 * abs(coeffs[ii])

        ## If new number (next order) to be added < tol, ignore it
        if  poly_vals[ii] < rel_tol:
            # print('No. of Leja points used (real phi) = ', ii)
            # print('----------Tolerance reached---------------')
            y_val[ii, :] = coeffs[ii] * y
            break

        ## To stop diverging
        elif poly_vals[ii] > 1e13:
            return 5 * interp_func

        else:
            y_val[ii, :] = coeffs[ii] * y

        if ii >= max_Leja_pts - 1:
            return 5 * interp_func

    ### ------------------------------------------------------------------- ###

    ### Choose polynomial terms up to the smallest term, ignore the rest
    if np.argmin(poly_vals[np.nonzero(poly_vals)]) + 1 == 0:               # Tolerance reached
        min_poly_val_x = np.argmin(poly_vals[np.nonzero(poly_vals)])

    else:
        min_poly_val_x = np.argmin(poly_vals[np.nonzero(poly_vals)]) + 1   # Starts to diverge

    ### Form the polynomial
    for jj in range(1, min_poly_val_x + 1):
        poly = poly + y_val[jj, :]

    ## Solution
    polynomial = poly.copy()

    return polynomial, ii * 2

################################################################################################

def imag_Leja_phi(u, dt, RHS_func, interp_func, c, Gamma, phi_func, rel_tol):
    """
    Parameters
    ----------
    u                       : 1D vector u (input)
    dt                      : Step size
    RHS_func	            : RHS function
    interp_func             : function to be multiplied to phi function
    c                       : Shifting factor
    Gamma                   : Scaling factor
    phi_func                : phi function
    rel_tol                 : Accuracy of the polynomial so formed

    Returns
    ----------
    polynomial              : Polynomial interpolation of 'interp_func' 
                              multiplied by 'phi_func'
                              at real Leja points
    ii * 2                  : # of RHS calls

    """

    def func(xx):

        np.seterr(divide = 'ignore', invalid = 'ignore')

        zz = (1j * dt * (c + Gamma*xx))
        var = phi_func(zz)

        if phi_func != phi_1 or phi_func != phi_2 or phi_func != phi_3 or phi_func != phi_4:
            print('Error: Phi function not defined!!')

        return var

    ### ------------------------------------------------------------------- ###

    Leja_X = Leja_Points()                                  # Leja Points
    coeffs = Divided_Difference(Leja_X, func)               # Polynomial Coefficients

    ### ------------------------------------------------------------------- ###

    ## a_0 term
    poly = interp_func.copy()
    poly = coeffs[0] * poly

    ### ------------------------------------------------------------------- ###

    ## a_1, a_2 .... a_n terms
    y = interp_func.copy()                                  # x values of the polynomial
    max_Leja_pts = len(coeffs)                              # Max number of Leja points
    poly_vals = np.zeros(max_Leja_pts)                      # Array for error incurred
    epsilon = 1e-7
    y_val = np.zeros((max_Leja_pts, len(u)))                # Stores x values till polynomial converges
    scale_fact = 1/Gamma                                    # Re-scaling factor

    ### ------------------------------------------------------------------- ###

    ## Iterate until convergence is reached
    for ii in range(1, max_Leja_pts):

        shift_fact = -c * scale_fact - Leja_X[ii - 1]       # Re-shifting factor

        ## function: function to be multiplied to the phi function applied to Jacobian
        function = y.copy()
        
        ## Compute the numerical Jacobian
        Jacobian_function = (RHS_func(u + (epsilon * function)) - RHS_func(u))/epsilon

        ## Re-scale and re-shift
        y = y * shift_fact
        y = y + scale_fact * Jacobian_function * (-1j)

        ## If new number (next order) to be added < tol, ignore it
        if  poly_vals[ii] < rel_tol:
            # print('No. of Leja points used (real phi) = ', ii)
            # print('----------Tolerance reached---------------')
            y_val[ii, :] = coeffs[ii] * y
            break

        ## To stop diverging
        elif poly_vals[ii] > 1e13:
            return 5 * interp_func, ii * 2

        else:
            y_val[ii, :] = coeffs[ii] * y

        if ii >= max_Leja_pts - 1:
            return 5 * interp_func, ii * 2

    ### ------------------------------------------------------------------- ###

    ### Choose polynomial terms up to the smallest term, ignore the rest
    if np.argmin(poly_vals[np.nonzero(poly_vals)]) + 1 == 0:               # Tolerance reached
        min_poly_val_x = np.argmin(poly_vals[np.nonzero(poly_vals)])

    else:
        min_poly_val_x = np.argmin(poly_vals[np.nonzero(poly_vals)]) + 1   # Starts to diverge

    ### Form the polynomial
    for jj in range(1, min_poly_val_x + 1):
        poly = poly + y_val[jj, :]

    ## Solution
    polynomial = poly.copy()

    return np.real(polynomial), ii * 2

################################################################################################