"""
Created on Thu Jul 23 15:54:52 2020

@author: Pranab JD

Description: -
        Interpolation at Leja points
"""

import numpy as np

################################################################################################

### Phi Functions ('z' is assumed to be a double)

def phi_1(z):
    
    phi_1_array = np.zeros(len(z))
    
    for ii in range(len(z)):
        if abs(z[ii]) <= 1e-7:
            phi_1_array[ii] = 1 + z[ii] * (1./2. + z[ii] * (1./6. + z[ii] * (1./24. + 1./120. * z[ii])))
        else:
            phi_1_array[ii] = (np.exp(z[ii]) - 1)/z[ii]
            
    return phi_1_array

def phi_2(z):
    
    phi_2_array = np.zeros(len(z))
    
    for ii in range(len(z)):
        if abs(z[ii]) <= 1e-6:
            phi_2_array[ii] = 1./2. + z[ii] * (1./6. + z[ii] * (1./24. + z[ii] * (1./120. + 1./720. * z[ii])));
        else:
            phi_2_array[ii] = (np.exp(z[ii]) - z[ii] - 1)/z[ii]**2
        
    return phi_2_array

def phi_3(z):
    
    phi_3_array = np.zeros(len(z))
    
    for ii in range(len(z)):
        if abs(z[ii]) <= 1e-6:
            phi_3_array[ii] = 1./6. + z[ii] * (1./24. + z[ii] * (1./120. + z[ii] * (1./720. + 1./5040. * z[ii])));
        else:
            phi_3_array[ii] = (np.exp(z[ii]) - z[ii]**2/2 - z[ii] - 1)/z[ii]**3
    
    return phi_3_array

def phi_4(z):
    
    phi_4_array = np.zeros(len(z))
    
    for ii in range(len(z)):
        if abs(z[ii]) <= 1e-6:
            phi_4_array[ii] = 1./24. + z[ii] * (1./120. + z[ii] * (1./720. + z[ii] * (1./5040. + 1./40320. * z[ii])));
        else:
            phi_4_array[ii] = (np.exp(z[ii]) - z[ii]**3/6 - z[ii]**2/2 - z[ii] - 1)/z[ii]**4
        
    return phi_4_array

def phi_5(z):
    
    phi_5_array = np.zeros(len(z))
    
    for ii in range(len(z)):
        if abs(z[ii]) <= 1e-5:
            phi_5_array[ii] = 1./120. + z[ii] * (1./720. + z[ii] * (1./5040. + z[ii] * (1./40320. + 1./362880. * z[ii])));
        else:
            phi_5_array[ii] = (np.exp(z[ii]) - z[ii]**4/24 - z[ii]**3/6 - z[ii]**2/2 - z[ii] - 1)/z[ii]**5
        
    return phi_5_array

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

    N = len(X)
    div_diff = func(X)
    
    for ii in range(1, N):
        div_diff[ii:N] = (div_diff[ii:N] - div_diff[ii - 1])/(X[ii:N] - X[ii - 1])

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
    polynomial              : Polynomial interpolation of the matrix exponential multiplied
                              to 'u' at real Leja points
    ii                      : # of RHS calls

    """

    ### Matrix exponential (scaled and shifted)
    def func(xx):
        return np.exp(dt * (c + Gamma*xx))

    ### Array of Leja points
    Leja_X = Leja_Points()   
    
    ### Compute the polynomial coefficients
    coeffs = Divided_Difference(Leja_X, func) 

    ### ------------------------------------------------------------------- ###

    ### a_0 term (form the polynomial)
    poly = u.copy()
    poly = coeffs[0] * poly
    
    ### ------------------------------------------------------------------- ###

    ### a_1, a_2 .... a_n terms
    max_Leja_pts = 500                                      # Max # of Leja points    
    y = u.copy()                                            # To avoid changing input vector 'u'

    ### Iterate until convergence is reached
    for ii in range(1, max_Leja_pts):
        
        ## Compute numerical Jacobian (for linear eqs., this is the RHS evaluation at y)
        Jacobian_function = RHS_func(y)

        ## Re-scale and re-shift
        y = y * (-c/Gamma - Leja_X[ii - 1])
        y = y + (Jacobian_function/Gamma)

        ## Approx. error incurred (accuracy)
        poly_error = np.linalg.norm(y)/len(y) * abs(coeffs[ii])

        ## If new number (next order) to be added < tol, break loop
        if  poly_error < rel_tol:
            # print("No. of Leja points used (real exp) = ", ii)
            # print('----------Tolerance reached---------------')
            poly = poly + (coeffs[ii] * y)
            break

        else:
            ## Add the new term to the polynomial
            poly = poly + (coeffs[ii] * y)

        if ii == max_Leja_pts - 1:
            print("Error!! Max. # of Leja points reached!!")
            break

    ### ------------------------------------------------------------------- ###

    ### Solution
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
                              multiplied by 'phi_func' at real Leja points
    ii * 2                  : # of RHS calls

    """

    ### Phi function applied to 'interp_func' (scaled and shifted)
    def func(xx):

        # np.seterr(divide = 'ignore', invalid = 'ignore')
        zz = (dt * (c + Gamma*xx))
        var = phi_func(zz)

        if phi_func != phi_1 and phi_func != phi_2 and phi_func != phi_3 and phi_func != phi_4 and phi_func != phi_5:
            print('Error: Phi function not defined!!')

        return var

    ### Array of Leja points
    Leja_X = Leja_Points()   
    
    ### Compute the polynomial coefficients
    coeffs = Divided_Difference(Leja_X, func) 

    ### ------------------------------------------------------------------- ###

    ## a_0 term (form the polynomial)
    poly = interp_func.copy()
    poly = coeffs[0] * poly

    ### ------------------------------------------------------------------- ###

    ## a_1, a_2 .... a_n terms
    max_Leja_pts = 500                                      # Max number of Leja points
    y = interp_func.copy()                                  # x values of the polynomial
    epsilon = 1e-7

    ## Iterate until converges
    for ii in range(1, max_Leja_pts):
        
        ## Compute the numerical Jacobian
        Jacobian_function = (RHS_func(u + (epsilon * y)) - RHS_func(u))/epsilon

        ## Re-scale and re-shift
        y = y * (-c/Gamma - Leja_X[ii - 1])
        y = y + (Jacobian_function/Gamma)

        ## Approx. error incurred (accuracy)
        poly_error = np.linalg.norm(y)/len(y) * abs(coeffs[ii])

        ## If new number (next order) to be added < tol, break loop
        if  poly_error < rel_tol:
            # print("No. of Leja points used (real phi) = ", ii)
            # print('----------Tolerance reached---------------')
   
            poly = poly + (coeffs[ii] * y)
            break

        ### To stop diverging
        elif poly_error > 1e5:
            print()
            
            print("Starts to diverge after ", ii, " iterations with dt = ", dt)
            return 5 * interp_func, ii * 2

        else:
           poly = poly + (coeffs[ii] * y)

        # if ii >= max_Leja_pts - 1:
        #     # print(poly_vals[1:ii+1])
        #     print("Max. # of Leja points reached!")
        #     return 5 * interp_func, ii * 2

    ### ------------------------------------------------------------------- ###

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
                              multiplied by 'phi_func' at real Leja points
    ii * 2                  : # of RHS calls

    """

    def func(xx):

        # np.seterr(divide = 'ignore', invalid = 'ignore')

        zz = (1j * dt * (c + Gamma*xx))
        var = phi_func(zz)

        if phi_func != phi_1 and phi_func != phi_2 and phi_func != phi_3 and phi_func != phi_4 and phi_func != phi_5:
            print('Error: Phi function not defined!!')

        return var

    ### Array of Leja points
    Leja_X = Leja_Points()
    
    ### Compute the polynomial coefficients
    coeffs = Divided_Difference(Leja_X, func) 

    ### ------------------------------------------------------------------- ###

    ### a_0 term (form the polynomial)
    poly = u.copy()
    poly = coeffs[0] * poly
    
    ### ------------------------------------------------------------------- ###

    ## a_1, a_2 .... a_n terms
    epsilon = 1e-7
    max_Leja_pts = 500                                      # Max number of Leja points    
    y = interp_func.copy()                                  # To avoid changing input vector 'interp_func'

    ### ------------------------------------------------------------------- ###

    ### Iterate until converges
    for ii in range(1, max_Leja_pts):

        ## Compute numerical Jacobian
        Jacobian_function = (RHS_func(u + (epsilon * y)) - RHS_func(u))/epsilon

        ## Re-scale and re-shift
        y = y * (-c/Gamma - Leja_X[ii - 1])
        y = y + (-1j * Jacobian_function/Gamma )

        ## Error incurred
        poly_error = np.linalg.norm(y)/len(y) * abs(coeffs[ii])

        ## If new number (next order) to be added < tol, break loop
        if  poly_error < rel_tol:
            # print("No. of Leja points used (imag phi) = ", ii)
            # print('----------Tolerance reached---------------')
            poly = poly + (coeffs[ii] * y)
            break

        ### To stop diverging, restart simulations with smaller dt
        # elif poly_error > 1e5:
        #     # print()
        #     # print(poly_vals[1:ii])
        #     print("Starts to diverge after ", ii, " iterations with dt = ", dt)
        #     return 5*interp_func, ii * 2

        else:
            ## Add the new term to the polynomial
            poly = poly + (coeffs[ii] * y)

        if ii == max_Leja_pts - 1:
            print("Error!! Max. # of Leja points reached!!")
            break

    ### ------------------------------------------------------------------- ###

    ### Solution
    polynomial = poly.copy()

    return np.real(polynomial), ii * 2

################################################################################################