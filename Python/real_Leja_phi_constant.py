import numpy as np
from Divided_Difference import Divided_Difference

def real_Leja_phi(u, dt, RHS_function, interp_function, integrator_coeffs, c, Gamma, Leja_X, phi_function, tol):
    """
    Parameters
    ----------
    u                       : 1D vector u (input)
    dt                      : Step size
    RHS_function            : RHS function
    interp_function         : Function to be interpolated
    c                       : Shifting factor
    Gamma                   : Scaling factor
    Leja_X                  : Array of Leja points
    phi_function            : phi function
    tol                     : Accuracy of the polynomial so formed

    Returns
    ----------
    polynomial_array        : Polynomial interpolation of 'interp_function' 
                              multiplied by 'phi_function' at real Leja points
    2*ii                    : # of RHS calls

    """
    
    ### Initialize paramters and arrays
    epsilon = 1e-7
    num_interpolations = len(integrator_coeffs)                                 # Number of interpolations in vertical
    max_Leja_pts = len(Leja_X)                                                  # Max number of Leja points  
    phi_function_array = np.zeros((len(Leja_X), num_interpolations))            # Phi function applied to 'interp_function'
    poly_coeffs = np.zeros((len(Leja_X), num_interpolations))                   # Polynomial coefficients
    polynomial_array = np.zeros((len(interp_function), num_interpolations))     # Polynomial array
    poly_error = np.zeros(max_Leja_pts)                                         # Error estimate
    y = interp_function.copy()                                                  # To avoid changing 'interp_function'
    
    for ij in range(0, num_interpolations):
        
        ### Phi function applied to 'interp_function' (scaled and shifted); scaling down of c and Gamma (i.e. largest and smallest eigenvalue) by dt
        phi_function_array[:, ij] = phi_function(integrator_coeffs[ij] * dt * (c + Gamma*Leja_X))
        
        ### Compute the polynomial coefficients
        poly_coeffs[0:50, ij] = Divided_Difference(Leja_X[0:50], phi_function_array[0:50, ij]) 
        
        ### p_0 term
        polynomial_array[:, ij] = interp_function * poly_coeffs[0, ij]
    
    
    ### p_1, p_2, ...., p_n terms; iterate until converges
    for ii in range(1, max_Leja_pts):
        
        if ii%50 == 0:
            poly_coeffs[ii:ii+50, ij] = Divided_Difference(Leja_X[ii:ii+50], phi_function_array[ii:ii+50, ij]) 
        
        ### Compute numerical Jacobian
        Jacobian_function = (RHS_function(u + (epsilon * y)) - RHS_function(u - (epsilon * y)))/(2*epsilon)

        ### y = y * ((z - c)/Gamma - Leja_X)
        y = (Jacobian_function/Gamma) + (y * (-c/Gamma - Leja_X[ii - 1]))

        ### Error estimate
        poly_error[ii] = np.linalg.norm(y) * abs(poly_coeffs[ii, np.argmax(integrator_coeffs)])
        
        ### Keep adding terms to the polynomial
        for ij in range(0, num_interpolations):

            ### Add the new term to the polynomial
            polynomial_array[:, ij] = polynomial_array[:, ij] + (poly_coeffs[ii, ij] * y)
            
        ### If new term to be added < tol, break loop; safety factor = 0.1
        if  poly_error[ii] < 0.1*tol:
            print("# Leja points: ", ii)
            print("=============================================================")
            break

    return polynomial_array, 2*ii