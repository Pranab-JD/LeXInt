import numpy as np
from Divided_Difference import Divided_Difference

def imag_Leja_phi(u, dt, RHS_function, interp_function, integrator_coeffs, c, Gamma, Leja_X, phi_function, tol):
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
                              multiplied by 'phi_function' at imaginary Leja points
    ii + 1                  : # of RHS calls
    convergence             : 0 -> did not converge, 1-> converged

    """
    
    ### Number of interpolations in vertical
    num_interpolations = len(integrator_coeffs)

    ### Phi function applied to 'interp_function' (scaled and shifted); c & Gamma (largest eigenvalue) scaled by dt
    phi_function_array = np.zeros((len(Leja_X), num_interpolations), dtype = "complex")
    
    for ij in range(0, num_interpolations):
        phi_function_array[:, ij] = phi_function(integrator_coeffs[ij] * dt * (c + Gamma*Leja_X) * 1j)

    ### Compute the polynomial coefficients
    poly_coeffs = np.zeros((len(Leja_X), num_interpolations), dtype = "complex")
    
    for ij in range(0, num_interpolations):
        poly_coeffs[:, ij] = Divided_Difference(Leja_X, phi_function_array[:, ij]) 

    ### ------------------------------------------------------------------- ###
    
    ### Compute the polynomial
    polynomial_array = np.zeros((len(interp_function), num_interpolations), dtype = "complex")

    ### p_0 term
    for ij in range(0, num_interpolations):
        polynomial_array[:, ij] = interp_function * poly_coeffs[0, ij] + 0*1j

    ### p_1, p_2, ...., p_n terms
    epsilon = 1e-7
    convergence = 0                                         # 0 -> did not converge, 1-> converged
    max_Leja_pts = 100                                      # Max number of Leja points    
    y = interp_function.copy() + 0*1j                       # To avoid changing 'interp_function'
    
    ### Iterate until converges
    for ii in range(1, max_Leja_pts):
        
        ### Compute numerical Jacobian
        Jacobian_function = (RHS_function(u + (epsilon * y)) - RHS_function(u - (epsilon * y)))/(2*epsilon)

        ### Re-scale and re-shift
        y = y * (-c/Gamma - Leja_X[ii - 1])
        y = y + (Jacobian_function/Gamma) * (-1j)

        ### Error estimate
        poly_error = np.mean(abs(y)) * abs(poly_coeffs[ii, np.argmax(integrator_coeffs)])     
        
        ########### -------------------------------------- ###########

        ### Keep adding terms to the polynomial
        for ij in range(0, num_interpolations):
            
            ### To prevent diverging, restart simulations with smaller dt
            if poly_error > 1e7 or ii == max_Leja_pts - 1:
                convergence = 0
                polynomial_array[:, ij] = interp_function
                return np.real(polynomial_array), 2*ii, convergence
            
            ### Add the new term to the polynomial
            polynomial_array[:, ij] = polynomial_array[:, ij] + (poly_coeffs[ii, ij] * y)
            
        ### If new term to be added < tol, break loop; safety factor = 0.1
        if  poly_error < 0.1*tol:
            convergence = 1
            break

    return np.real(polynomial_array), 2*ii, convergence