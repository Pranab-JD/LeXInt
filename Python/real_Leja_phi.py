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
    ii + 1                  : # of RHS calls
    convergence             : 0 -> did not converge, 1-> converged

    """
    
    ### Number of interpolations at one go
    num_interpolations = len(integrator_coeffs)

    ### Phi function applied to 'interp_function' (scaled and shifted)
    phi_function_array = np.zeros((len(Leja_X), num_interpolations))
    
    for ij in range(0, num_interpolations):
        phi_function_array[:, ij] = phi_function(integrator_coeffs[ij] * dt * (c + Gamma*Leja_X))

    ### Compute the polynomial coefficients
    poly_coeffs = np.zeros((len(Leja_X), num_interpolations))
    
    for ij in range(0, num_interpolations):
        poly_coeffs[:, ij] = Divided_Difference(Leja_X, phi_function_array[:, ij]) 

    ### ------------------------------------------------------------------- ###
    
    ### Compute the polynomial
    polynomial_array = np.zeros((len(interp_function), num_interpolations))

    ### p_0 term
    for ij in range(0, num_interpolations):
        polynomial_array[:, ij] = interp_function * poly_coeffs[0, ij]

    ### p_1, p_2, ...., p_n terms
    epsilon = 1e-7
    convergence = 0                                         # 0 -> did not converge, 1-> converged
    max_Leja_pts = 100                                      # Max number of Leja points    
    rhs_u = RHS_function(u)                                 # RHS of the function at u
    y = interp_function.copy()                              # To avoid changing 'interp_function'
    
    ### Iterate until converges
    for ii in range(1, max_Leja_pts):
        
        ### Compute numerical Jacobian
        Jacobian_function = (RHS_function(u + (epsilon * y)) - rhs_u)/epsilon

        ### Re-scale and re-shift
        y = y * (-c/Gamma - Leja_X[ii - 1])
        y = y + (Jacobian_function/Gamma)

        ### Error estimate
        poly_error = np.mean(abs(y)) * abs(poly_coeffs[ii, 0:num_interpolations])
        
        ########### -------------------------------------- ###########

        ### Keep adding terms to the polynomial
        for ij in range(0, num_interpolations):
            
            ### To prevent diverging, restart simulations with smaller dt
            if poly_error[ij] > 1e7:
                convergence = 0
                return interp_function, ii, convergence
            
            ### Add the new term to the polynomial
            polynomial_array[:, ij] = polynomial_array[:, ij] + (poly_coeffs[ii, ij] * y)
            
        ### If new term to be added < tol, break loop; safety factor = 0.1
        if  poly_error[-1] < 0.1*tol:
            convergence = 1
            break

    return polynomial_array, ii + 1, convergence
