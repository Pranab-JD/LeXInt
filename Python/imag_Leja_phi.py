import numpy as np
from Divided_Difference import Divided_Difference

def imag_Leja_phi(u, dt, RHS_function, interp_vector, integrator_coeffs, c, Gamma, Leja_X, phi_function, tol):
    """
    To be used for computing "phi_function" applied to "interp_vector" (at imaginary Leja points) for
    exponential integrators that computes the Jacobian at every time step.


        Parameters
        ----------
        u                       : numpy array
                                    State variable(s)
        dt                      : double
                                    Step size
        RHS_function            : user-defined function 
                                    RHS function
        interp_vector           : numpy array
                                    Vector to be interpolated
        c                       : double
                                    Shifting factor
        Gamma                   : double
                                    Scaling factor
        Leja_X                  : numpy array
                                    Array of Leja points
        phi_function            : function
                                    phi function
        tol                     : double
                                    Accuracy of the polynomial so formed

        Returns
        ----------
        polynomial              : numpy array(s)
                                    Polynomial interpolation of 'interp_vector' 
                                    multiplied by 'phi_function' at real Leja points
        ii+1                    : int
                                    # of RHS calls
        convergence             : int
                                    0 -> did not converge, 1 -> converged

    """
    
    ### Initialize parameters and arrays
    epsilon = 1e-7
    convergence = 0                                                                             # 0 -> did not converge, 1 -> converged
    num_interpolations = len(integrator_coeffs)                                                 # Number of interpolations in vertical
    max_Leja_pts = len(Leja_X)                                                                  # Max number of Leja points
    phi_function_array = np.zeros((len(Leja_X), num_interpolations), dtype = "complex")         # Phi function applied to 'interp_vector'
    poly_coeffs = np.zeros((len(Leja_X), num_interpolations), dtype = "complex")                # Polynomial coefficients
    polynomial = np.zeros((len(interp_vector), num_interpolations), dtype = "complex")          # Polynomial array
    rhs_u = RHS_function(u)                                                                     # RHS evaluated at 'u'
    y = interp_vector.copy() + 0*1j                                                             # To avoid changing 'interp_vector'
    
    ### Loop for vertical implementation
    for ij in range(0, num_interpolations):
        
        ### Phi function applied to 'interp_vector' (scaled and shifted)
        phi_function_array[:, ij] = phi_function(integrator_coeffs[ij] * dt * (c + Gamma*Leja_X) * 1j)

        ### Compute polynomial coefficients
        poly_coeffs[:, ij] = Divided_Difference(Leja_X, phi_function_array[:, ij]) 

        ### p_0 term
        polynomial[:, ij] = interp_vector * poly_coeffs[0, ij] + 0*1j

    ### p_1, p_2, ...., p_n terms; iterate until converges
    for ii in range(1, max_Leja_pts):
        
        ### Compute numerical Jacobian
        Jacobian_function = (RHS_function(u + (epsilon * y)) - rhs_u)/epsilon
        
        ### y = y * ((z - c)/Gamma - Leja_X)
        y = (-1j * Jacobian_function/Gamma) + (y * (-c/Gamma - Leja_X[ii - 1]))

        ### Error estimate
        poly_error = np.linalg.norm(y) * abs(poly_coeffs[ii, np.argmax(integrator_coeffs)])     

        ### Keep adding terms to the polynomial
        for ij in range(0, num_interpolations):
            
            ### To prevent diverging, restart simulations with smaller dt
            if poly_error > 1e17:
                convergence = 0
                polynomial[:, ij] = u
                return np.real(polynomial), ii+1, convergence
            
            ### Add the new term to the polynomial
            polynomial[:, ij] = polynomial[:, ij] + (poly_coeffs[ii, ij] * y)
            
        ### If new term to be added < tol, break loop; safety factor = 0.25
        if  poly_error < 0.25*tol*np.linalg.norm(polynomial):
            convergence = 1
            # print("Leja points used: ", ii)
            break
        
        ### Warning flags
        if ii == max_Leja_pts - 1:
            print("Warning!! Max. # of Leja points reached without convergence!! Try increasing the number of Leja points. Max available: 10000.")
            break

    return np.real(polynomial), ii+1, convergence
