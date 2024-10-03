import numpy as np
from bisect import bisect_left
from Divided_Difference import Divided_Difference

def real_Leja_linear_exp(u, T_f, init_subs, RHS_function, integrator_coeff, c, Gamma, Leja_X, tol):
    """
    Computes the polynomial interpolation of matrix exponential applied to 'u' at real Leja points.


        Parameters
        ----------
        u                       : numpy array
                                    State variable(s)
        T_f                     : double
                                    Step size
        RHS_function            : user-defined function 
                                    RHS function
        integrator_coeff        : int
                                    Point where phi function is to be evaluated
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
        total_iters             : int
                                    Total number of Leja points used

    """
    
    ###? Initialize parameters and arrays
    y = u.copy()                                                  #* To avoid changing 'u'
    y_backup = u.copy()                                           #* Backup y - To avoid changing 'u'
    dt = T_f/init_subs                                            #* Initial substep size
    dt_backup = dt                                                #* To be used if dt is increased by large value
    
    max_Leja_pts = len(Leja_X)                                    #* Max number of Leja points
    time_elapsed = 0                                              #* Simulation time elapsed                                          
    substeps = 0                                                  #* Counter for number of substeps
    total_iters = 0                                               #* Counter for Leja iterations
    convergence = [0]                                             #* Check for convergence
    
    ###! Time loop
    while time_elapsed < T_f:
        
        ###? Array to store error incurred (needs to be set to zeros for every substep)
        poly_error = np.zeros(max_Leja_pts);                            
        
        ###* Adjust final time substep
        if time_elapsed + dt > T_f:
            dt = T_f - time_elapsed
        
        ###? Compute polynomial coefficients
        poly_coeffs = Divided_Difference(Leja_X, np.exp(integrator_coeff * dt * (c + Gamma*Leja_X)))
            
        ###? Set y = polynomial; save y_backup (same dt)
        if convergence[-1] == 1:
            
            y = polynomial;
            y_backup = polynomial;

        ###? Set 'y' to previous value (reduce dt)
        elif convergence[-1] == 0:

            y = y_backup;
            
        ###? Form the first term of the s^{th} polynomial: p_0 = d_0 * y_0
        polynomial = poly_coeffs[0] * y

        ###? p_1, p_2, ...., p_n terms; iterate until converges
        for ii in range(1, max_Leja_pts):

            ###? y = y * ((z - c)/Gamma - Leja_X)
            y = (RHS_function(y)/(T_f*Gamma)) + (y * (-c/Gamma - Leja_X[ii - 1]))
            
            ###? Keep adding terms to the polynomial
            polynomial = polynomial + (poly_coeffs[ii] * y)

            ###? Error estimate; poly_error = |coeffs[nn]| ||y||
            poly_error[ii] = np.linalg.norm(y) * abs(poly_coeffs[ii])
            
            ###! Warning: Check for diverging values, if so, restart iteration with smaller dt
            if ii == max_Leja_pts - 1 or poly_error[ii] > 1e10:

                print("Computations wasted: ", ii, " dt = ", dt)
                
                ###* Update parameters
                convergence.append(0)
                total_iters = total_iters + ii

                ###? Check whether or not this is one of the initial stages
                if substeps != 0:
                    ###* If not, slowly decrease dt (since Leja converged for previous dt, we are in the vicinity)
                    dt = 0.95*dt_backup
                else:
                    ###* If yes, drastically decrese dt (get to a value of dt such that Leja converges)
                    dt = 0.15 * dt
                break

            ###? If new term to be added < tol, break loop
            if  poly_error[ii] < (tol*np.linalg.norm(polynomial) + tol):
                
                # print("Converged! # of Leja points used (exp): ", ii, " dt = ", dt)
                
                ###* Update parameters
                convergence.append(1)
                dt_backup = dt
                time_elapsed = time_elapsed + dt
                total_iters = total_iters + ii
                substeps = substeps + 1

                ###? If simulations never failed to converged, increase dt
                if any(conv == 0 for conv in convergence[bisect_left(convergence, 1):-1]) == False:
                    dt = 1.002*dt                    
                
                break

    # print(substeps)
    return polynomial, total_iters, substeps