from Leja_Interpolation import *

################################################################################################

def EXPRB32(u, dt, RHS_function, c, Gamma, tol, Real_Imag):
    """
    Parameters
    ----------
	u               : 1D vector u (input)
	dt              : Step size
	RHS_function	: RHS function
    c               : Shifting factor
    Gamma           : Scaling factor
    tol             : Accuracy of the polynomial so formed
    Real_Imag       : 0 - Real, 1 - Imaginary

    Returns
    -------
    u_exprb2        : 1D vector u (output) after time dt (2nd order)
    u_exprb3        : 1D vector u (output) after time dt (3rd order)
    num_rhs_calls   : # of RHS calls
    
    """

    ### Interpolate on either real Leja or imaginary Leja points.
    if Real_Imag == 0:
        Leja_phi = real_Leja_phi
    elif Real_Imag == 1:
        Leja_phi = imag_Leja_phi
    else:
        print("Error!! Choose 0 for real or 1 for imaginary Leja points.")
    
    ### RHS of PDE at u
    f_u = RHS_function(u)
    
    ### Function to compute the nonlinear remainder at stage 'y'
    def Nonlinear_remainder(y):
        
        epsilon = 1e-7
        
        ### J(u) * y
        Linear_y = (RHS_function(u + (epsilon * y)) - f_u)/epsilon

        ### F(y) = f(y) - (J(u) * y)
        Nonlinear_y = RHS_function(y) - Linear_y
        
        return Nonlinear_y

    ############## --------------------- ##############

    ### Internal stage 1; interpolation 1
    u_flux, rhs_calls_1, convergence = Leja_phi(u, dt, RHS_function, f_u*dt, c, Gamma, phi_1, tol)
    
    ## If it does not converge, return (try with smaller dt)
    if convergence == 0:

        u_exprb2 = u;
        u_exprb3 = 2.1*u;
        num_rhs_calls = rhs_calls_1;

        return u_exprb2, u_exprb3, num_rhs_calls

    ### 2nd order solution
    u_exprb2 = u + u_flux

    ############## --------------------- ##############

    ### Nonlinear remainder at u
    Nonlinear_u = Nonlinear_remainder(u)

    ### Nonlinear remainder at u_exprb2
    Nonlinear_a = Nonlinear_remainder(u_exprb2)

    ############## --------------------- ##############

    ### Interpolation 2 (Nonlinear remainders always converge)
    u_nl_3, rhs_calls_2, convergence = Leja_phi(u, dt, RHS_function, 2*(Nonlinear_a - Nonlinear_u)*dt, c, Gamma, phi_3, tol)
    
    ### 3rd order solution
    u_exprb3 = u_exprb2 + u_nl_3

    ## Proxy of computational cost
    num_rhs_calls = rhs_calls_1 + rhs_calls_2 + 5

    return u_exprb2, u_exprb3, num_rhs_calls