import numpy as np
from real_Leja_linear_exp import real_Leja_linear_exp

def A_tilde(A, B, v):
    """
    Form the augmented matrix.

    Args:
        A (function handle R^n -> R^n)
        B (matrix, n*p)
        v (vector, n+p)

    Returns:
        y (vector) : A_tilde . v, where A_tilde = [A B; 0 K] and K = [0 I; 0 0]
    
        
    Reference: 

        R.B. Sidje, Expokit: A Software Package for Computing Matrix Exponentials, ACM Trans. Math. Softw. 24 (1) (1998) 130 - 156.
        doi:10.1145/285861.285868

    """

    [p, n] = np.shape(B)
    
    y = np.concatenate([A(v[0:n]).reshape(1, n) + np.dot(v[n:n+p].reshape(1, p), B), [v[n+1:n+p]], np.array([0]).reshape(1, 1)], axis = 1)
    
    return y.reshape(np.shape(y)[1])


def linear_phi(interp_vector, T_final, substeps, Jacobian_vector, integrator_coeff, c, Gamma, Leja_X, tol):
    """
    Evaluates a linear combinaton of the phi functions as the 
    exponential of an augmented matrix.
    
     polynomial[0:n] = phi_0(A) u(:, 1) + phi_1(A) u(:, 2) + ... + phi_p(A) u(:, p+1)

    Args:
        interp_vector (vector n*(p+1))      : Vector to evaluated/interpolated
        dt (double)                         : Step size
        Jacobian_vector (function handle)   : Jacobian-vector product (multiplied by dt)
        c (double)                          : Shifting factor
        Gamma (double)                      : Scaling factor
        Leja_X (vector)                     : Array of Leja points
        tol (double)                        : Accuracy of the polynomial so formed

    Returns:
         polynomial[0:n] (vector)           : Linear combinaton of the phi functions
    

    Reference: 

        R.B. Sidje, Expokit: A Software Package for Computing Matrix Exponentials, ACM Trans. Math. Softw. 24 (1) (1998) 130 - 156.
        doi:10.1145/285861.285868

    """
    
    ############## --------------------- ##############

    ###TODO: Interpolate on either real Leja or imaginary Leja points
    # if Real_Imag == 0:
    #     Leja_phi = real_Leja_phi
    # elif Real_Imag == 1:
    #     Leja_phi = imag_Leja_phi
    # else:
    #     print("Error!! Choose 0 for real or 1 for imaginary Leja points.")
    
    ############## --------------------- ##############
    
    [m, n] = np.shape(interp_vector)
    B = np.flipud(interp_vector[1:m])
    p = m - 1

    Atx = lambda x: A_tilde(Jacobian_vector, B, x)
    
    v = np.concatenate([interp_vector[0], np.zeros(p-1), [1]])

    polynomial, rhs_calls, substeps = real_Leja_linear_exp(v, T_final, substeps, Atx, integrator_coeff, c, Gamma, Leja_X, tol)
    
    return polynomial[0:n], rhs_calls, substeps