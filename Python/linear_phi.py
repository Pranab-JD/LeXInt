import numpy as np
from A_tilde import A_tilde
from real_Leja_linear_exp import *

def linear_phi(interp_vector, dt, Jacobian_vector, c, Gamma, Leja_X, tol):
    
    [m, n] = np.shape(interp_vector)
    B = np.flipud(interp_vector[1:m])
    p = m - 1

    Atx = lambda x: A_tilde(Jacobian_vector, B, x)

    v = np.concatenate([interp_vector[0], np.zeros(p-1), [1]])

    polynomial = real_Leja_exp(v, dt, Atx, c, Gamma, Leja_X, tol)
    
    return polynomial[0]