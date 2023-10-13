import numpy as np

###? Jacobian_vector = (RHS(u + epsilon*v) - RHS(u))/epsilon
def Jacobian(RHS, u, v, *args):
    
    ###* epsilon is normalised to norm(u)
    epsilon = 1e-7 * np.linalg.norm(u)
    
    ###* J(u) * v = (RHS(u + epsilon*v) - RHS(u))/epsilon
    Jacobian_vector = (RHS(u + (epsilon * v), *args) - RHS(u, *args))/epsilon
    
    return Jacobian_vector
