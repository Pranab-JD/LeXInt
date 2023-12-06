import numpy as np

###? Jacobian_vector = (RHS(u + epsilon*v) - RHS(u))/epsilon
def Jacobian(RHS, u, v, rhs_u, *args):
   
    ###* epsilon is normalised to norm(u)
    epsilon = 1e-7 * np.linalg.norm(u)
    
    ###* J(u) * v = (RHS(u + epsilon*v) - RHS(u))/epsilon
    Jacobian_vector = (RHS(u + (epsilon * v), *args) - rhs_u)/epsilon
    
    return Jacobian_vector