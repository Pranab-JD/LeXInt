import numpy as np

###? Jacobian_vector = (RHS(u + epsilon*v) - RHS(u - epsilon*v))/(2*epsilon)
def Jacobian(RHS, u, v, *args):
    
    ###* epsilon has to be normalised to RHS(u)
    epsilon = 1e-7 * np.linalg.norm(RHS(u, *args))
    
    ###* J(u) * y = (RHS(u + epsilon*v) - RHS(u - epsilon*v))/(2*epsilon)
    Jac_vec = (RHS(u + (epsilon * v), *args) - RHS(u - (epsilon * v), *args))/(2 * epsilon)
    
    return Jac_vec