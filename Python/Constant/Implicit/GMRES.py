import scipy.sparse.linalg as spla

class counter:

    def __init__(self):
        self.count = 0

    def increment(self, x):
        self.count = self.count + 1

def GMRES(A, u, x0, tol):
    """
    Parameters
    ----------
    A        		: N x N matrix
    u        		: 1D vector u (input)
    tol      		: Accuracy of the solution

    Returns
    -------
    u_sol       	: Converged solution
    iters.count   	: # of iterations

    """

    ## Initialize object for the counter class
    iters = counter()

    ## Solve using GMRES
    u_sol, convergence = spla.gmres(A, u, x0 = u, tol = tol, callback = iters.increment)

    if convergence > 0:
        print("Fails to converge up to the give tolerance!")
        print("Iterations: ", convergence)

    return u_sol, iters.count
