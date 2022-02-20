from scipy.sparse import identity
from GMRES import *

def Crank_Nicolson(u, dt, A, tol):
	"""
	Parameters
	----------
	u           : 1D vector u (input)
	dt          : Step size
	A        	: N x N matrix
	tol 		: Desired accuracy of the solution

	Returns
	-------
	u_cn   	    : 1D vector u (output) after time dt (2nd order)
	iters 		: # of RHS evaluations

	"""

	rhs = u + (0.5 * (A.dot(u)) * dt)
	u_cn, iters = GMRES(identity(A.shape[0])-(0.5*dt*A), rhs, u, tol)

	return u_cn, iters
