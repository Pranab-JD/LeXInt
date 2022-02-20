def RK2(u, dt, RHS_func):
	"""
	Parameters
	----------
	u       	: 1D vector u (input)
	dt      	: Step size
	RHS_func	: RHS function

	Returns
	-------
	u_rk2   	: 1D vector u (output) after time dt (2nd order)
	2    		: # of RHS evaluations

	"""

	k1 = dt * RHS_func(u)
	k2 = dt * RHS_func(u + k1)

	## Solution
	u_rk2 = u + 1./2.*(k1 + k2)

	return u_rk2, 2