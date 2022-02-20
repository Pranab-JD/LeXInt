def RKF45(u, dt, RHS_func, *args):
	"""
	Parameters
	----------
	u           : 1D vector u (input)
	dt          : Step size
	RHS_func	: RHS function

	Returns
	-------
	u_rkf5   	: 1D vector u (output) after time dt (5th order)
	6    		: # of RHS evaluations

	"""

	k1 = dt * RHS_func(u)
	k2 = dt * RHS_func(u + k1/4)
	k3 = dt * RHS_func(u + 3./32.*k1 + 9./32.*k2)
	k4 = dt * RHS_func(u + 1932./2197.*k1 - 7200./2197.*k2 + 7296./2197.*k3)
	k5 = dt * RHS_func(u + 439./216.*k1 - 8*k2 + 3680./513.*k3 - 845./4104.*k4)
	k6 = dt * RHS_func(u - 8./27.*k1 + 2*k2 - 3544./2565.*k3 + 1859./4104.*k4 - 11./40.*k5)

	### Solution
	## u_rkf4 = u + (25./216.*k1 + 1408./2565.*k3 + 2197./4101.*k4 - 1./5.*k5)
	u_rkf5 = u + (16./135.*k1 + 6656./12825.*k3 + 28561./56430.*k4 - 9./50.*k5 + 2./55.*k6)

	return u_rkf5, 6
