def RK43(u, dt, RHS_func, *args):
	"""
	Parameters
	----------
	u           : 1D vector u (input)
	dt          : Step size
	RHS_func	: RHS function

	Returns
	-------
	u_rk3   	: 1D vector u (output) after time dt (3rd order)
	u_rk4   	: 1D vector u (output) after time dt (4th order)
	6    		: # of RHS evaluations

	Note    	: |u_rk3 - u_rk4| gives 3rd order error estimate 

	"""

	k1 = dt * RHS_func(u)
	k2 = dt * RHS_func(u + k1/2)
	k3 = dt * RHS_func(u + k2/2)
	k4 = dt * RHS_func(u + k3)
	k5 = dt * RHS_func(u + 439./216.*k1 - 8*k2 + 3680./513.*k3 - 845./4104.*k4)
	k6 = dt * RHS_func(u - 8./27.*k1 + 2*k2 - 3544./2565.*k3 + 1859./4140.*k4 - 11./40.*k5)

	### Solution
	u_rk3 = u + (1./6.*k1 + 1./3.*k2 + 1./3.*k3 + 1./6.*k4)
	u_rk4 = u + (16./135.*k1 + 6656./12825.*k3 + 28561./56430.*k4 - 9./50.*k5 + 2./55.*k6)

	return u_rk3, u_rk4, 6
