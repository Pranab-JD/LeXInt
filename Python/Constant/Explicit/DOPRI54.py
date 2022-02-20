def DOPRI54(u, dt, RHS_func, *args):
	"""
	Parameters
	----------
	u           : 1D vector u (input)
	dt          : Step size
	RHS_func	: RHS function

	Returns
	-------
	u_dopri4   	: 1D vector u (output) after time dt (4th order)
	u_dopri5   	: 1D vector u (output) after time dt (5th order)
	7    		: # of RHS evaluations

	"""

	k1 = dt * RHS_func(u)
	k2 = dt * RHS_func(u + k1/5)
	k3 = dt * RHS_func(u + 3./40.*k1 + 9./40.*k2)
	k4 = dt * RHS_func(u + 44./45.*k1 - 56./15.*k2 + 32./9.*k3)
	k5 = dt * RHS_func(u + 19372./6561.*k1 - 25360./2187.*k2 + 64448./6561.*k3 - 212./729.*k4)
	k6 = dt * RHS_func(u + 9017./3168.*k1 - 355./33.*k2 + 46732./5247.*k3 + 49./176.*k4 - 5103./18656.*k5)
	k7 = dt * RHS_func(u + 35./384.*k1 + 500./1113.*k3 + 125./192.*k4 - 2187./6784.*k5 + 11./84.*k6)

	### Solution
	## u_dopri4 = u + (5179./57600.*k1 + 7571./16695.*k3 + 393./640.*k4 - 92097./339200.*k5 + 187./2100.*k6 + 1./40.*k7)
	u_dopri5 = u + (35./384.*k1 + 500./1113.*k3 + 125./192.*k4 - 2187./6784.*k5 + 11./84.*k6)

	return u_dopri5, 7