def Cash_Karp(u, dt, RHS_func, *args):
	"""
	Parameters
	----------
	u           : 1D vector u (input)
	dt          : Step size
	RHS_func	: RHS function

	Returns
	-------
	u_ck5   	: 1D vector u (output) after time dt (5th order)
	5    		: # of RHS evaluations

	"""

	k1 = dt * RHS_func(u)
	k2 = dt * RHS_func(u + k1/5)
	k3 = dt * RHS_func(u + 3./40.*k1 + 9./40.*k2)
	k4 = dt * RHS_func(u + 3./10.*k1 - 9./10.*k2 + 6./5.*k3)
	k5 = dt * RHS_func(u - 11./54.*k1 + 5./2.*k2 - 70./27.*k3 + 35./27.*k4)
	k6 = dt * RHS_func(u + 1631./55296.*k1 + 175./512.*k2 + 575./13824.*k3 + 44275./110592.*k4 + 253./4096.*k5)

	### Solution
	u_ck4 = u + (2825./27648.*k1 + 18575./48384.*k3 + 13525./55296.*k4 + 277./14336.*k5 + 1./4.*k6)
	u_ck5 = u + (37./378.*k1 + 250./621.*k3 + 125./594.*k4 + 512./1771.*k6)

	return u_ck4, u_ck5, 6
