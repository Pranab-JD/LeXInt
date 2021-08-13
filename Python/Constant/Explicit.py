"""
Created on Thu Aug 12 19:15:17 2021

@author: Pranab JD

Description: -
    	Contains explicit integrators

"""

################################################################################################

def RK2(u, dt, RHS_func):
	"""
	Parameters
	----------
	u       	: 1D vector u (input)
	dt      	: Step size
	RHS_func	: RHS function

	Returns
	-------
	u_sol   	: 1D vector u (output) after time dt (2nd order)
	2    		: # of RHS evaluations

	"""

	k1 = dt * RHS_func(u)
	k2 = dt * RHS_func(u + k1)

	## Solution
	u_sol = u + 1./2.*(k1 + k2)

	return u_sol, 2


################################################################################################

def RK4(u, dt, RHS_func):
	"""
	Parameters
	----------
	u       	: 1D vector u (input)
	dt      	: Step size
	RHS_func	: RHS function

	Returns
	-------
	u_sol   	: 1D vector u (output) after time dt (2nd order)
	4    		: # of RHS evaluations

	"""

	k1 = dt * RHS_func(u)
	k2 = dt * RHS_func(u + k1/2)
	k3 = dt * RHS_func(u + k2/2)
	k4 = dt * RHS_func(u + k3)

	## Solution
	u_rk4 = u + 1./6.*(k1 + 2*k2 + 2*k3 + k4)

	return u_rk4, 4

################################################################################################