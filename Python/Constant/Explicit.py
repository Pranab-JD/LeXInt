"""
Created on Thu Aug 12 19:15:17 2021

@author: Pranab JD

Description: -
        Contains explicit integrators

"""

################################################################################################

def RK2(u, dt, *A):
	"""
	Parameters
	----------
	u       : 1D vector u (input)
	dt      : Step size
	*A		: N x N matrix A, power to which u is raised

	Returns
	-------
	u_rk2   : 1D vector u (output) after time dt (2nd order)
	2, 4    : # of matrix-vector products

	"""

	if len(A) == 2:

		# A[0] - matrix, A[1] - index of 'u'
		A = A[0]; m = A[1] 

		k1 = dt * (A.dot(u**m))
		k2 = dt * (A.dot((u + k1)**m))

		## Solution
		u_rk2 = u + 1./2. * (k1 + k2)

		return u_rk2, 2

	elif len(A) == 4:

		# A[0] - matrix 1, A[1] - index of 'u1', A[0] - matrix 2, A[1] - index of 'u2'
		A1 = A[0]; m1 = A[1]; A2 = A[2]; m2 = A[3]

		k1 = dt * (A1.dot(u**m1) + A2.dot(u**m2))
		k2 = dt * (A1.dot((u + k1)**m1) + A2.dot((u + k1)**m2))

		## Solution
		u_rk2 = u + 1./2. * (k1 + k2)

		return u_rk2, 4
		
	else:
		print("Error!!! Check number of input matrices!!")


################################################################################################

def RK4(u, dt, *A):
	"""
	Parameters
	----------
	u       : 1D vector u (input)
	dt      : Step size
	*A		: N x N matrix A, power to which u is raised

	Returns
	-------
	u_rk4   : 1D vector u (output) after time dt (4th order)
	4, 8    : # of matrix-vector products

	"""

	if len(A) == 2:

		# A[0] - matrix, A[1] - index of 'u'
		A = A[0]; m = A[1] 

		k1 = dt * (A.dot(u**m))
		k2 = dt * (A.dot((u + k1/2)**m))
		k3 = dt * (A.dot((u + k2/2)**m))
		k4 = dt * (A.dot((u + k3)**m))

		## Solution
		u_rk4 = u + 1./6.*(k1 + 2*k2 + 2*k3 + k4)

		return u_rk4, 4

	elif len(A) == 4:

		# A[0] - matrix 1, A[1] - index of 'u1', A[0] - matrix 2, A[1] - index of 'u2'
		A1 = A[0]; m1 = A[1]; A2 = A[2]; m2 = A[3]

		k1 = dt * (A1.dot(u**m1) + A2.dot(u**m2))
		k2 = dt * (A1.dot((u + k1/2)**m1) + A2.dot((u + k1/2)**m2))
		k3 = dt * (A1.dot((u + k2/2)**m1) + A2.dot((u + k1/2)**m2))
		k4 = dt * (A1.dot((u + k3)**m1) + A2.dot((u + k1/2)**m2))

		## Solution
		u_rk4 = u + 1./6.*(k1 + 2*k2 + 2*k3 + k4)

		return u_rk4, 8
		
	else:
		print("Error!!! Check number of input matrices!!")

################################################################################################