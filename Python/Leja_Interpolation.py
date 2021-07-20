"""
Created on Thu Jul 23 15:54:52 2020

@author: Pranab JD

Description: -
        Interpolation at Leja points
"""

import numpy as np

################################################################################################

### Phi Functions ('z' is assumed to be a double)

def phi_1(z):
    if abs(z) <= 1e-6:
        return 1 + z * (1./2. + z * (1./6. + z * (1./24. + 1./120. * z)))
    else:
        return (np.exp(z) - 1)/z

def phi_2(z):
    if abs(z) <= 1e-5:
        return 1./2. + z * (1./6. + z * (1./24. + z * (1./120. + 1./720. * z)));
    else:
        return (np.exp(z) - z - 1)/z**2

def phi_3(z):
    if abs(z) <= 1e-5:
        return 1./6. + z * (1./24. + z * (1./120. + z * (1./720. + 1./5040. * z)));
    else:
        return (np.exp(z) - z**2/2 - z - 1)/z**3

def phi_4(z):
    if abs(z) <= 1e-4:
        return 1./24. + z * (1./120. + z * (1./720. + z * (1./5040. + 1./40320. * z)));
    else:
        return (np.exp(z) - z**3/6 - z**2/2 - z - 1)/z**4

################################################################################################

def Leja_Points():
    """
    Load Leja points from binary file

    """
    dt = np.dtype("f8")
    return np.fromfile('real_leja_d.bin', dtype = dt)

def Divided_Difference(X, func):
    """
    Parameters
    ----------
    X       : Leja points
    func    : func(X)

    Returns
    -------
    div_diff : Polynomial coefficients

    """

    div_diff = func(X)

    for ii in range(1, int(len(X))):
        for jj in range(ii):

            div_diff[ii] = (div_diff[ii] - div_diff[jj])/(X[ii] - X[jj])

    return div_diff

def Gershgorin(A):
    """
    Parameters
    ----------
    A        : N x N matrix

    Returns
    -------
    eig_real : Largest real eigen value (negative magnitude)
    eig_imag : Largest imaginary eigen value

    """

    A_Herm = (A + A.T.conj())/2
    A_SkewHerm = (A - A.T.conj())/2

    row_sum_real = np.zeros(np.shape(A)[0])
    row_sum_imag = np.zeros(np.shape(A)[0])

    for ii in range(len(row_sum_real)):
        row_sum_real[ii] = np.sum(abs(A_Herm[ii, :]))
        row_sum_imag[ii] = np.sum(abs(A_SkewHerm[ii, :]))

    eig_real = - np.max(row_sum_real)       # Has to be NEGATIVE
    eig_imag = np.max(row_sum_imag)

    return eig_real, eig_imag

def Power_iteration(A, u, m):
    """
    Parameters
    ----------
    A             : N x N matrix A
    u             : Vector u
    m             : Index of u (u^m)

    Returns
    -------
    eigen_val[ii] : Largest eigen value (within 10% accuracy)
    (ii + 1) * 2  : Number of matrix-vector products
                    (ii + 1): No. of iterations (starts from 0)

    """

    A_Herm = (A + A.T.conj())/2                     # Hermitian matrix has real eigen values
    A_SkewHerm = (A - A.T.conj())/2                 # Hermitian matrix has imag eigen values

    def eigens(A):

        tol = 0.1                                   # 10% tolerance
        niters = 1000                               # Max. # of iterations
        epsilon = 1e-7
        eigen_val = np.zeros(niters)                # Array of max. eigen value at each iteration
        vector = np.zeros(len(u)); vector[0] = 1    # Initial estimate of eigen vector

        for ii in range(niters):

            eigen_vector = (A.dot((u + (epsilon * vector))**m) - (A.dot(u**m)))/epsilon

            eigen_val[ii] = np.max(abs(eigen_vector))

            ## Convergence is to be checked for eigen values, not eigen vectors
            ## since eigen values converge faster than eigen vectors
            if (abs(eigen_val[ii] - eigen_val[ii - 1]) <= tol * eigen_val[ii]):
                break
            else:
                eigen_vector = eigen_vector/eigen_val[ii]       # Normalize eigen vector to eigen value
                vector = eigen_vector.copy()                    # New estimate of eigen vector

        return eigen_val[ii], ii                                # Last eigen value, # of matrix-vector products

    eigen_real, i1 = eigens(A_Herm)
    eigen_imag, i2 = eigens(A_SkewHerm)


    ## Real eigen value has to be negative
    eigen_real = - eigen_real

    return eigen_real, eigen_imag, ((2 * (i1 + 1)) + (2 * (i2 + 1)))

################################################################################################

def real_Leja_exp(A, u, dt, c, Gamma):
    """
    Parameters
    ----------
    A               : N x N matrix
    u               : Vector u
    m               : Index of u (u^m), = 1 for linear equations
    dt              : Step size
    Leja_X          : Leja points
    c          : Shifting factor
    Gamma      : Scaling factor

    Returns
    ----------
    np.real(u) : Polynomial interpolation of u
                      at real Leja points
    ii              : No. of matrix-vector products

    """

    def func(xx):
        return np.exp(dt * (c + Gamma*xx))

    Leja_X = Leja_Points()                                 # Leja Points
    coeffs = Divided_Difference(Leja_X, func)              # Polynomial Coefficients

    ### ------------------------------------------------------------------- ###

    ## a_0 term
    poly = u.copy()
    poly = coeffs[0] * poly

    ## a_1, a_2 .... a_n terms
    max_Leja_pts = 100                                     # Max number of Leja points
    y = u.copy()                                           # x values of the polynomial
    poly_vals = np.zeros(max_Leja_pts)                     # Array for error incurred
    poly_tol = 1e-4                                        # Accuracy of the polynmomial so formed
    y_val = np.zeros((max_Leja_pts, len(u)))               # Stores x values till polynomial converges
    scale_fact = 1/Gamma                                   # Re-scaling factor

    ### ------------------------------------------------------------------- ###

    ## Iterate until convergence is reached
    for ii in range(1, max_Leja_pts):

        shift_fact = -c * scale_fact - Leja_X[ii - 1]      # Re-shifting factor

        ## function: function to be multiplied to the matrix exponential of the Jacobian
        function = y.copy()
        Jacobian_function = A.dot(function)

        y = y * shift_fact
        y = y + scale_fact * Jacobian_function

        ## Error incurred
        poly_vals[ii] = (sum(abs(y)**2)/len(y))**0.5 * abs(coeffs[ii])

        ## If new number (next order) to be added < tol, ignore it
        if  poly_vals[ii] < poly_tol:
            # print('No. of Leja points used (real phi) = ', ii)
            # print('----------Tolerance reached---------------')
            y_val[ii, :] = coeffs[ii] * y
            break

        ## To stop diverging
        elif poly_vals[ii] > 1e13:
            return 20 * u, ii

        else:
            y_val[ii, :] = coeffs[ii] * y

        if ii >= max_Leja_pts - 1:
            print("Error!! No. of Leja points not sufficient!!")
            return 20 * u, ii

    ### ------------------------------------------------------------------- ###

    ### Choose polynomial terms up to the smallest term, ignore the rest
    if np.argmin(poly_vals[np.nonzero(poly_vals)]) + 1 == 0:               # Tolerance reached
        min_poly_val_x = np.argmin(poly_vals[np.nonzero(poly_vals)])

    else:
        min_poly_val_x = np.argmin(poly_vals[np.nonzero(poly_vals)]) + 1   # Starts to diverge

    ### Form the polynomial
    for jj in range(1, min_poly_val_x + 1):
        poly = poly + y_val[jj, :]

    ## Solution
    u_sol = poly.copy()

    return u_sol, ii + 1

################################################################################################

def real_Leja_phi(u, nonlin_matrix_vector, dt, c, Gamma, phi_func, *A):
    """
    Parameters
    ----------
    u                       : Vector u
    nonlin_matrix_vector    : function to be multiplied to phi function
    dt                      : self.dt
    c                       : Shifting factor
    Gamma                   : Scaling factor
    phi_func                : phi function
    *A						: N x N matrix A, power to which u is raised

    Returns
    ----------
    np.real(u_real)         : Polynomial interpolation of
                              nonlinear part using the phi
                              function at imaginary Leja points
    ii * len(A)             : No. of matrix-vector products

    """

    def func(xx):

        np.seterr(divide = 'ignore', invalid = 'ignore')

        zz = (dt * (c + Gamma*xx))
        var = phi_func(zz)

        if phi_func != phi_1 or phi_func != phi_2 or phi_func != phi_3 or phi_func != phi_4:
            print('Error: Phi function not defined!!')

        return var

    ### ------------------------------------------------------------------- ###

    Leja_X = Leja_Points()                                  # Leja Points
    coeffs = Divided_Difference(Leja_X, func)               # Polynomial Coefficients

    ### ------------------------------------------------------------------- ###

    ## a_0 term
    poly = nonlin_matrix_vector.copy()
    poly = coeffs[0] * poly

    ### ------------------------------------------------------------------- ###

    ## a_1, a_2 .... a_n terms
    y = nonlin_matrix_vector.copy()                         # Max number of Leja points
    max_Leja_pts = len(coeffs)                              # x values of the polynomial
    poly_vals = np.zeros(max_Leja_pts)                      # Array for error incurred
    poly_tol = 1e-4                                         # Accuracy of the polynmomial so formed
    epsilon = 1e-7                                          # Stores x values till polynomial converges
    y_val = np.zeros((max_Leja_pts, len(u)))                # Re-scaling factor
    scale_fact = 1/Gamma                                    # Re-scaling factor

    ### ------------------------------------------------------------------- ###

    ## Iterate until convergence is reached
    for ii in range(1, max_Leja_pts):

        shift_fact = -c * scale_fact - Leja_X[ii - 1]       # Re-shifting factor

        ## function: function to be multiplied to the phi function applied to Jacobian
        function = y.copy()

        if len(A) == 2:
            A_nl = A[0]; m_nl = A[1]
            Jacobian_function = (A_nl.dot((u + (epsilon * function))**m_nl) - A_nl.dot(u**m_nl))/epsilon

        elif len(A) == 3:
            A_nl = A[0]; m_nl = A[1]; A_lin = A[2]
            Jacobian_function = (A_nl.dot((u + (epsilon * function))**m_nl) - A_nl.dot(u**m_nl))/epsilon + A_lin.dot(function)

        else:
            print("Error! Check number of matrices!!")

        y = y * shift_fact
        y = y + scale_fact * Jacobian_function

        ## Error incurred
        poly_vals[ii] = (sum(abs(y)**2)/len(y))**0.5 * abs(coeffs[ii])

        ## If new number (next order) to be added < tol, ignore it
        if  poly_vals[ii] < poly_tol:
            # print('No. of Leja points used (real phi) = ', ii)
            # print('----------Tolerance reached---------------')
            y_val[ii, :] = coeffs[ii] * y
            break

        ## To stop diverging
        elif poly_vals[ii] > 1e13:
            return 20 * nonlin_matrix_vector, ii * len(A)

        else:
            y_val[ii, :] = coeffs[ii] * y

        if ii >= max_Leja_pts - 1:
            return 20 * nonlin_matrix_vector, ii * len(A)

    ### ------------------------------------------------------------------- ###

    ### Choose polynomial terms up to the smallest term, ignore the rest
    if np.argmin(poly_vals[np.nonzero(poly_vals)]) + 1 == 0:               # Tolerance reached
        min_poly_val_x = np.argmin(poly_vals[np.nonzero(poly_vals)])

    else:
        min_poly_val_x = np.argmin(poly_vals[np.nonzero(poly_vals)]) + 1   # Starts to diverge

    ### Form the polynomial
    for jj in range(1, min_poly_val_x + 1):
        poly = poly + y_val[jj, :]

    ## Solution
    nl_mat_vec_real = poly.copy()

    return nl_mat_vec_real, ii * len(A)

################################################################################################

def imag_Leja_phi(u, nonlin_matrix_vector, dt, c, Gamma, phi_func, *A):
    """
    Parameters
    ----------
    u                       : Vector u
    nonlin_matrix_vector    : function to be multiplied to phi function
    dt                      : self.dt
    c                       : Shifting factor
    Gamma                   : Scaling factor
    phi_func                : phi function
    *A						: N x N matrix A, power to which u is raised

    Returns
    ----------
    np.real(u_imag)         : Polynomial interpolation of
                              nonlinear part using the phi
                              function at imaginary Leja points
    ii * len(A)             : No. of matrix-vector products

    """

    def func(xx):

        np.seterr(divide = 'ignore', invalid = 'ignore')

        zz = (1j * dt * (c + Gamma*xx))
        var = phi_func(zz)

        if phi_func != phi_1 or phi_func != phi_2 or phi_func != phi_3 or phi_func != phi_4:
            print('Error: Phi function not defined!!')

        return var

    ### ------------------------------------------------------------------- ###

    Leja_X = Leja_Points()                                  # Leja Points
    coeffs = Divided_Difference(Leja_X, func)               # Polynomial Coefficients

    ### ------------------------------------------------------------------- ###

    ## a_0 term
    poly = nonlin_matrix_vector.copy()
    poly = coeffs[0] * poly

    ### ------------------------------------------------------------------- ###

    ## a_1, a_2 .... a_n terms
    y = nonlin_matrix_vector.copy()                         # Max number of Leja points
    max_Leja_pts = len(coeffs)                              # x values of the polynomial
    poly_vals = np.zeros(max_Leja_pts)                      # Array for error incurred
    poly_tol = 1e-4                                         # Accuracy of the polynmomial so formed
    epsilon = 1e-7                                          # Stores x values till polynomial converges
    y_val = np.zeros((max_Leja_pts, len(u)))                # Re-scaling factor
    scale_fact = 1/Gamma                                    # Re-scaling factor

    ### ------------------------------------------------------------------- ###

    ## Iterate until convergence is reached
    for ii in range(1, max_Leja_pts):

        shift_fact = -c * scale_fact - Leja_X[ii - 1]       # Re-shifting factor

        ## function: function to be multiplied to the phi function applied to Jacobian
        function = y.copy()

        if len(A) == 2:
            A_nl = A[0]; m_nl = A[1]
            Jacobian_function = (A_nl.dot((u + (epsilon * function))**m_nl) - A_nl.dot(u**m_nl))/epsilon

        elif len(A) == 3:
            A_nl = A[0]; m_nl = A[1]; A_lin = A[2]
            Jacobian_function = (A_nl.dot((u + (epsilon * function))**m_nl) - A_nl.dot(u**m_nl))/epsilon + A_lin.dot(function)

        else:
            print("Error! Check number of matrices!!")

        y = y * shift_fact
        y = y + scale_fact * Jacobian_function * (-1j)

        ## Error incurred
        poly_vals[ii] = (sum(abs(y)**2)/len(y))**0.5 * abs(coeffs[ii])

        ## If new number (next order) to be added < tol, ignore it
        if  poly_vals[ii] < poly_tol:
            # print('No. of Leja points used (real phi) = ', ii)
            # print('----------Tolerance reached---------------')
            y_val[ii, :] = coeffs[ii] * y
            break

        ## To stop diverging
        elif poly_vals[ii] > 1e13:
            return 20 * nonlin_matrix_vector, ii * len(A)

        else:
            y_val[ii, :] = coeffs[ii] * y

        if ii >= max_Leja_pts - 1:
            return 20 * nonlin_matrix_vector, ii * len(A)

    ### ------------------------------------------------------------------- ###

    ### Choose polynomial terms up to the smallest term, ignore the rest
    if np.argmin(poly_vals[np.nonzero(poly_vals)]) + 1 == 0:               # Tolerance reached
        min_poly_val_x = np.argmin(poly_vals[np.nonzero(poly_vals)])

    else:
        min_poly_val_x = np.argmin(poly_vals[np.nonzero(poly_vals)]) + 1   # Starts to diverge

    ### Form the polynomial
    for jj in range(1, min_poly_val_x + 1):
        poly = poly + y_val[jj, :]

    ## Solution
    nl_mat_vec_imag = poly.copy()

    return np.real(nl_mat_vec_imag), ii * len(A)

################################################################################################
