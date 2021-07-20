"""
Created on Wed Aug 19 16:17:29 2020

@author: Pranab JD

Description: -
        Contains exponential integrators for 2-matrix equations
        (du/dt = A_nl.u^m_nl + A_lin.u)

        The exponential integrators in this code have been
        optimzed for a combination of 1 nonlinear operator
        and 1 linear operator.

        For embedded schemes, the # of matrix-vector products
        for the higher-order solution takes into account the 
        # of matrix-vector operations for the lower-order
        solution.

"""

from Leja_Interpolation import *

################################################################################################

def ETD(A_nl, m_nl, A_lin, u, dt, c, Gamma, Real_Imag_Leja):
    """
    Parameters
    ----------
    A_nl            : Nonlinear operator matrix (A)
    m_nl            : Index of u (u^m_nl)
    A_lin           : Linear operator matrix (A)
    u               : 1D vector u (Input)
    dt              : Step size
    c, Gamma        : Shifting and scaling factors
    Real_Imag_Leja  : 0 - Real, 1 - Imag

    Returns
    -------
    u_etd           : 1D vector u (output) after time dt (2nd order)
    mat_vec_num     : # of matrix-vector products

    """
    
    ## Use either Real Leja or imaginary Leja
    if Real_Imag_Leja == 0:
        Leja_phi = real_Leja_phi
    elif Real_Imag_Leja == 1:
        Leja_phi = imag_Leja_phi
    else:
        print('Error in choosing Real/Imag Leja!!')

    ############## --------------------- ##############

    ### Linear operator
    f_u_lin = A_lin.dot(u)

    ### Nonlinear operator
    f_u_nl = A_nl.dot(u**m_nl)

    u_sol, mat_vec_num = Leja_phi(u, (f_u_lin + f_u_nl), dt, c, Gamma, phi_1, A_nl, m_nl, A_lin)

    ### ETD Solution
    u_etd = u + (u_sol * dt)

    return u_etd, mat_vec_num

##############################################################################

def EXPRB42(A_nl, m_nl, A_lin, u, dt, c, Gamma, Real_Imag_Leja):
    """
    Parameters
    ----------
    A_nl            : Nonlinear operator matrix (A)
    m_nl            : Index of u (u^m_nl)
    A_lin           : Linear operator matrix (A)
    u               : 1D vector u (Input)
    dt              : Step size
    c, Gamma        : Shifting and scaling factors
    Real_Imag_Leja  : 0 - Real, 1 - Imag

    Returns
    -------
    u_exprb42       : 1D vector u (output) after time dt (4th order)
    mat_vec_num     : # of matrix-vector products

    """

    ## Use either Real Leja or imaginary Leja
    if Real_Imag_Leja == 0:
        Leja_phi = real_Leja_phi
    elif Real_Imag_Leja == 1:
        Leja_phi = imag_Leja_phi
    else:
        print('Error in choosing Real/Imag Leja!!')

    epsilon = 1e-7

    ############## --------------------- ##############

    ### Linear operator
    f_u_lin = A_lin.dot(u)

    ### Nonlinear operator
    f_u_nl = A_nl.dot(u**m_nl)

    ############## --------------------- ##############

    ### Internal stage 1
    a_n_f, its_a = Leja_phi(u, (f_u_lin + f_u_nl), 3*dt/4, c, Gamma, phi_1, A_nl, m_nl, A_lin)
    a_n = u + (a_n_f * 3*dt/4)

    ############## --------------------- ##############

    ### J(u) * u
    Linear_u = (A_nl.dot((u + (epsilon * u))**m_nl) - f_u_nl)/epsilon

    ### F(u) = f(u) - (J(u) * u)
    Nonlin_u = f_u_nl - Linear_u

    ### J(u) * a
    Linear_a = (A_nl.dot((u + (epsilon * a_n))**m_nl) - f_u_nl)/epsilon

    ### F(a) = f(a) - (J(u) * a)
    Nonlin_a = A_nl.dot(a_n**m_nl) - Linear_a

    ############## --------------------- ##############

    u_1, its_1 = Leja_phi(u, (f_u_lin + f_u_nl), dt, c, Gamma, phi_1, A_nl, m_nl, A_lin)
    u_nl_3, its_3 = Leja_phi(u, (Nonlin_a - Nonlin_u), dt, c, Gamma, phi_3, A_nl, m_nl)

    ### 4th order solution
    u_exprb42 = u + (u_1 * dt) + (u_nl_3 * 32*dt/9)

    mat_vec_num = 8 + its_a + its_1 + its_3

    return u_exprb42, mat_vec_num

##############################################################################

def EXPRB32(A_nl, m_nl, A_lin, u, dt, c, Gamma, Real_Imag_Leja):
    """
    Parameters
    ----------
    A_nl            : Nonlinear operator matrix (A)
    m_nl            : Index of u (u^m_nl)
    A_lin           : Linear operator matrix (A)
    u               : 1D vector u (Input)
    dt              : Step size
    c, Gamma        : Shifting and scaling factors
    Real_Imag_Leja  : 0 - Real, 1 - Imag

    Returns
    -------
    u_exprb2        : 1D vector u (output) after time dt (2nd order)
    mat_vec_num_2   : # of matrix-vector products for 2nd order solution
    u_exprb3        : 1D vector u (output) after time dt (3rd order)
    mat_vec_num_3   : # of matrix-vector products for 3rd order solution

    """

    ## Use either Real Leja or imaginary Leja
    if Real_Imag_Leja == 0:
        Leja_phi = real_Leja_phi
    elif Real_Imag_Leja == 1:
        Leja_phi = imag_Leja_phi
    else:
        print('Error in choosing Real/Imag Leja!!')

    epsilon = 1e-7

    ############## --------------------- ##############

    ### Linear operator
    f_u_lin = A_lin.dot(u)

    ### Nonlinear operator
    f_u_nl = A_nl.dot(u**m_nl)

    ############## --------------------- ##############

    ### Internal stage 1 (2nd order solution)
    a_n_f, its_a = Leja_phi(u, (f_u_lin + f_u_nl), dt, c, Gamma, phi_1, A_nl, m_nl, A_lin)
    a_n = u + (a_n_f * dt)

    u_exprb2 = a_n; mat_vec_num_2 = 2 + its_a

    ############## --------------------- ##############

    ### J(u) * u
    Linear_u = (A_nl.dot((u + (epsilon * u))**m_nl) - f_u_nl)/epsilon

    ### F(u) = f(u) - (J(u) * u)
    Nonlin_u = f_u_nl - Linear_u

    ### J(u) * a
    Linear_a = (A_nl.dot((u + (epsilon * a_n))**m_nl) - f_u_nl)/epsilon

    ### F(a) = f(a) - (J(u) * a)
    Nonlin_a = A_nl.dot(a_n**m_nl) - Linear_a

    ############## --------------------- ##############

    ### 3rd order solution
    u_3, its_3 = Leja_phi(u, 2*(Nonlin_a - Nonlin_u), dt, c, Gamma, phi_3, A_nl, m_nl)
    u_exprb3 = u_exprb2 + (u_3 * dt)

    mat_vec_num_3 = 5 + its_a + its_3

    return u_exprb2, mat_vec_num_2, u_exprb3, mat_vec_num_3

##############################################################################

def EXPRB43(A_nl, m_nl, A_lin, u, dt, c, Gamma, Real_Imag_Leja):
    """
    Parameters
    ----------
    A_nl            : Nonlinear operator matrix (A)
    m_nl            : Index of u (u^m_nl)
    A_lin           : Linear operator matrix (A)
    u               : 1D vector u (Input)
    dt              : Step size
    c, Gamma        : Shifting and scaling factors
    Real_Imag_Leja  : 0 - Real, 1 - Imag

    Returns
    -------
    u_exprb3        : 1D vector u (output) after time dt (3rd order)
    mat_vec_num_3   : # of matrix-vector products for 3rd order solution
    u_exprb4        : 1D vector u (output) after time dt (4th order)
    mat_vec_num_4   : # of matrix-vector products for 4th order solution

    """

    ## Use either Real Leja or imaginary Leja
    if Real_Imag_Leja == 0:
        Leja_phi = real_Leja_phi
    elif Real_Imag_Leja == 1:
        Leja_phi = imag_Leja_phi
    else:
        print('Error in choosing Real/Imag Leja!!')

    epsilon = 1e-7

    ############## --------------------- ##############

    ### Linear operator
    f_u_lin = A_lin.dot(u)

    ### Nonlinear operator
    f_u_nl = A_nl.dot(u**m_nl)

    ############## --------------------- ##############

    ### Internal stage 1
    a_n_f, its_a = Leja_phi(u, (f_u_lin + f_u_nl), dt/2, c, Gamma, phi_1, A_nl, m_nl, A_lin)
    a_n = u + a_n_f * dt/2

    ############## --------------------- ##############

    ### J(u) * u
    Linear_u = (A_nl.dot((u + (epsilon * u))**m_nl) - f_u_nl)/epsilon

    ### F(u) = f(u) - (J(u) * u)
    Nonlin_u = f_u_nl - Linear_u

    ### J(u) * a
    Linear_a = (A_nl.dot((u + (epsilon * a_n))**m_nl) - f_u_nl)/epsilon

    ### F(a) = f(a) - (J(u) * a)
    Nonlin_a = A_nl.dot(a_n**m_nl) - Linear_a

    ############# --------------------- ##############

    ### Internal stage 2
    b_n_f, its_b_1 = Leja_phi(u, (f_u_lin + f_u_nl), dt, c, Gamma, phi_1, A_nl, m_nl, A_lin)
    b_n_nl, its_b_2 = Leja_phi(u, (Nonlin_a - Nonlin_u), dt, c, Gamma, phi_1, A_nl, m_nl)

    b_n = u + (b_n_f * dt) + (b_n_nl * dt)

    ############# --------------------- ##############

    ### J(u) * b
    Linear_b = (A_nl.dot((u + (epsilon * b_n))**m_nl) - f_u_nl)/epsilon

    ### F(b) = f(b) - (J(u) * b)
    Nonlin_b = A_nl.dot(b_n**m_nl) - Linear_b

    ############# --------------------- ##############
    
    ### 3rd and 4th order solutions
    u_1 = b_n_f
    u_nl_3, its_3 = Leja_phi(u, (-14*Nonlin_u + 16*Nonlin_a - 2*Nonlin_b), dt, c, Gamma, phi_3, A_nl, m_nl)
    u_nl_4, its_4 = Leja_phi(u, (36*Nonlin_u - 48*Nonlin_a + 12*Nonlin_b), dt, c, Gamma, phi_4, A_nl, m_nl)

    u_exprb3 = u + (u_1 * dt) + (u_nl_3 * dt)
    u_exprb4 = u_exprb3 + (u_nl_4 * dt)

    mat_vec_num_3 = 12 + its_a + its_b_1 + its_b_2 + its_3
    mat_vec_num_4 = 12 + its_a + its_b_1 + its_b_2 + its_3 + its_4

    return u_exprb3, mat_vec_num_3, u_exprb4, mat_vec_num_4

##############################################################################