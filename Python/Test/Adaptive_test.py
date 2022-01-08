"""
Created on Sat Aug 14 11:31:19 2021

@author: Pranab JD

Description: Order of convergence for different schemes
"""

import os
import sys
import shutil
import numpy as np
# import matplotlib.pyplot as plt
from scipy.sparse import csr_matrix

sys.path.insert(1, "../Adaptive/Embedded_Explicit/")
from Embedded_explicit import *

sys.path.insert(1, "../Adaptive/")
import Eigenvalues

sys.path.insert(1, "../Adaptive/EXPRB/")
from EXPRB import *

sys.path.insert(1, "../Adaptive/EPIRK/")
from EPIRK import *

### ------------------------------------------------------ ###

### Initialize_matrices
N = 256                 # Number of points along X
xmin = 0                # Min value of X
xmax = 1                # Max value of X
eta = 1                 # Peclet number
dx = (xmax - xmin)/N    # Grid spacing

## Periodic boundaries
X = np.linspace(xmin, xmax, N, endpoint = False)

### Parameters
adv_cfl = dx/eta
dif_cfl = dx**2/2
dt_cfl = min(adv_cfl, dif_cfl)
tmax = 5e-2
R = eta/dx
F = 1/dx**2     

### Set up matrices
A_adv = np.zeros((N, N))
A_dif = np.zeros((N, N))

for ij in range(N):
    
    ## 3rd order upwind 
    A_adv[ij, int(ij + 2) % N] = -1/6
    A_adv[ij, int(ij + 1) % N] =  6/6
    A_adv[ij, ij % N]          = -3/6
    A_adv[ij, int(ij - 1) % N] = -2/6

    ## 2nd order centered difference
    A_dif[ij, int(ij + 1) % N] =  1
    A_dif[ij, ij % N]          = -2
    A_dif[ij, int(ij - 1) % N] =  1

A_adv = csr_matrix(R * A_adv)
A_dif = csr_matrix(F * A_dif)
    
### ------------------------------------------------------ ###
    
def Viscous_Burgers():
    
    ### Initial conditions
    sigma = 0.02
    x_0 = 0.9
    np.seterr(divide = 'ignore')
    u0 = 1 + (np.exp(1 - (1/(1 - (2 * X - 1)**2)))) + 1./2. * np.exp(-(X - x_0)**2/(2 * sigma**2))
    u = u0.copy()

    ### Spectrum of the matrices
    eigen_min_dif = 0.0 
    eigen_max_dif, eigen_imag_dif = Eigenvalues.Gershgorin(A_dif)      # Max real, imag eigen value
    
    ### Scaling and shifting factors
    c = 0.5 * (eigen_max_dif + eigen_min_dif)
    Gamma = 0.25 * (eigen_min_dif - eigen_max_dif)
    
    return u, c, Gamma
    
def RHS_func(u):
    
    ### Viscous Burgers' Equation
    flux_u = A_dif.dot(u) + (0.5 * A_adv.dot(u**2))
    
    return flux_u
    
def solve():
    
    ### Parameters
    dt = 0.9 * dt_cfl
    
    time = 0                                            # Time
    counter = 0                                         # Counter for # of time steps
    count_mv = 0                                        # Counter for matrix-vector products
        
    dt_history = []                                     # Array - dt used
    time_arr = []                                       # Array - time elapsed after each time step
    
    Method_order = 3                                    # Order of the time integrator (error estimator)
    tol = 1e-8
    emax = '{:5.1e}'.format(tol)
    
    ### Create required files/directories
    path = os.path.expanduser("./Test_data/Adaptive/eta_1/EXPRB43/tol_" + str(emax))
    if os.path.exists(path):
        shutil.rmtree(path)                     # remove previous directory with same name
    os.makedirs(path, 0o777)                    # create directory with access rights

    ### Solve viscous Burgers' equation
    u, c, Gamma = Viscous_Burgers()
    
    while time < tmax:
        
        if time + dt > tmax:
            dt = tmax - time
            
        # u_low, u_high, rhs_calls_1 = RKF45(u, dt, RHS_func)
        u_low, u_high, rhs_calls_1 = EXPRB43(u, dt, RHS_func, c, Gamma, tol, 0)
        
        ### Error
        error = np.mean(abs(u_low - u_high))
        
        if error > tol:
            
            new_dt = dt * (tol/error)**(1/(Method_order + 1))
            dt = 0.8 * new_dt                       # Safety factor
            
            # u_low, u_high, rhs_calls_2 = RKF45(u, dt, RHS_func)
            u_low, u_high, rhs_calls_2 = EXPRB43(u, dt, RHS_func, c, Gamma, tol, 0)
        
            error = np.mean(abs(u_low - u_high))
            
        else:
            rhs_calls_2 = 0
        
        ### Update u and time
        u = u_high.copy()
        time = time + dt
        
        dt_history.append(dt)
        time_arr.append(time)
        count_mv = count_mv + rhs_calls_1 + rhs_calls_2
        
        ### dt for next time step
        new_dt = dt * (tol/error)**(1/(Method_order + 1))
        dt = 0.8 * new_dt                       # Safety factor
        
    ## Write final data to files
    file_final_sol = open(path + "/Final_data_sol.txt", 'w+')
    file_final_sol.write(' '.join(map(str, u_high)) % u_high)
    file_final_sol.close()
    
    ### Write simulation results to file
    file_res = open(path + '/Results.txt', 'w+')
    file_res.write('Number of matrix-vector products = %d' % count_mv + '\n' + '\n')
    file_res.write(' '.join(map(str, dt_history)) % dt_history + '\n' + '\n')
    file_res.write(' '.join(map(str, time_arr)) % time_arr)
    file_res.close()

                
### Call the function
solve()