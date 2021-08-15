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

sys.path.insert(1, "../Constant/Explicit/")
import Explicit

sys.path.insert(1, "../Constant/")
import Eigenvalues

sys.path.insert(1, "../Constant/EXPRB/")
import EXPRB

### ------------------------------------------------------ ###

### Initialize_matrces
N = 100                 # Number of points along X
xmin = 0                # Min value of X
xmax = 1                # Max value of X
eta = 50                # Peclet number
dx = (xmax - xmin)/N    # Grid spacing

## Periodic boundaries
X = np.linspace(xmin, xmax, N, endpoint = False)

### Parameters
adv_cfl = dx/eta
dif_cfl = dx**2/2
dt_cfl = min(adv_cfl, dif_cfl)
tmax = 1e-2
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
    
    ### Solve viscous Burgers' eqation
    u, c, Gamma = Viscous_Burgers()
    
    ### Temporal paramerters    
    time_elapsed = 0.0
    dt = 0.09 * dt_cfl
    
    ### Create required files/directories
    path = os.path.expanduser("./Test_data/Rosenbrock_Euler/Data_" + str(float(dt/dt_cfl)) + "_dt_cfl")
    if os.path.exists(path):
        shutil.rmtree(path)                     # remove previous directory with same name
    os.makedirs(path, 0o777)                    # create directory with access rights
    
    while time_elapsed < tmax:
        
        if time_elapsed + dt > tmax:
            dt = tmax - time_elapsed
            
        # u_new, rhs_calls =  Explicit.RK2(u, dt, RHS_func)
        u_new, rhs_calls =  EXPRB.Rosenbrock_Euler(u, dt, RHS_func, c, Gamma, 1e-4, 0)
        
        ### Update u and time
        u = u_new.copy()
        time_elapsed = time_elapsed + dt
        
    ## Write final data to files
    file_final_sol = open(path + "/Final_data_sol.txt", 'w+')
    file_final_sol.write(' '.join(map(str, u_new)) % u_new)
    file_final_sol.close()
                
### Call the function
solve()