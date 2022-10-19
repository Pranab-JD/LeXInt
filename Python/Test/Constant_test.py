"""
Created on Sat Aug 14 11:31:19 2021

@author: Pranab JD

Description: Constant Test
"""

import os, sys, shutil
import numpy as np
from scipy.sparse import csr_matrix

from datetime import datetime

startTime = datetime.now()

### ------------------------------------------------------ ###

sys.path.insert(1, "../Constant/")
from EXPRB_EPIRK import *

sys.path.insert(1, "../")
from Eigenvalues import *

##############################################################################

### Initialize parameters
N = 256
eta = 10
tmax = 0.01

## Periodic boundaries
X = np.linspace(0, 1, N, endpoint = False)
dx = X[2] - X[1]

### CFL conditions
adv_cfl = dx/eta
dif_cfl = dx**2/2
dt_cfl = min(adv_cfl, dif_cfl)
print("Adv. CFL: ", adv_cfl)
print("Dif. CFL: ", dif_cfl)
print()

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

A_adv = csr_matrix(A_adv * eta/dx)
A_dif = csr_matrix(A_dif/dx**2)
    
### ------------------------------------------------------ ###

def Burgers():
    
    ### Initial condition
    sigma = 0.02; x_0 = 0.9
    np.seterr(divide = 'ignore')
    u0 = 1 + (np.exp(1 - (1/(1 - (2 * X - 1)**2)))) + 1./2. * np.exp(-(X - x_0)**2/(2 * sigma**2))
    u = u0.copy()
    
    return u
    
def Allen_Cahn():
    
    ### Initial condition
    u0 = 0.1*(1 + np.cos(2*np.pi*X))
    u = u0.copy()
    
    return u

def Burgers_RHS_function(u):

    ### Viscous Burgers' Equation
    flux_u = A_dif.dot(u) + (0.5*A_adv.dot(u**2))

    return flux_u

def AC_RHS_function(u):

    ### Allen-Cahn
    flux_u = A_dif.dot(u) + 100*(u - u**3)
    
    return flux_u

### ------------------------------------------------------ ###

### Spectrum of the matrices
eigen_min_dif = 0.0 
eigen_max_dif, eigen_imag_dif = Gershgorin(A_dif)      # Max real, imag eigenvalue
# eigen_max_dif, its_power = Power_iteration(u, RHS_function)

### Scaling and shifting factors
c = 0.5 * (eigen_max_dif + eigen_min_dif)
Gamma = 0.25 * (eigen_min_dif - eigen_max_dif)

### ------------------------------------------------------ ###

def solve(N_cfl):
    
    ### Parameters
    dt = N_cfl * dt_cfl
    
    time_elapsed = 0.0                                  # Time
    time_step = 0                                       # Number of time steps
    count_mv = 0                                        # Counter for matrix-vector products
    
    tol = 1e-7                                          # Desired accuracy
        
    ############## --------------------- ##############
    
    ### Solve the given equation
    
    ### Choose proper initial condition
    u = Allen_Cahn()
    
    ### Choose step size (dt)
    ncfl = '{:1.2f}'.format(N_cfl)
    print("N x dt(CFL): ", N_cfl)
    
    ### Read Leja points
    Leja_X = np.loadtxt("../Leja_10000.txt")
    Leja_X = Leja_X[0:100]
    
    ############## --------------------- ##############
    
    ### Start timer
    tolTime = datetime.now()

    while time_elapsed < tmax:
        
        if time_elapsed + dt > tmax:
            dt = tmax - time_elapsed
            
        u_new, rhs_calls = EPIRK4s3A(u, dt, AC_RHS_function, c, Gamma, Leja_X, tol, 0)
        
        ### Update u and time
        u = u_new.copy()
        time_elapsed = time_elapsed + dt
        time_step = time_step + 1
        count_mv = count_mv + rhs_calls
        
    ############## --------------------- ##############
        
    ### Stop timer
    tol_time = datetime.now() - tolTime
    
    ### Create required files/directories
    # path = os.path.expanduser("./Test_data/Constant/Burgers/N_64_eta_200/EPIRK4s3A/N_cfl_" + str(ncfl))
    # if os.path.exists(path):
    #     shutil.rmtree(path)                     # remove previous directory with same name
    # os.makedirs(path, 0o777)                    # create directory with access rights
    
    # ### Write final data to file
    # final_data = open(path + "/Final_data.txt", 'w+')
    # final_data.write(' '.join(map(str, u)) % u)
    # final_data.close()
    
    # ### Write simulation results to file
    # file_res = open(path + "/Results.txt", "w+")
    # file_res.write("Time elapsed (secs): %s" % str(tol_time) + "\n" + "\n")
    # file_res.write("Number of matrix-vector products = %d" % count_mv + "\n" + "\n")
    # file_res.write("Step sizes" + "\n")
    # file_res.write(str(N_cfl * dt_cfl) + "\n")
    # file_res.write("Time steps" + "\n")
    # file_res.write(str(time_step) + "\n")
    # file_res.close()

    print("Time elapsed: ", tol_time)
    print("Total RHS Calls: ", count_mv)
    print("\n========================================================\n")

##############################################################################                
                
### Call the function

N_cfl = [1]

for cfl in N_cfl:
    solve(cfl)


print('Total Time Elapsed = ', datetime.now() - startTime)