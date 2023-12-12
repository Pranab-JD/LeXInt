"""
Created on Sat Aug 14 11:31:19 2022

@author: Pranab JD

Description: Adaptive Test
"""

import os, sys, shutil
import numpy as np
from scipy.sparse import csr_matrix

from datetime import datetime

startTime = datetime.now()

### ------------------------------------------------------ ###

sys.path.insert(1, "../Variable/")
from Var_ExpInt import *

sys.path.insert(1, "../")
from Eigenvalues import *

##############################################################################

### Initialize parameters
N = 300             # Number of grid points
eta = 10            # Peclet number
tmax = 0.1          # Final simulation time

## Periodic boundaries
X = np.linspace(0, 1, N, endpoint = False)
dx = X[2] - X[1]    # Grid spacing

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

def Burgers_RHS_function(u):

    ### Viscous Burgers' Equation
    flux_u = A_dif.dot(u) + (0.5*A_adv.dot(u**2))

    return flux_u

def Allen_Cahn():
    
    ### Initial condition
    u0 = 0.1*(1 + np.cos(2*np.pi*X))
    u = u0.copy()
    
    return u

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
    
def solve(problem, integrator, order, tol):
    
    ### Choose proper initial condition and RHS function
    if problem == "Allen_Cahn":
        u = Allen_Cahn()
        RHS_function = AC_RHS_function
    elif problem == "Burgers":
        u = Burgers()
        RHS_function = Burgers_RHS_function
    else:
        print("Problem not defined!")

    print("Problem: ", problem)
    print("Integrator: ", integrator.__name__)
    
    ### Parameters
    dt = 5*dt_cfl
    time = 0                                            # Time
    time_step = 0                                       # Number of time steps
    count_mv = 0                                        # Counter for matrix-vector products
        
    dt_history = []                                     # Array - dt used
    time_arr = []                                       # Array - time elapsed after each time step
    
    ############## --------------------- ##############

    ### Choose tolerance
    emax = '{:5.1e}'.format(tol)
    print("Tolerance: ", "{:e}".format(tol))
    
    ### Read Leja points
    Leja_X = np.loadtxt("../Leja_10000.txt")
    
    ### Computing divided differences may take a while. This is why we start
    ### with 100 - 500 Leja points. If you get the warning 
    ### "Warning!! Max. # of Leja points reached without convergence!!",
    ### increase the number of Leja points or comment out the following line.
    Leja_X = Leja_X[0:500]     
    
    ############## --------------------- ##############
    
    ### Solve the given equation; start timer
    tolTime = datetime.now()
    
    while time < tmax:
        
        ### Final time step
        if time + dt > tmax:
            dt = tmax - time
            
        ### Solve
        u_low, u_high, rhs_calls_1 = integrator(u, dt, RHS_function, c, Gamma, Leja_X, tol, 0)
        
        ### Error
        error = np.mean(abs(u_low - u_high))
        
        if error > tol:
            
            while error > tol:
            
                new_dt = dt * (tol/error)**(1/(order + 1))
                dt = 0.9 * new_dt                       # Safety factor
                
                ### Solve (with smaller value of dt)
                u_low, u_high, rhs_calls_2 = integrator(u, dt, RHS_function, c, Gamma, Leja_X, tol, 0)
            
                error = np.mean(abs(u_low - u_high))
            
        else:
            rhs_calls_2 = 0
            
        ### Update u and time
        u = u_high.copy()
        time = time + dt
        time_step = time_step + 1
        count_mv = count_mv + rhs_calls_1 + rhs_calls_2
        
        ### Append data to arrays
        dt_history.append(dt)
        time_arr.append(time)
        print("Error: ", error)
        print()
        
        ### dt for next time step
        new_dt = dt * (tol/error)**(1/(order + 1))
        dt = 0.9 * new_dt                       # Safety factor
        
    ############## --------------------- ##############
        
    ### Stop timer
    tol_time = datetime.now() - tolTime
    
    ### Create required files/directories
    # path = os.path.expanduser("./Test_data/Adaptive/" + str(problem) + "/T_final_" + str(tmax) + "/N_" + str(N) \
    #                           + "_eta_" + str(eta)  + "/" + str(integrator.__name__) + "/tol_" + str(emax))
    # if os.path.exists(path):
    #     shutil.rmtree(path)                     # remove previous directory with same name
    # os.makedirs(path, 0o777)                    # create directory with access rights
    
    # ### Write final data to file
    # final_data = open(path + "/Final_data.txt", 'w+')
    # final_data.write(' '.join(map(str, u)) % u)
    # final_data.close()
    
    # ### Write simulation results to file
    # file = open(path + '/Results.txt', 'w+')
    # file.write("Time elapsed (secs): %s" % str(tol_time) + "\n" + "\n")
    # file.write('Number of matrix-vector products = %d' % count_mv + '\n' + '\n')
    # file.write(' '.join(map(str, dt_history)) % dt_history + '\n' + '\n')
    # file.write(' '.join(map(str, time_arr)) % time_arr)
    # file.close()

    print("\nTime elapsed: ", tol_time)
    print("Total RHS Calls: ", count_mv)
    print("Number of time steps: ", time_step)
    print("\n========================================================\n")
    
##############################################################################
                
### Call the function

### solve(problem, integrator, order of error-estimate (depends on the integrator), tol)
solve("Allen_Cahn", EXPRB43, 3, 1e-6)

print('Total Time Elapsed = ', datetime.now() - startTime)
