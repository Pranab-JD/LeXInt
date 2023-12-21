"""
Created on Sat Aug 14 11:31:19 2022

@author: Pranab JD

Description: Constant Test
"""

import os, sys, shutil
import numpy as np
from scipy.sparse import csr_matrix
import matplotlib.pyplot as plt

from datetime import datetime

startTime = datetime.now()

sys.path.insert(1, "../Constant/")
from Cons_ExpInt import *

sys.path.insert(1, "../")
from Eigenvalues import *

##############################################################################

###? Initialize parameters
N = 400             # Number of grid points
eta = 10            # Peclet number
tmax = 0.5          # Final simulation time

###? Periodic boundaries
X = np.linspace(0, 1, N, endpoint = False)
dx = X[2] - X[1]    # Grid spacing

###? CFL conditions
adv_cfl = dx/eta
dif_cfl = dx**2/2
dt_cfl = min(adv_cfl, dif_cfl)
print("Adv. CFL: ", adv_cfl)
print("Dif. CFL: ", dif_cfl)
print()

###? Set up matrices
A_adv = np.zeros((N, N))
A_dif = np.zeros((N, N))

for ij in range(N):
    
    ##* 3rd order upwind
    A_adv[ij, int(ij + 2) % N] = -1/6
    A_adv[ij, int(ij + 1) % N] =  6/6
    A_adv[ij, ij % N]          = -3/6
    A_adv[ij, int(ij - 1) % N] = -2/6

    ##* 2nd order centered difference
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

###? Spectrum of the matrices
eigen_min_dif = 0.0 
eigen_max_dif, eigen_imag_dif = Gershgorin(A_dif)      # Max real, imag eigenvalue
# eigen_max_dif, its_power = Power_iteration(u, RHS_function)

###? Scaling and shifting factors
c = 0.5 * (eigen_max_dif + eigen_min_dif)
Gamma = 0.25 * (eigen_min_dif - eigen_max_dif)

### ------------------------------------------------------ ###

def solve(problem, integrator, N_cfl):
    
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

    ###? Parameters
    dt = N_cfl * dt_cfl
    time_elapsed = 0.0                                  # Time
    time_step = 0                                       # Number of time steps
    count_mv = 0                                        # Counter for matrix-vector products
    substeps = 1                                        # Initial guess for substeps
    tol = 1e-12                                         # Desired accuracy
    
    if integrator.__name__ == "EPI3":
        u_prev = np.zeros((len(u), 1))
    elif integrator.__name__ == "EPI4":
        u_prev = np.zeros((len(u), 2))
    elif integrator.__name__ == "EPI5":
        u_prev = np.zeros((len(u), 3))
    elif integrator.__name__ == "EPI6":
        u_prev = np.zeros((len(u), 4))

    ### Choose step size (dt)
    ncfl = '{:1.2f}'.format(N_cfl)
    print("N x dt(CFL): ", N_cfl, "x dt(CFL)")
    
    ### Read Leja points
    Leja_X = np.loadtxt("../Leja_10000.txt")
    
    ###! Computing divided differences may take a while. This is why we start
    ###! with 100 - 500 Leja points. If you get the warning 
    ###! "Warning!! Max. # of Leja points reached without convergence!!",
    ###! increase the number of Leja points or comment out the following line.
    Leja_X = Leja_X[0:1000]  
    
    ############## --------------------- ##############
    
    ###? Start timer
    tolTime = datetime.now()

    ###? Time loop
    while time_elapsed < tmax:
        
        if time_elapsed + dt > tmax:
            dt = tmax - time_elapsed

        ###? Integrate
        if (integrator.__name__ == "EPI3" and time_step < 1) or (integrator.__name__ == "EPI4" and time_step < 2)\
        or (integrator.__name__ == "EPI5" and time_step < 3) or (integrator.__name__ == "EPI6" and time_step < 4):
                
            ###? First few time steps are integrated using EXPRB43
            u_new, rhs_calls, substeps = EXPRB43(u, dt, substeps, RHS_function, c, Gamma, Leja_X, tol, 0)
        
        elif (integrator.__name__ == "EPI3" and time_step > 0) or (integrator.__name__ == "EPI4" and time_step > 1)\
          or (integrator.__name__ == "EPI5" and time_step > 2) or (integrator.__name__ == "EPI6" and time_step > 3):
                
            ###? EPI
            u_new, rhs_calls, substeps = integrator(u, u_prev, dt, substeps, RHS_function, c, Gamma, Leja_X, tol, 0)

        else:

            ###? EPIRK and EXPRB
            u_new, rhs_calls, substeps = integrator(u, dt, substeps, RHS_function, c, Gamma, Leja_X, tol, 0)

        ###? Update previous solutions for EPI methods
        if integrator.__name__ == "EPI3":
            u_prev = u.copy()
        elif integrator.__name__ == "EPI4":
            u_prev[:, 1] = u_prev[:, 0].copy()
            u_prev[:, 0] = u.copy()
        elif integrator.__name__ == "EPI5":
            u_prev[:, 2] = u_prev[:, 1].copy()
            u_prev[:, 1] = u_prev[:, 0].copy()
            u_prev[:, 0] = u.copy()
        elif integrator.__name__ == "EPI6":
            u_prev[:, 3] = u_prev[:, 2].copy()
            u_prev[:, 2] = u_prev[:, 1].copy()
            u_prev[:, 1] = u_prev[:, 0].copy()
            u_prev[:, 0] = u.copy()

        ###? Update parameters
        time_elapsed = time_elapsed + dt
        time_step = time_step + 1
        count_mv = count_mv + rhs_calls
        u = u_new.copy()
        
    ###? Stop timer
    tol_time = datetime.now() - tolTime

    ############## --------------------- ##############
    
    ###* Create required files/directories
    path = os.path.expanduser("./Test_data/Constant/" + str(problem) + "/T_final_" + str(tmax) + "/N_" + str(N) \
                              + "_eta_" + str(eta)  + "/" + str(integrator.__name__) + "/N_cfl_" + str(ncfl))
    if os.path.exists(path):
        shutil.rmtree(path)                     # remove previous directory with same name
    os.makedirs(path, 0o777)                    # create directory with access rights
    
    ###* Write final data to file
    final_data = open(path + "/Final_data.txt", 'w+')
    final_data.write(' '.join(map(str, u)) % u)
    final_data.close()
    
    ###* Write simulation results to file
    file_res = open(path + "/Results.txt", "w+")
    file_res.write("Time elapsed (secs): %s" % str(tol_time) + "\n" + "\n")
    file_res.write("Number of matrix-vector products = %d" % count_mv + "\n" + "\n")
    file_res.write("Step size" + "\n")
    file_res.write(str(N_cfl * dt_cfl) + "\n" + "\n")
    file_res.write("Time steps" + "\n")
    file_res.write(str(time_step) + "\n")
    file_res.close()

    print("\nTime elapsed: ", tol_time)
    print("Time steps: ", time_step)
    print("Total RHS Calls: ", count_mv)
    print("\n========================================================\n")

##############################################################################                
                
###? Call the function
###? solve(problem, integrator, N_CFL)

solve("Burgers", EPI5, 5)

N_cfl = [30, 50, 100, 150, 200, 300]

# for nn in N_cfl:
#     solve("Burgers", EPI3, nn)     #* N_CFL = 10 (factor multiplied to dt_CFL, dt = 10 * dt_CFL)

print('Total Time Elapsed = ', datetime.now() - startTime)
print()