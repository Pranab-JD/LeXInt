"""
Created on Sat Aug 14 19:35:13 2021

@author: Pranab JD

Description: Plot order of convergence for different schemes
"""

import numpy as np
import matplotlib.pyplot as plt

### -------------------------------------------------------- ###

### Reference data
data_ref = "Test_data/Reference/Data_0.0005_dt_cfl/Final_data_sol.txt"
ref = np.loadtxt(data_ref)

### -------------------------------------------------------- ###

def compute_error(integrator):
    
    ### Data Sets
    file_1 = "Test_data/" + str(integrator) + "/Data_0.09_dt_cfl/Final_data_sol.txt"
    file_2 = "Test_data/" + str(integrator) + "/Data_0.1_dt_cfl/Final_data_sol.txt"
    file_3 = "Test_data/" + str(integrator) + "/Data_0.25_dt_cfl/Final_data_sol.txt"
    file_4 = "Test_data/" + str(integrator) + "/Data_0.5_dt_cfl/Final_data_sol.txt"
    file_5 = "Test_data/" + str(integrator) + "/Data_0.75_dt_cfl/Final_data_sol.txt"

    data_1 = np.loadtxt(file_1)
    data_2 = np.loadtxt(file_2)
    data_3 = np.loadtxt(file_3)
    data_4 = np.loadtxt(file_4)
    data_5 = np.loadtxt(file_5)
    
    ### Error (l1 norm)
    error_1 = np.mean(abs(ref - data_1))
    error_2 = np.mean(abs(ref - data_2))
    error_3 = np.mean(abs(ref - data_3))
    error_4 = np.mean(abs(ref - data_4))
    error_5 = np.mean(abs(ref - data_5))
    
    return np.array([error_1, error_2, error_3, error_4, error_5])

### -------------------------------------------------------- ###

rk2_error = compute_error("RK2")
rk4_error = compute_error("RK4")
# roseu_error = compute_error("Rosenbrock_Euler")
exprb42_error = compute_error("EXPRB42")

### -------------------------------------------------------- ###

### Plots
dt = np.array([0.09, 0.1, 0.25, 0.5, 0.75])

plt.figure(figsize = (8, 6), dpi = 150)
plt.plot(dt, dt * 2e-6, 'c-')
plt.plot(dt, dt * 4e-6, 'm-')
plt.loglog(dt, rk2_error, 'rd:', label = 'RK2')
plt.loglog(dt, rk4_error, 'bo--', label = 'RK4')
plt.legend()
plt.savefig("./RK_order.png")