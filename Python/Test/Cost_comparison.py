"""
Created on Sat Aug 14 19:35:13 2022

@author: Pranab JD

Description: Cost vs. error plots
"""

import re, os.path
import numpy as np
import matplotlib.pyplot as plt

### Select integrators
integrator_1 = "EXPRB32"
integrator_2 = "EXPRB54s4"
integrator_3 = "EPIRK4s3"
integrator_4 = "EPIRK4s3A"

### Given Data Sets
path_A = "Test_data/Adaptive/AC/N_64/" + str(integrator_1) + "/"
path_B = "Test_data/Adaptive/AC/N_64/" + str(integrator_2) + "/"

path_C = "Test_data/Adaptive/AC/N_64/" + str(integrator_3) + "/"
path_D = "Test_data/Adaptive/AC/N_64/" + str(integrator_4) + "/"

### Reference data
file_ref = "Test_data/Adaptive/AC/N_64/EXPRB32/tol_1.0e-11/Final_data.txt"
data_set_ref = np.loadtxt(file_ref)

##############################################################################

def l2_norm(data_set):

    norm = 0
    for ii in range(0, len(data_set_ref)):
        norm = norm + (data_set[ii] - data_set_ref[ii])**2

    return norm**0.5/len(data_set_ref)


def compute_error(path):

    file_0 = path + "tol_1.0e-02/Final_data.txt"
    file_1 = path + "tol_1.0e-03/Final_data.txt"
    file_2 = path + "tol_1.0e-04/Final_data.txt"
    file_3 = path + "tol_1.0e-05/Final_data.txt"
    file_4 = path + "tol_1.0e-06/Final_data.txt"
    file_5 = path + "tol_1.0e-07/Final_data.txt"
    file_6 = path + "tol_1.0e-08/Final_data.txt"

    data_set_0 = np.loadtxt(file_0)
    data_set_1 = np.loadtxt(file_1)
    data_set_2 = np.loadtxt(file_2)
    data_set_3 = np.loadtxt(file_3)
    data_set_4 = np.loadtxt(file_4)
    data_set_5 = np.loadtxt(file_5)
    data_set_6 = np.loadtxt(file_6)

    error_0 = l2_norm(data_set_0)
    error_1 = l2_norm(data_set_1)
    error_2 = l2_norm(data_set_2)
    error_3 = l2_norm(data_set_3)
    error_4 = l2_norm(data_set_4)
    error_5 = l2_norm(data_set_5)
    error_6 = l2_norm(data_set_6)

    return np.array([error_0, error_1, error_2, error_3, error_4, error_5, error_6])


def num_matrix_vector(path):

    tol_files = ['tol_1.0e-02/', 'tol_1.0e-03/', 'tol_1.0e-04/',
                 'tol_1.0e-05/', 'tol_1.0e-06/', 'tol_1.0e-07/',
                 'tol_1.0e-08/']

    cost = np.zeros(len(tol_files))
    time_array = np.zeros(len(tol_files))
    count_jj = 0

    for jj in tol_files:
        if os.path.exists(path + jj + 'Results.txt'):
            with open(path + jj + 'Results.txt') as sim:
                lines_sim = sim.readlines()
                
                time = lines_sim[0]
                time = re.findall(r'[\d.]+', time)
                hr = np.fromstring(time[0], dtype = float, sep = ' ')
                mn = np.fromstring(time[1], dtype = float, sep = ' ')
                sc = np.fromstring(time[2], dtype = float, sep = ' ')
                time_elapsed = (hr*60) + (mn*60) + sc

                cost_time = lines_sim[2]
                cost_time = re.findall(r'[\d.]+', cost_time)
                cost_time = np.fromstring(cost_time[0], dtype = float, sep = ' ')

            cost[count_jj] = cost_time
            time_array[count_jj] = time_elapsed
            
            count_jj = count_jj + 1

    return cost, time_array
    
##############################################################################

### Function calls
cost_A, time_A = num_matrix_vector(path_A)
cost_B, time_B = num_matrix_vector(path_B)

cost_C, time_C = num_matrix_vector(path_C)
cost_D, time_D = num_matrix_vector(path_D)

error_A = compute_error(path_A)
error_B = compute_error(path_B)

error_C = compute_error(path_C)
error_D = compute_error(path_D)

##############################################################################

### Plots
tol = np.logspace(-2, -8, 7)

fig = plt.figure(figsize = (14, 6), dpi = 100)

print(error_D)

plt.subplot(1, 2, 1)
plt.loglog(time_A, tol, 'rd:', label = str(integrator_1), markersize = 12)
plt.loglog(time_B, tol, 'bH:', label = str(integrator_2), markersize = 12)
plt.loglog(time_C, tol, 'mo:', label = str(integrator_3), markersize = 12)
plt.loglog(time_D, tol, 'gP:', label = str(integrator_4), markersize = 12)

plt.xlabel("Time (s)", fontsize = 18)
plt.ylabel("Tolerance", fontsize = 18)

plt.tick_params(axis = 'x', which = 'major', labelsize = 18)
plt.tick_params(axis = 'y', which = 'major', labelsize = 18)
plt.minorticks_off()

plt.legend(prop = {'size': 18}, ncol = 1)

plt.subplot(1, 2, 2)
plt.loglog(time_A, error_A, 'rd:', label = str(integrator_1), markersize = 12)
plt.loglog(time_B, error_B, 'bH:', label = str(integrator_2), markersize = 12)
plt.loglog(time_C, error_C, 'mo:', label = str(integrator_3), markersize = 12)
plt.loglog(time_D, error_D, 'gP:', label = str(integrator_4), markersize = 12)

plt.xlabel("Time (s)", fontsize = 18)
plt.ylabel("l2 error", fontsize = 18)

plt.tick_params(axis = 'x', which = 'major', labelsize = 18)
plt.tick_params(axis = 'y', which = 'major', labelsize = 18)
plt.minorticks_off()

plt.legend(prop = {'size': 18}, ncol = 1)

plt.tight_layout()

plt.savefig("./Test_data/Adaptive/AC/Cost_N_64_test.eps")

##############################################################################