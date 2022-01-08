"""
Created on Sat Aug 14 19:35:13 2021

@author: Pranab JD

Description: Plot order of convergence for different schemes
"""

import re, os.path
import numpy as np
import matplotlib.pyplot as plt


### Given Data Sets
path_rkf45 = "Test_data/Adaptive/RKF45/"
path_exprb32 = "Test_data/Adaptive/EXPRB32/"
path_exprb43 = "Test_data/Adaptive/EXPRB43/"
path_epirk5p1 = "Test_data/Adaptive/EPIRK5P1/"
path_exprb54s4 = "Test_data/Adaptive/EXPRB54s4/"
path_exprb54s5 = "Test_data/Adaptive/EXPRB54s5/"

############################## Compute l2 error ##############################

### Open data files
file_ref = "Test_data/Adaptive/Reference/Final_data_sol.txt"

data_set_ref = np.loadtxt(file_ref)

Nx = len(data_set_ref)

##############################################################################

def l2_norm(data_set):

    norm = 0
    for ii in range(0, len(data_set_ref)):
        norm = norm + (data_set[ii] - data_set_ref[ii])**2

    return norm**0.5/len(data_set_ref)

def compute_error(path):

    file_0 = path + "tol_1.0e-03/Final_data_sol.txt"
    file_1 = path + "tol_1.0e-04/Final_data_sol.txt"
    file_2 = path + "tol_1.0e-05/Final_data_sol.txt"
    file_3 = path + "tol_1.0e-06/Final_data_sol.txt"
    file_4 = path + "tol_1.0e-07/Final_data_sol.txt"
    file_5 = path + "tol_1.0e-08/Final_data_sol.txt"
    file_6 = path + "tol_1.0e-09/Final_data_sol.txt"
    # file_7 = path + "tol_1.0e-10/Final_data_sol.txt"

    data_set_0 = np.loadtxt(file_0)
    data_set_1 = np.loadtxt(file_1)
    data_set_2 = np.loadtxt(file_2)
    data_set_3 = np.loadtxt(file_3)
    data_set_4 = np.loadtxt(file_4)
    data_set_5 = np.loadtxt(file_5)
    data_set_6 = np.loadtxt(file_6)
    # data_set_7 = np.loadtxt(file_7)

    error_0 = l2_norm(data_set_0)
    error_1 = l2_norm(data_set_1)
    error_2 = l2_norm(data_set_2)
    error_3 = l2_norm(data_set_3)
    error_4 = l2_norm(data_set_4)
    error_5 = l2_norm(data_set_5)
    error_6 = l2_norm(data_set_6)
    # error_7 = l2_norm(data_set_7)

    return np.array([error_0, error_1, error_2, error_3, error_4, error_5, error_6])

#################################### Cost ####################################

path_rkf45 = "Test_data/Adaptive/RKF45/"
path_exprb32 = "Test_data/Adaptive/EXPRB32/"
path_exprb43 = "Test_data/Adaptive/EXPRB43/"
path_epirk5p1 = "Test_data/Adaptive/EPIRK5P1/"
path_exprb54s4 = "Test_data/Adaptive/EXPRB54s4/"
path_exprb54s5 = "Test_data/Adaptive/EXPRB54s5/"

### -------------------------------------------------------- ###

def num_matrix_vector(path):

    tol_files = ['tol_1.0e-03/', 'tol_1.0e-04/', 'tol_1.0e-05/',
                 'tol_1.0e-06/', 'tol_1.0e-07/', 'tol_1.0e-08/',
                 'tol_1.0e-09/']

    cost = np.zeros(7)
    count_jj = 0

    for jj in tol_files:
        if os.path.exists(path + jj + 'Results.txt'):
            with open(path + jj + 'Results.txt') as sim:
                lines_sim = sim.readlines()

                cost_time = lines_sim[0]
                cost_time = re.findall(r'[\d.]+', cost_time)
                cost_time = np.fromstring(cost_time[0], dtype = float, sep = ' ')

        cost[count_jj] = cost_time
        count_jj = count_jj + 1

    return cost
    
##############################################################################

### Function calls

rkf45_cost = num_matrix_vector(path_rkf45)
exprb32_cost = num_matrix_vector(path_exprb32)
exprb43_cost = num_matrix_vector(path_exprb43)
epirk5p1_cost = num_matrix_vector(path_epirk5p1)
exprb54s4_cost = num_matrix_vector(path_exprb54s4)
exprb54s5_cost = num_matrix_vector(path_exprb54s5)

rkf45_error = compute_error(path_rkf45)
exprb32_error = compute_error(path_exprb32)
exprb43_error = compute_error(path_exprb43)
epirk5p1_error = compute_error(path_epirk5p1)
# exprb54s4_error = compute_error(path_exprb54s4)
# exprb54s5_error = compute_error(path_exprb54s5)

##############################################################################

### Plots
tol = np.array([1e-3, 1e-4, 1e-5, 1e-6, 1e-7, 1e-8, 1e-9])

fig = plt.figure(figsize = (12, 6), dpi = 150)

plt.subplot(1, 2, 1)
plt.loglog(rkf45_cost, tol, 'co-', label = "RKF45")
plt.loglog(exprb32_cost, tol, 'rd-', label = "EXPRB32")
plt.loglog(exprb43_cost, tol, 'bH-', label = "EXPRB43")
plt.loglog(epirk5p1_cost, tol, 'gP-', label = "EPIRK5P1")
plt.loglog(exprb54s4_cost, tol, 'm*-', label = "EXPRB54s4")
plt.loglog(exprb54s5_cost, tol, 'k>-', label = "EXPRB54s5")

plt.xlabel("# of matrix-vector products", fontsize = 18)
plt.ylabel("Tolerance", fontsize = 18)
plt.tick_params(axis = 'x', which = 'major', labelsize = 16)
plt.tick_params(axis = 'y', which = 'major', labelsize = 16)

plt.tight_layout()
plt.legend(prop = {'size': 14}, ncol = 1, loc = 1)


plt.subplot(1, 2, 2)
plt.loglog(rkf45_cost, rkf45_error, 'co-', label = "RKF45")
plt.loglog(exprb32_cost, exprb32_error, 'rd-', label = "EXPRB32")
plt.loglog(exprb43_cost, exprb43_error, 'bH-', label = "EXPRB43")
# plt.loglog(epirk5p1_cost, epirk5p1_error, 'gP-', label = "EPIRK5P1")
# plt.loglog(exprb54s4_cost[0:6], exprb54s4_error, 'm*-', label = "EXPRB54s4")
# plt.loglog(exprb54s5_cost[0:6], exprb54s5_error, 'k>-', label = "EXPRB54s5")

plt.xlabel("# of matrix-vector products", fontsize = 18)
plt.ylabel("l2 error", fontsize = 18)
plt.tick_params(axis = 'x', which = 'major', labelsize = 16)
plt.tick_params(axis = 'y', which = 'major', labelsize = 16)

plt.tight_layout(pad = 1.75)

plt.savefig("./Test_data/Adaptive/Cost_plot.eps")
plt.show()

##############################################################################