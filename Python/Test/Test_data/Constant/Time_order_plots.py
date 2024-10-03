"""
Created on Thu Nov 2 11:55 2023

@author: PJD
"""

import numpy as np
import matplotlib.pyplot as plt

### ======================================================= ###

### Reference Solution
path = "./Burgers/T_final_0.5/N_400_eta_10/"

file_ref = path + "/EPI5/N_cfl_5.00/Final_data.txt"
data_set_ref = np.loadtxt(file_ref)

def l1_error(data_set):
    return np.mean(abs(data_set - data_set_ref))/np.linalg.norm(data_set_ref)

def compute_error(solver):

	file_1 = path + solver + "/N_cfl_30.00/Final_data.txt"
	file_2 = path + solver + "/N_cfl_50.00/Final_data.txt"
	file_3 = path + solver + "/N_cfl_100.00/Final_data.txt"
	file_4 = path + solver + "/N_cfl_150.00/Final_data.txt"
	file_5 = path + solver + "/N_cfl_200.00/Final_data.txt"
	file_6 = path + solver + "/N_cfl_300.00/Final_data.txt"

	data_set_1 = np.loadtxt(file_1)
	data_set_2 = np.loadtxt(file_2)
	data_set_3 = np.loadtxt(file_3)
	data_set_4 = np.loadtxt(file_4)
	data_set_5 = np.loadtxt(file_5)
	data_set_6 = np.loadtxt(file_6)

	error_A = l1_error(data_set_1)
	error_B = l1_error(data_set_2)
	error_C = l1_error(data_set_3)
	error_D = l1_error(data_set_4)
	error_E = l1_error(data_set_5)
	error_F = l1_error(data_set_6)

	error_array = np.array([error_A, error_B, error_C, error_D, error_E, error_F])

	return error_array

### ======================================================= ###

### Given Data Sets
solver_1 = "EPI3"
solver_2 = "EPI4"
solver_3 = "EPI5"
solver_4 = "EPI6"

error_1 = compute_error(solver_1)
error_2 = compute_error(solver_2)
error_3 = compute_error(solver_3)
error_4 = compute_error(solver_4)

### ======================================================= ###

### Plots
X = np.array([30, 50, 100, 150, 200, 300])

plt.figure(figsize = (8, 6), dpi = 200)

plt.loglog(X, error_1, 'bo-', label = "EPI3")
plt.loglog(X, error_2, 'rd-', label = "EPI4")
plt.loglog(X, error_3, 'gH-', label = "EPI5")
plt.loglog(X, error_4, 'mP-', label = "EPI6")


# plt.loglog(X, 4e-15*X**3, "c-", label = "O(3)")
# plt.loglog(X, 1e-18*X**4, "k-", label = "O(4)")
# plt.loglog(X, 1e-20*X**5, "m-", label = "O(5)")

plt.tick_params(axis = 'x', which = 'major', labelsize = 16)
plt.tick_params(axis = 'y', which = 'major', labelsize = 16)

plt.title("Temporal Order", fontsize = 20)
plt.xlabel("Step size", fontsize = 16)
plt.ylabel("l1 error", fontsize = 16)

plt.legend(prop = {'size': 14}, ncol = 2)

plt.savefig(path + "/Temporal_order_B.eps")

### ======================================================= ###