"""
Created on Sat Aug 14 19:35:13 2021

@author: Pranab JD

Description: Plot order of convergence for different schemes
"""

import re
import numpy as np
import matplotlib.pyplot as plt


### Given Data Sets
file_1 = "Test_data/Adaptive/RKF45/tol_1.0e-07/Final_data_sol.txt"
file_2 = "Test_data/Adaptive/EXPRB32/tol_1.0e-07/Final_data_sol.txt"
file_3 = "Test_data/Adaptive/EXPRB43/tol_1.0e-07/Final_data_sol.txt"
file_4 = "Test_data/Adaptive/Reference/Final_data_sol.txt"

data_set_1 = np.loadtxt(file_1)
data_set_2 = np.loadtxt(file_2)
data_set_3 = np.loadtxt(file_3)
data_set_4 = np.loadtxt(file_4)

X = np.linspace(0, 1, len(data_set_1))

##############################################################################

### Plots

plt.figure(figsize = (8, 6), dpi = 150)

plt.plot(X, data_set_1, 'c--', label = "RKF45")
plt.plot(X, data_set_2, 'r-.', label = "EXPRB32")
plt.plot(X, data_set_3, 'b:', label = "EXPRB43")
plt.plot(X, data_set_4, 'g.', label = "Reference")

plt.xlabel("X", fontsize = 18)
plt.ylabel("U(X, t)", fontsize = 18)
plt.tick_params(axis = 'x', which = 'major', labelsize = 16)
plt.tick_params(axis = 'y', which = 'major', labelsize = 16)

plt.tight_layout()
plt.legend(prop = {'size': 17})
plt.savefig("./Final_plot_1e-7.eps")
