#!/bin/bash

#SBATCH -p tv
#SBATCH -o cpp_.out
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -t 10:00:00

## Compile using g++
g++ test_2D.cpp -O3 -fopenmp -o executable

export OMP_PROC_BIND=close
export OMP_PLACES=cores
export OMP_DISPLAY_ENV=true
export OMP_NESTED=false
export OMP_NUM_THREADS=32

printf "Compiled successfully... Yaayy!\n"

printf "====================================================\n"
printf "Starting 1st set ....\n"

## E.g.: Say, domain = 1024 * 1024 (2^10 * 2^10), dt = 2*dt_cfl, tol = 1e-8, and T_final = 1.0, then run
## E.g.: ./executable 10 2 1e-8 1.0

./executable 9 2 1e-10 1e-4

printf "Completed simulations ....\n"
printf "====================================================\n"