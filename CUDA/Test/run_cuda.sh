#!/bin/bash

#SBATCH -p tv
#SBATCH -o cuda_.out
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -t 10:00:00

## Load available CUDA module
module load cuda/11.2

## Compile using nvcc (cuBLAS required)
nvcc test_2D.cu -O3 -o executable -lcublas

printf "Compiled successfully... Yaayy!\n"

printf "====================================================\n"
printf "Starting 1st set ....\n"

## E.g.: Say, domain = 1024 * 1024 (2^10 * 2^10), dt = 2*dt_cfl, tol = 1e-8, and T_final = 1.0, then run
## E.g.: ./executable 10 2 1e-8 1.0

./executable 10 5 1e-10 5e-5

printf "Completed simulations ....\n"
printf "====================================================\n"
