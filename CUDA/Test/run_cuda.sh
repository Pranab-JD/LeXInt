#!/bin/bash

#SBATCH -p a100
#SBATCH -o cuda_tol.out
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -t 51:00:00

module load cuda/11.2
nvcc test_2D.cu -O3 -o executable -lcublas

printf "Compiled successfully... Yaayy!\n"

printf "====================================================\n"
printf "Starting 1st set ....\n"

./executable 13 5 1e-10 5e-5

printf "====================================================\n"