#!/bin/bash

#SBATCH -p tv
#SBATCH -o cpp_.out
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -t 10:00:00

g++ test_2D.cpp -O3 -fopenmp -o executable

export OMP_PROC_BIND=close
export OMP_PLACES=cores
export OMP_DISPLAY_ENV=true
export OMP_NESTED=false
export OMP_NUM_THREADS=32

printf "Compiled successfully... Yaayy!\n"

printf "====================================================\n"

printf "Starting 1st set ....\n"
./executable 10 5 1e-10 5e-5

printf "====================================================\n"
