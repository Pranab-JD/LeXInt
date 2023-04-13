#!/bin/bash

#SBATCH -p gtx
#SBATCH -o job.out
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -t 00:10:00

module load cuda/11.2 

nvcc *.cu -o executable -lcublas 
./executable