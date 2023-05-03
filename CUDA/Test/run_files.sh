#!/bin/bash

#SBATCH -p tv
#SBATCH -o job2.out
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -t 00:05:00

# module load cuda/11.2 
# nvcc *.cu -O3 -o executable -lcublas 

g++ *.cpp -O3 -fopenmp -o executable

export OMP_PROC_BIND=close
export OMP_PLACES=cores
export OMP_DISPLAY_ENV=true
export OMP_NESTED=false
export OMP_NUM_THREADS=32

echo "Compiled successfully... Yaayy!"
echo ""

./executable