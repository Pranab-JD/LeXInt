#!/bin/bash

#SBATCH -p gtx
#SBATCH -o job.out
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -t 00:02:00

# module load cuda/11.2
# nvcc test_2D.cu -O3 -o executable -lcublas 

g++ test_2D.cpp -O3 -fopenmp -o executable

# export OMP_PROC_BIND=close
# export OMP_PLACES=cores
# export OMP_DISPLAY_ENV=true
# export OMP_NESTED=false
# export OMP_NUM_THREADS=1

echo "Compiled successfully... Yaayy!"
echo ""

./executable