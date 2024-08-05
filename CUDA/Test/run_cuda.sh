#!/bin/bash
#SBATCH --account=EUHPC_D14_051
#SBATCH --partition=boost_usr_prod
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=1
#SBATCH --gres=gpu:1
#SBATCH --time=00:20:00
#SBATCH --job-name=lexint
#SBATCH -o build/cuda_bw.out

## Compile using nvcc (cuBLAS required)
# nvcc ../test_2D.cu -O3 -o lexint_cuda -lcublas

printf "====================================================\n"
date

## E.g.: Say, domain = 1024 * 1024 (2^10 * 2^10), dt = 2*dt_cfl, tol = 1e-8, and T_final = 1.0, then run
## E.g.: ./executable 10 2 1e-8 1.0

# ./lexint_cuda 12 1 1e-12 2e-6
# ./lexint_cuda 12 10 1e-12 2e-5
# ./lexint_cuda 12 100 1e-12 2e-4

# ./lexint_cuda 13 1 1e-12 2e-7
# ./lexint_cuda 13 10 1e-12 2e-6
# ./lexint_cuda 13 100 1e-12 2e-5

# ./lexint_cuda 14 1 1e-12 2e-8
./lexint_cuda 14 100 1e-12 2e-7
# ./lexint_cuda 14 100 1e-12 2e-6

printf "Completed simulations ....\n"
date
printf "====================================================\n"
