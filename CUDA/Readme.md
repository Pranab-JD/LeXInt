# CUDA

![C++](https://img.shields.io/badge/c++-%2300599C.svg?style=for-the-badge&logo=c%2B%2B&logoColor=white)
![nVIDIA](https://img.shields.io/badge/nVIDIA-%2376B900.svg?style=for-the-badge&logo=nVIDIA&logoColor=white)
![VS Code](https://img.shields.io/badge/VSCode-0078D4?style=for-the-badge&logo=visual%20studio%20code&logoColor=white)
![Github](https://img.shields.io/badge/GitHub-100000?style=for-the-badge&logo=github&logoColor=white)

Test examples for C++ and CUDA implementations can be found in *Test &rarr; Test_2D.cpp* and *Test &rarr; Test_2D.cu*, respectively. Problems considered include the linear Diffusion-Advection equation and the nonlinear Burgers' equation. To run scripts, use: `bash run_cuda.sh` or `sbatch run_cuda.sh`. To add other problems, simply define the relevant *RHS_function* (as defined in Burgers_2D.hpp or Dif_Adv_2D.hpp) and the initial condition(s) in the test files.

## Requirements
- gcc and nvcc compilers
- NVIDIA GPU
- CUDA 11.2 (or later)
