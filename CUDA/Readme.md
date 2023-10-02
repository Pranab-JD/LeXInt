# CUDA
![C++](https://img.shields.io/badge/c++-%2300599C.svg?style=for-the-badge&logo=c%2B%2B&logoColor=white)
![nVIDIA](https://img.shields.io/badge/nVIDIA-%2376B900.svg?style=for-the-badge&logo=nVIDIA&logoColor=white)
![VS Code](https://img.shields.io/badge/VSCode-0078D4?style=for-the-badge&logo=visual%20studio%20code&logoColor=white)
![Github](https://img.shields.io/badge/GitHub-100000?style=for-the-badge&logo=github&logoColor=white)

Test examples for C++ and CUDA implementations can be found in *Test &rarr; Test_2D.cpp* and *Test &rarr; Test_2D.cu*, respectively.  To run the codes, use `sbatch run_cpp.sh` or `sbatch run_cuda.sh`. Problems considered include the linear diffusion-advection equation and the nonlinear Burgers' equation. To add other problems, simply define the relevant RHS function (as defined in *Burgers_2D.hpp* or *Dif_Adv_2D.hpp*) and the initial condition(s) in the test files.
Test examples for C++ and CUDA implementations can be found in *Test &rarr; Test_2D.cpp* and *Test &rarr; Test_2D.cu*, respectively.  To run the codes, use `bash run_cpp.sh` or `bash run_cuda.sh`. Alternatively, you could also use *sbatch* instead of *bash* if you have *slurm* installed on your computer. Problems considered include the linear diffusion-advection equation and the nonlinear Burgers' equation. To add other problems, simply define the relevant RHS function (as defined in *Burgers_2D.hpp* or *Dif_Adv_2D.hpp*) and the initial condition(s) in the test files.

## Requirements
- gcc and nvcc compilers
- NVIDIA GPU
- CUDA 11.2 (or later)
## Remarks
1. Before running the test files, please select (comment or uncomment) the desired problem and integrator (lines 79 - 89) in *Test_2D.cpp* or *Test_2D.cu*. 
2. If you get the error *"Warning!! Max. number of Leja points reached without convergence!!"*, consider reducing the time step size (dt) or increasing the number of Leja points (line 130 in *Leja.hpp*).
3. For multidimensional problems, the (input/output) data containers are expected to lie contiguous in memory. 
4. If the user-specified RHS function consists of additional parameters, one could potentially construct a ***class*** and have these supplementary parameters localised to the ***class***:
```cpp
struct RHS
{
    RHS(*args)
    void operator(input, output)
    {
        rhs(input, output, *args)
    }
}
