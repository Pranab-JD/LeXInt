# CUDA -> Integrators

Here, we have a collection of exponential integrators. Rosenbrock--Euler and EPIRK4s3B do not have an embedded error estimate, whilst the rest do. 

To call one (or more) of these exponential integrators,
    1. Add ``#include "../Leja_GPU.hpp"`` in the main file (main.cpp or main.cu).
    2. Create an object of the class ``Leja_GPU(N, integrator_name)``. ``N`` corresponds to the total number of grid points and ``integrator_name`` corresponds to the desired exponential integrator. E.g., ``Leja_GPU<RHS> leja_gpu{N, EXPRB32}``; where ``RHS`` corresponds to the RHS class that contains the RHS operator.
    3. Invoke the object of the class ``Leja_GPU`` as ``leja_gpu.embed_exp_int(RHS, u, u_low, u_sol, error, N, Leja_X, c, Gamma, tol, dt, iters, GPU_access);`` for embedded exponential integraotrs or ``leja_gpu.exp_int(RHS, u, u_sol, N, Leja_X, c, Gamma, tol, dt, iters, GPU_access);`` for non-embedded exponential integrators.


