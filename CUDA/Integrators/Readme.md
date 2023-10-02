#  [LeXInt::CUDA::Integrators](#)

Here, we have a collection of exponential integrators. Rosenbrock--Euler and EPIRK4s3B do not have an embedded error estimate, whilst the rest do. These exponential integrators call the ``real_Leja_phi`` function to approximate exponential-like functions applied to the relevant vectors.

## Invoking the exponential integrators

- Add ``#include "./LeXInt/CUDA/Leja.hpp"`` in the main file (main.cpp or main.cu).
    
- Create an object of the class ``Leja(N, integrator_name)``. 'N' is the total number of grid points and 'integrator_name' corresponds to the desired exponential integrator. E.g., ``Leja<RHS> leja_gpu{N, EXPRB32}``; where ``RHS``is RHS class that contains the RHS operator.

- Invoke the object of the class ``Leja`` as ``leja_gpu.embed_exp_int`` for embedded exponential integrators or ``leja_gpu.exp_int`` for non-embedded exponential integrators. For more info, see `Test -> test_2D.cu (lines 257 and 276)`.

### Technical Aspects

* `c` and `Gamma` have to be determined prior to invoking an exponential integrator. See `Test -> test_2D.cu (lines 193 to 198)`.
* `iters` determines the number of Leja iterations per time step. This may be considered as a proxy of the computational cost.

