# C++

A C++ implementation of a bunch of exponential integrators where the $\varphi_l$ functions are approximated using the Leja interpolation scheme. 

Embedded error estimates have been included for most integrators. These solvers return both the higher- and lower-order solutions, the difference of which can be used for estimating the stepsize (dt) for the next time step. Some intergators do not possess an embedded error (see `Leja.hpp`). One may use Richardson extrapolation to compute the error estimate in these cases. 

## Remarks:
1. A sample definition of the RHS function can be found in *Test &rarr; Burgers.hpp*. Any user-defined problem can be described in a similar way. 

2. To compute the **exact** solution of a homogenous linear problem, choose `real_Leja_exp`, whilst for non-homogenous problems, choose `real_Leja_phi_nl`.

3. For homogenous or non-honogenous linear problems, one may choose to go up to 10000 Leja points. However, one should be careful that the coefficients of the polynomial does not go below $10^{-13} - 10^{-16}$.

4. For nonlinear problems, at most 100 - 150 Leja points should suffice.
