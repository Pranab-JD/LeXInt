#  [LeXInt](#)

[**Le**](#)ja interpolation for e[**X**](#)ponential [**Int**](#)egrators is a temporal integration package exponential integrators using the method of polynomial interpolation at Leja points.

Exponential Rosenbrock (EXPRB) and Exponential Propagation Iterative Runge-Kutta (EPIRK) methods use the Leja interpolation method to compute the $\varphi_l(z)$ functions. For linear PDEs, one can get the ***exact*** solution (in time) by directly computing the matrix exponential using the functions "real_Leja_exp" and/or "imag_Leja_exp".

Examples for constant and adaptive (or variable) step size implementation for the Leja-based exponential integrators can be found in Python -> Test -> Constant_test.py or Adaptive_test.py. Test problems considered include the Burgers' equation and the Allen-Cahn equation.

## Literature
The publication associated with this code:

Deka, Einkemmer, and Tokman (2022), *LeXInt: Package for Exponential Integrators employing Leja interpolation*, arXiv:[2208.08269](
https://doi.org/10.48550/arXiv.2208.08269)

Other references:
1. Caliari et al. (2014), *Comparison of software for computing the action of the matrix exponential*, BIT Numer. Math., [54, 113](https://doi.org/10.1007/s10543-013-0446-0)

2. Deka \& Einkemmer (2022), *Efficient adaptive step size control for exponential integrators*, Comput. Math. Appl., [123, 59](https://doi.org/10.1016/j.camwa.2022.07.011)

3. Deka \& Einkemmer (2022), *Exponential Integrators for Resistive Magnetohydrodynamics: Matrix-free Leja Interpolation and Efficient Adaptive Time Stepping*, ApJS, [259, 57](https://doi.org/10.3847/1538-4365/ac5177)

## Contact
Pranab J. Deka  (<pranab.deka@uibk.ac.at>) <br />
Lukas Einkemmer (<lukas.einkemmer@uibk.ac.at>) <br />
Mayya Tokman  (<mtokman@ucmerced.edu>)
