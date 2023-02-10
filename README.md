#  [LeXInt](#)


![Python](https://img.shields.io/badge/Python-FFD43B?style=for-the-badge&logo=python&logoColor=blue)
![C++](https://img.shields.io/badge/c++-%2300599C.svg?style=for-the-badge&logo=c%2B%2B&logoColor=white)
![nVIDIA](https://img.shields.io/badge/nVIDIA-%2376B900.svg?style=for-the-badge&logo=nVIDIA&logoColor=white)
![VS Code](https://img.shields.io/badge/VSCode-0078D4?style=for-the-badge&logo=visual%20studio%20code&logoColor=white)
![Github](https://img.shields.io/badge/GitHub-100000?style=for-the-badge&logo=github&logoColor=white)

<a href="https://ascl.net/2208.009"><img src="https://img.shields.io/badge/ascl-2208.009-blue.svg?colorY=262255" alt="ascl:2208.009" /></a>

[**Le**](#)ja interpolation for e[**X**](#)ponential [**Int**](#)egrators is a temporal integration package for exponential integrators using the method of polynomial interpolation at Leja points.

Exponential Rosenbrock (EXPRB) and Exponential Propagation Iterative Runge-Kutta (EPIRK) methods use the Leja interpolation method to compute the $\varphi_l(z)$ functions for nonlinear differential equations. For homogenous linear PDEs, one can get the ***exact*** solution (in time) by directly computing the matrix exponential using the functions ``real_Leja_exp`` and/or ``imag_Leja_exp``, whereas for nonhomogenous linear PDEs, one can use ``real_Leja_phi_nl`` and/or ``imag_Leja_phi_nl``. The algorithmic details can be found in the cited literature. 

## Requirements
- Python 3.10 (or later)

## Literature
The publication associated with this code:

Deka, Einkemmer, and Tokman (2022), *LeXInt: Package for Exponential Integrators employing Leja interpolation*, SoftwareX, 21, 101302 <br />
[[DOI]](https://doi.org/10.1016/j.softx.2022.101302) [[arXiv:2208.08269]](https://doi.org/10.48550/arXiv.2208.08269)

Other references:
- Caliari et al. (2014), *Comparison of software for computing the action of the matrix exponential*, BIT Numer. Math., 54, 113 <br />
[[DOI]](https://doi.org/10.1007/s10543-013-0446-0)

- Deka \& Einkemmer (2022), *Efficient adaptive step size control for exponential integrators*, Comput. Math. Appl., 123, 59 <br />
[[DOI]](https://doi.org/10.1016/j.camwa.2022.07.011) [[arXiv:2102.02524]](https://doi.org/10.48550/arXiv.2102.02524)

- Deka \& Einkemmer (2022), *Exponential Integrators for Resistive Magnetohydrodynamics: Matrix-free Leja Interpolation and Efficient Adaptive Time Stepping*, ApJS, 259, 57 <br />
[[DOI]](https://doi.org/10.3847/1538-4365/ac5177) [[arXiv:2108.13622]](https://doi.org/10.48550/arXiv.2108.13622)

- Hochbruck \& Ostermann (2010), *Exponential integrators*, Acta Numer., 19, 209 <br />
[[DOI]](https://doi.org/10.1017/S0962492910000048)

## Future Prospects
We are developing a GPU (NVIDIA CUDA) implementation of the Leja method for exponential integrators which will soon be publicly available.

## Contact
Pranab J. Deka  (<pranab.deka@uibk.ac.at>) <br />
Lukas Einkemmer (<lukas.einkemmer@uibk.ac.at>) <br />
Mayya Tokman  (<mtokman@ucmerced.edu>)

In case of technical issues, please contact Pranab J. Deka.

## Acknowledgements
Alexander Morrigl contributed to the development of the CUDA and C++ versions.
