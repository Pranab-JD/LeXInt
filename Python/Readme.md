# Python

![Python](https://img.shields.io/badge/Python-FFD43B?style=for-the-badge&logo=python&logoColor=blue)
![VS Code](https://img.shields.io/badge/VSCode-0078D4?style=for-the-badge&logo=visual%20studio%20code&logoColor=white)
![Github](https://img.shields.io/badge/GitHub-100000?style=for-the-badge&logo=github&logoColor=white)

Test examples for constant and adaptive (or variable) step size implementation for the Leja-based exponential integrators can be found in *Python &rarr; Test &rarr; Constant_test.py* or *Adaptive_test.py*. Problems considered include the Burgers' equation and the Allen-Cahn equation. To run scripts, use the following commands: `python3 Constant_test.py` or `python3 Adaptive_test.py`. To add other problems, simply define the relevant *RHS_function* and the initial condition(s).

## Requirements
- Python 3.10 (or later)

Remarks:
1. It is expected that the rhs function is defined in the following way:

```python
def RHS_function(u):

	### stencil_applied_to_u = *Apply stencil to 'u'* or *Compute matrix-vector products*

	return stencil_applied_to_u
```

   If different stencils are used for different physical phenomena (e.g. centered differences for diffusion and upwind for advection), the two stencils applied to 'u' vector are to be combined together.

2. LeXInt can be used for multidimensional problems, once the state variable(s) is(are) vectorised or flattened.

3. RHS function calls are expected to be the most expensive part of any computation. However, if the RHS function is relatively simple, or if the problem size is small, the computation of the polynomial coefficients using divided differences may become substantial. To avoid unnecessary computation of polynomial coefficients, we set the default number of Leja points (to be used) to 500 in the test problems. If you get the warning *"Warning!! Max. # of Leja points reached without convergence!!"*, consider increasing the number of Leja points to 1000, 2000, etc. or decreasing the step size (dt).
