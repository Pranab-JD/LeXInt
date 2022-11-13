# Exponential Integrators (Leja Interpolation)

Remarks:
1. It is expected that the rhs function is defined in the following way:

```python
def RHS_function(u):

	### stencil_applied_to_u = Apply stencil to 'u'

	return stencil_applied_to_u
```

If different stencils are used for different physical phenomena (e.g. centered differences for diffusion and upwind for advection), the two stencils applied to 'u' vector are to be combined together.

2. LeXInt can be used for multidimensional problems, once vectorised.

3. RHS function calls are expected to be the most expensive part of any computation. However, if the RHS function is relatively simple, or if the problem size is small, the computation of the polynomial coefficients using divided differences may become substantial. To avoid unnecessary computation of polynomial coefficients, we set the (default) number of Leja points (to be used) to 500 in the test problems. If you get the wanring *"Warning!! Max. # of Leja points reached without convergence!!"*, consider increasing the default number to 1000, 2000, etc.
