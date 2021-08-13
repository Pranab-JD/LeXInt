### Exponential Integrators (Leja Interpolation)

It is expected that the rhs function is defined in the following way:

```python
def RHS_func(u):

	### stencil_applied_to_u = Apply stencil to 'u'
	
	return stencil_applied_to_u
```
If different stencils are used for different physical phenomena (e.g. centered differences for diffusion and upwind for advection), the two stencils applied to 'u' vector are to be combined together.
