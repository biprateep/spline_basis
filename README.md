# spline_basis

[![Template](https://img.shields.io/badge/Template-LINCC%20Frameworks%20Python%20Project%20Template-brightgreen)](https://lincc-ppt.readthedocs.io/en/latest/)


A python library to generate spline basis functions. Check out examples in the [notebooks](docs/notebooks/).

### Installation:
From PyPi: `pip install spline-basis`

NumPy is the default backend. To use the splines with PyTorch tensors or JAX arrays, install the optional extras:
`pip install "spline-basis[torch]"`, `pip install "spline-basis[jax]"` (or both: `"spline-basis[torch,jax]"`).

### Backends:
Every class accepts NumPy arrays (or lists), PyTorch tensors and JAX arrays, and returns results on the same backend, dtype and device as its inputs:

```python
import torch
from splinebasis import Isplines, ISplineBasis

isplines = Isplines(order=3, lower=0, upper=1, num_knots=10)
x = torch.linspace(0, 1, 100, requires_grad=True)
values = isplines.evaluate_all(x)  # (100, isplines.n) tensor, differentiable w.r.t. x

basis = ISplineBasis(order=3, num_basis=10, lower=0, upper=1, n_grid=100)
weights = torch.rand(10, requires_grad=True)
curve = basis(weights)  # torch tensor, differentiable w.r.t. weights
```

`ISplineBasis`/`MSplineBasis` compute their basis on the backend of `grid` (NumPy by default; `backend="torch"` or `"jax"` overrides it). JAX evaluation works inside `jax.jit`, `jax.grad` and `jax.vmap`.

Currently supported basis functions:
1. $\mathrm{M}$-Splines: Non-negative spline functions which integrate to 1 over the support of the basis function.
2. $\mathrm{I}$-Splines: Monotone spline functions defined as the integral of M-splines.



**Notes:**
1. The M/I splines are based on [`Ramsay (1988)`](https://www.jstor.org/stable/2245395) and the [`Praat manual`](http://www.fon.hum.uva.nl/praat/manual/spline.html). The API follows the implementation in the [`dms_variants`](https://jbloomlab.github.io/dms_variants/index.html) python package ([`dms_variants` Source Code](https://jbloomlab.github.io/dms_variants/_modules/dms_variants/ispline.html#Isplines)), but the splines are evaluated with a vectorized form of de Boor's recursion, written once for all backends: at each point only the `order` non-zero B-splines are computed, so the cost hardly grows with the number of basis functions.


maintainer: Biprateep Dey
contact: `biprateep@pitt.edu`
