# spline_basis

[![Template](https://img.shields.io/badge/Template-LINCC%20Frameworks%20Python%20Project%20Template-brightgreen)](https://lincc-ppt.readthedocs.io/en/latest/)


A python library to generate spline basis functions. Check out examples in the [notebooks](docs/notebooks/).

### Installation:
From PyPi: `pip install spline-basis`

Currently supported basis functions:
1. $\mathrm{M}$-Splines: Non-negative spline functions which integrate to 1 over the support of the basis function.
2. $\mathrm{I}$-Splines: Monotone spline functions defined as the integral of M-splines.



**Notes:**
1. The M/I splines are based on [`Ramsay (1988)`](https://www.jstor.org/stable/2245395)and the [`Praat manual`](http://www.fon.hum.uva.nl/praat/manual/spline.html). This specific implementation is a jit-compiled version of the implementation in the [`dms_variants`](https://jbloomlab.github.io/dms_variants/index.html) python package but with a different API ([`dms_variants` Source Code](https://jbloomlab.github.io/dms_variants/_modules/dms_variants/ispline.html#Isplines)).

TODO:
1. Enable `cahe=True` for `nb.jit`
2. replace asserts with exceptions.


maintainer: Biprateep Dey
contact: `biprateep@pitt.edu`
