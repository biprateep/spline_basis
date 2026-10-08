"""Shared implementation of :class:`MSplineBasis` and :class:`ISplineBasis`."""

import numpy as np

from . import _utils
from ._backend import get_backend


def _convert(a, like):
    """Array `a` (any backend) as an array of `like`'s backend, dtype and device."""
    source, target = get_backend(a), get_backend(like)
    if source is target and source.name == "torch":
        return a.to(dtype=like.dtype, device=like.device)
    return target.constant(source.to_numpy(a), like=like)


class _SplineBasis:
    r"""Weighted sum of a spline family evaluated on a fixed grid.

    Subclasses set :meth:`_make_family`, which builds the spline family
    (:class:`Msplines` or :class:`Isplines`) on a mesh.

    The basis is computed once, on the backend of `grid` (or `backend`, NumPy
    by default). Calls with PyTorch or JAX `weights` return arrays of that
    backend, so the weighted sum is differentiable with respect to `weights`
    and `constant`.
    """

    def __init__(self, order, num_basis, lower=None, upper=None, n_grid=None, grid=None, backend=None):
        """See main class docstring."""
        self.order = _utils.validate_order(order)
        self.num_basis = _utils.validate_num_basis(num_basis, self.order)
        self.num_mesh_points = self.num_basis + 2 - self.order  # num_splines = num_mesh_points + 2 - order

        be = get_backend(grid, backend=backend)
        grid_numpy = (
            None if grid is None else get_backend(grid).to_numpy(get_backend(grid).as_float_array(grid))
        )
        self.lower, self.upper, x = _utils.build_grid(lower, upper, n_grid, grid_numpy)
        if grid is not None and get_backend(grid) is be:
            # keep the dtype and device of the grid the user passed
            self.x = be.as_float_array(grid)
        else:
            self.x = (
                be.as_float_array(x) if be.name == "numpy" else be.constant(x, like=be.as_float_array(0.0))
            )
        self.backend = be.name

        self.mesh = np.linspace(self.lower, self.upper, self.num_mesh_points)
        family = self._make_family(self.mesh)
        self.basis_vectors = family.evaluate_all(self.x)
        self.basis_derivatives = family.derivatives_all(self.x)
        self._converted = {}

    def _make_family(self, mesh):
        raise NotImplementedError

    def _basis_like(self, attribute, weights):
        """`basis_vectors` or `basis_derivatives` on the backend, dtype and device of `weights`."""
        basis = getattr(self, attribute)
        key = (attribute, type(weights).__module__, str(weights.dtype), str(getattr(weights, "device", "")))
        if key not in self._converted:
            self._converted[key] = _convert(basis, like=weights)
        return self._converted[key]

    def _weighted_sum(self, attribute, weights, constant):
        if get_backend(weights).name == "numpy":
            # NumPy or Python weights take the backend and dtype of the basis
            weights = np.asarray(weights, dtype="float")
            _utils.validate_weights(weights, self.num_basis, constant)
            basis = getattr(self, attribute)
            weights = _convert(weights, like=basis)
        else:
            weights = get_backend(weights).as_float_array(weights)
            _utils.validate_weights(weights, self.num_basis, constant)
            basis = self._basis_like(attribute, weights)
        return basis @ weights + constant

    def __call__(self, weights, constant=0.0):
        r"""Weighted sum of spline family .
        Parameters
        ----------
        weights : 1-D array-like
            Weights for each member of the spline family. A PyTorch tensor or
            JAX array gives a result of that backend (on its device and dtype).
        constant : float or 0-dimensional array
            Constant offset to be added to the Spline family.
        Returns
        -------
        array
            The weighted sum for each point in the grid.
        """
        return self._weighted_sum("basis_vectors", weights, constant)

    def derivatives(self, weights, constant=0.0):
        r"""Derivative of the weighted sum of the spline family.
        Parameters
        ----------
        weights : 1-D array-like
            Weights for each member of the spline family. A PyTorch tensor or
            JAX array gives a result of that backend (on its device and dtype).
        constant : float or 0-dimensional array
            Constant offset to be added to the derivative of the spline family.

        Returns
        -------
        array
            Derivative of the weighted sum of the spline family evaluated
            at each point in the grid (with an optional constant offset).
        """
        return self._weighted_sum("basis_derivatives", weights, constant)
