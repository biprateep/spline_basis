"""Argument validation shared by the spline families and spline bases."""

import numbers

import numpy as np


def _is_int(value):
    """True for Python and NumPy integers (but not bools)."""
    return isinstance(value, numbers.Integral) and not isinstance(value, (bool, np.bool_))


def _is_real(value):
    """True for Python and NumPy real scalars (but not bools)."""
    return isinstance(value, numbers.Real) and not isinstance(value, (bool, np.bool_))


def validate_order(order):
    """Return `order` as an int, raising `ValueError` if it is not an int >= 1."""
    if not (_is_int(order) and order >= 1):
        raise ValueError(f"`order` not int >= 1: {order}")
    return int(order)


def validate_bounds(lower, upper):
    """Return `lower` and `upper` as floats, raising `ValueError` unless `lower < upper`."""
    if not (_is_real(lower) and _is_real(upper)):
        raise ValueError(f"`lower` and `upper` not int or float: {lower}, {upper}")
    if not lower < upper:
        raise ValueError(f"`lower` not less than `upper`: {lower}, {upper}")
    return float(lower), float(upper)


def validate_sorted_unique(values, name, min_length):
    """Return `values` as a 1-D float array of strictly increasing values."""
    arr = np.array(values, dtype="float")
    if arr.ndim != 1:
        raise ValueError(f"`{name}` not array-like of dimension 1: {values}")
    if len(arr) < min_length:
        raise ValueError(f"`{name}` not length >= {min_length}: {values}")
    if not np.array_equal(arr, np.unique(arr)):
        raise ValueError(f"`{name}` elements not unique and sorted: {values}")
    return arr


def build_mesh(order, lower, upper, num_knots, mesh):
    """Mesh sequence for a spline family, from either `mesh` or `lower`/`upper`/`num_knots`.

    Returns
    -------
    tuple
        `(lower, upper, mesh)`.
    """
    if mesh is None:
        if lower is None or upper is None or num_knots is None:
            raise ValueError("if `mesh` is None, then `lower`, `upper`, and `num_knots` must be specified")
        lower, upper = validate_bounds(lower, upper)
        if not (_is_int(num_knots) and num_knots >= 2 * order):
            raise ValueError(f"`num_knots` not int >= {2 * order}: {num_knots}")
        mesh = np.linspace(lower, upper, int(num_knots) - 2 * order + 2)
    else:
        mesh = validate_sorted_unique(mesh, "mesh", min_length=2)
        lower, upper = mesh[0], mesh[-1]
    return lower, upper, mesh


def build_knots(order, mesh):
    """Knot sequence of a spline family of `order` with the given `mesh`."""
    return np.concatenate([np.full(order, mesh[0]), mesh[1:-1], np.full(order, mesh[-1])])


def build_grid(lower, upper, n_grid, grid):
    """Evaluation grid for a spline basis, from either `grid` or `lower`/`upper`/`n_grid`.

    Returns
    -------
    tuple
        `(lower, upper, x)`.
    """
    if grid is None:
        if lower is None or upper is None or n_grid is None:
            raise ValueError("if `grid` is None, then `lower`, `upper`, and `n_grid` must be specified")
        lower, upper = validate_bounds(lower, upper)
        if not (_is_int(n_grid) and n_grid >= 1):
            raise ValueError(f"`n_grid` not int >= 1: {n_grid}")
        x = np.linspace(lower, upper, int(n_grid))
    else:
        x = validate_sorted_unique(grid, "grid", min_length=2)
        lower, upper = x[0], x[-1]
    return lower, upper, x


def validate_num_basis(num_basis, order):
    """Return `num_basis` as an int; at least `order` members are needed for a mesh of >= 2 points."""
    if not (_is_int(num_basis) and num_basis >= order):
        raise ValueError(f"`num_basis` not int >= {order}: {num_basis}")
    return int(num_basis)


def validate_member(i, n, invalid_i):
    """True if `i` is a member `1 <= i <= n`; False if not and `invalid_i` is 'zero', else `ValueError`."""
    if invalid_i not in ("raise", "zero"):
        raise ValueError(f"invalid `invalid_i` of {invalid_i}")
    if _is_int(i) and 1 <= i <= n:
        return True
    if invalid_i == "raise":
        raise ValueError(f"invalid spline member `i` of {i}")
    return False


def validate_x(be, x, lower, upper):
    """Return `x` as a 1-D float array of backend `be`, raising `ValueError` if it lies outside `[lower, upper]`.

    The range check is skipped while JAX traces a function (`jax.jit`, `jax.grad`), where values are unknown.
    """
    x = be.as_float_array(x)
    if x.ndim != 1:
        raise ValueError("`x` is not array-like of dimension 1")
    if be.is_concrete(x) and bool(((x < lower) | (x > upper)).any()):
        raise ValueError(f"`x` outside {lower} and {upper}: {x}")
    return x


def validate_weights(weights, num_basis, constant):
    """Check that `weights` is a 1-D array of length `num_basis` and `constant` a real scalar.

    `weights` may belong to any backend, and `constant` may be a 0-dimensional array of any backend.
    """
    if getattr(weights, "ndim", None) != 1:
        raise ValueError("`weights` is not array-like of dimension 1")
    if weights.shape[0] != num_basis:
        raise ValueError(f"`weights` not length {num_basis}: {weights}")
    if not (_is_real(constant) or getattr(constant, "ndim", None) == 0):
        raise ValueError(f"`constant` not int or float: {constant}")
