"""Minimal array backends (NumPy, PyTorch, JAX) for the spline algorithms.

The spline code in :mod:`splinebasis._core` only uses arithmetic operators,
integer indexing and the handful of operations wrapped here, so the same
implementation runs on every backend. PyTorch and JAX are imported lazily and
are only needed when their arrays are passed in (or explicitly requested).
"""

import functools

import numpy as np

BACKENDS = ("numpy", "torch", "jax")


class _NumpyBackend:
    name = "numpy"

    def as_float_array(self, x):
        x = np.asarray(x)
        return x if np.issubdtype(x.dtype, np.floating) else x.astype(np.float64)

    def constant(self, a, like):
        """NumPy array `a` as an array of `like`'s backend, dtype and device."""
        return np.asarray(a, dtype=like.dtype)

    def to_numpy(self, x):
        return np.asarray(x)

    def is_concrete(self, x):
        return True

    def searchsorted_right(self, sorted_sequence, x):
        return np.searchsorted(sorted_sequence, x, side="right")

    def clip(self, a, lower, upper):
        return np.clip(a, lower, upper)

    def ones_like(self, x):
        return np.ones_like(x)

    def zeros_like(self, x):
        return np.zeros_like(x)

    def stack_columns(self, columns):
        return np.stack(columns, axis=1)

    def arange(self, n, like):
        return np.arange(n)

    def scatter_columns(self, n_columns, indices, values):
        """Dense `(len(values), n_columns)` array with `values[r, c]` at column `indices[r, c]`."""
        out = np.zeros((values.shape[0], n_columns), dtype=values.dtype)
        np.put_along_axis(out, indices, values, axis=1)
        return out

    def reverse_cumsum(self, a):
        """Cumulative sum along axis 1 from the last column to the first."""
        return np.flip(np.cumsum(np.flip(a, axis=1), axis=1), axis=1)


class _TorchBackend:
    name = "torch"

    def __init__(self):
        import torch

        self.torch = torch

    def as_float_array(self, x):
        x = self.torch.as_tensor(x)
        return x if x.is_floating_point() else x.to(self.torch.get_default_dtype())

    def constant(self, a, like):
        # torch rejects the negative strides that e.g. np.flip produces
        return self.torch.as_tensor(np.ascontiguousarray(a), dtype=like.dtype, device=like.device)

    def to_numpy(self, x):
        return x.detach().cpu().numpy()

    def is_concrete(self, x):
        return True

    def searchsorted_right(self, sorted_sequence, x):
        return self.torch.searchsorted(sorted_sequence, x.contiguous(), right=True)

    def clip(self, a, lower, upper):
        return self.torch.clamp(a, lower, upper)

    def ones_like(self, x):
        return self.torch.ones_like(x)

    def zeros_like(self, x):
        return self.torch.zeros_like(x)

    def stack_columns(self, columns):
        return self.torch.stack(columns, dim=1)

    def arange(self, n, like):
        return self.torch.arange(n, device=like.device)

    def scatter_columns(self, n_columns, indices, values):
        out = self.torch.zeros((values.shape[0], n_columns), dtype=values.dtype, device=values.device)
        return out.scatter(1, indices, values)

    def reverse_cumsum(self, a):
        return self.torch.flip(self.torch.cumsum(self.torch.flip(a, dims=[1]), dim=1), dims=[1])


class _JaxBackend:
    name = "jax"

    def __init__(self):
        import jax
        import jax.numpy as jnp

        self.jax = jax
        self.jnp = jnp

    def as_float_array(self, x):
        x = self.jnp.asarray(x)
        return x if self.jnp.issubdtype(x.dtype, self.jnp.inexact) else x.astype(self.jnp.result_type(float))

    def constant(self, a, like):
        return self.jnp.asarray(np.asarray(a), dtype=like.dtype)

    def to_numpy(self, x):
        return np.asarray(x)

    def is_concrete(self, x):
        """False inside `jax.jit` / `jax.grad` tracing, where values cannot be inspected."""
        return not isinstance(x, self.jax.core.Tracer)

    def searchsorted_right(self, sorted_sequence, x):
        return self.jnp.searchsorted(sorted_sequence, x, side="right")

    def clip(self, a, lower, upper):
        return self.jnp.clip(a, lower, upper)

    def ones_like(self, x):
        return self.jnp.ones_like(x)

    def zeros_like(self, x):
        return self.jnp.zeros_like(x)

    def stack_columns(self, columns):
        return self.jnp.stack(columns, axis=1)

    def arange(self, n, like):
        return self.jnp.arange(n)

    def scatter_columns(self, n_columns, indices, values):
        rows = self.jnp.arange(values.shape[0])[:, None]
        out = self.jnp.zeros((values.shape[0], n_columns), dtype=values.dtype)
        return out.at[rows, indices].set(values)

    def reverse_cumsum(self, a):
        return self.jnp.flip(self.jnp.cumsum(self.jnp.flip(a, axis=1), axis=1), axis=1)


_BACKEND_CLASSES = {"numpy": _NumpyBackend, "torch": _TorchBackend, "jax": _JaxBackend}


@functools.lru_cache(maxsize=None)
def backend_by_name(name):
    """The backend called `name`, one of :data:`BACKENDS`."""
    if name not in _BACKEND_CLASSES:
        raise ValueError(f"`backend` not one of {BACKENDS}: {name}")
    try:
        return _BACKEND_CLASSES[name]()
    except ImportError as err:
        raise ImportError(
            f"the {name} backend needs {name} installed: `pip install spline-basis[{name}]`"
        ) from err


def _array_backend_name(a):
    """Name of the backend that array `a` belongs to, or None for NumPy and Python objects."""
    module = type(a).__module__.split(".")[0]
    if module == "torch":
        return "torch"
    if module in ("jax", "jaxlib"):
        return "jax"
    return None


def get_backend(*arrays, backend=None):
    """Backend for `arrays`: PyTorch or JAX if any of them is a tensor/array of that library, else NumPy.

    Parameters
    ----------
    *arrays
        Arrays (or Python objects) to infer the backend from.
    backend : {None, 'numpy', 'torch', 'jax'}
        Use this backend instead of inferring it.
    """
    if backend is not None:
        return backend_by_name(backend)
    found = None
    for a in arrays:
        name = _array_backend_name(a)
        if name is None:
            continue
        if found is not None and name != found:
            raise TypeError(f"cannot mix {found} and {name} arrays")
        found = name
    return backend_by_name(found or "numpy")
