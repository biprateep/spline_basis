import numpy as np

from . import _core, _utils
from ._backend import get_backend
from ._basis import _SplineBasis

__all__ = ["Msplines", "MSplineBasis"]


class Msplines:
    r"""Implements M-splines (see `Ramsay (1988)`_).
    Parameters
    ----------
    order : int
        Sets :attr:`Msplines.order`.
    lower : float
        Sets :attr:`Msplines.mesh`.
    upper: float
        Sets :attr:`Msplines.x`.
    num_knots : int
        Sets :attr:`Msplines.knots`. Number of knots for the spline.
    mesh : 1-D array-like
        Sets :attr:`Msplines.mesh`.
    Attributes
    ----------
    order : int
        Order of spline, :math:`k` in notation of `Ramsay (1988)`_.
        Polynomials are of degree :math:`k - 1`.
    lower : float
        Lower end of interval spanned by the splines (first point in mesh).
    upper : float
        Upper end of interval spanned by the splines (last point in mesh).
    mesh : np.ndarray
        Mesh sequence, :math:`\xi_1 < \ldots < \xi_q` in the notation
        of `Ramsay (1988)`_. This class implements **fixed** mesh sequences.
    n : int
        Number of members in spline, denoted as :math:`n` in `Ramsay (1988)`_.
        Related to number of points :math:`q` in the mesh and the order
        :math:`k` by :math:`n = q - 2 + k`.
    knots : np.ndarray
        The knot sequence, :math:`t_1, \ldots, t_{n + k}` in the notation of
        `Ramsay (1988)`_.

    Note
    ----
    `x` may be a NumPy array (or any array-like), a PyTorch tensor or a JAX
    array, and results are returned on the same backend, dtype and device.
    PyTorch and JAX results are differentiable with respect to `x`, and the
    JAX evaluation works inside `jax.jit` (the range check on `x` is then
    skipped).

    .. _`Ramsay (1988)`: https://www.jstor.org/stable/2245395
    """

    def __init__(
        self,
        order: int,
        lower: float = None,
        upper: float = None,
        num_knots: int = None,
        mesh: np.ndarray = None,
    ):
        """See main class docstring."""
        self.order = _utils.validate_order(order)
        self.lower, self.upper, self.mesh = _utils.build_mesh(self.order, lower, upper, num_knots, mesh)
        self.knots = _utils.build_knots(self.order, self.mesh)
        self.n = len(self.knots) - self.order
        assert self.n == len(self.mesh) - 2 + self.order

    def __call__(self, x, i, invalid_i="raise"):
        r"""Evaluate spline :math:`M_i` at point(s) x.
        Parameters
        ----------
        x : 1-D array-like
            Points at which to evaluate the spline.
        i : int
            Spline member :math:`M_i`, where :math:`1 \le i \le`
            :attr:`Msplines.n`.
        invalid_i : {'raise', 'zero'}
            If `i` is invalid, do we raise an error or return 0?
        Returns
        -------
        array
            The values of the M-spline evaluated at each x.

        """
        be = get_backend(x)
        x = _utils.validate_x(be, x, self.lower, self.upper)
        if not _utils.validate_member(i, self.n, invalid_i):
            return be.zeros_like(x)
        return self._evaluate(be, x, derivative=False)[:, i - 1]

    def derivatives(self, x, i, invalid_i="raise"):
        r"""Evaluate first derivative of spline :math:`M_i` at point(s) x.
        Parameters
        ----------
        x : 1-D array-like
            Points at which to evaluate the spline.
        i : int
            Spline member :math:`M_i`, where :math:`1 \le i \le`
            :attr:`Msplines.n`.
        invalid_i : {'raise', 'zero'}
            If `i` is invalid, do we raise an error or return 0?
        Returns
        -------
        array
            The values of the first derivative of M-spline evaluated at each x.

        """
        be = get_backend(x)
        x = _utils.validate_x(be, x, self.lower, self.upper)
        if not _utils.validate_member(i, self.n, invalid_i):
            return be.zeros_like(x)
        return self._evaluate(be, x, derivative=True)[:, i - 1]

    def evaluate_all(self, x):
        r"""Evaluate all members :math:`M_1, \ldots, M_n` at point(s) x.
        Parameters
        ----------
        x : 1-D array-like
            Points at which to evaluate the splines.
        Returns
        -------
        array
            Array of shape ``(len(x), n)`` whose column `i - 1` is :math:`M_i`.
        """
        be = get_backend(x)
        return self._evaluate(be, _utils.validate_x(be, x, self.lower, self.upper), derivative=False)

    def derivatives_all(self, x):
        r"""Evaluate the first derivatives of all members at point(s) x.
        Parameters
        ----------
        x : 1-D array-like
            Points at which to evaluate the derivatives.
        Returns
        -------
        array
            Array of shape ``(len(x), n)`` whose column `i - 1` is :math:`dM_i/dx`.
        """
        be = get_backend(x)
        return self._evaluate(be, _utils.validate_x(be, x, self.lower, self.upper), derivative=True)

    def _evaluate(self, be, x, derivative):
        return _core.msplines(be, x, be.constant(self.knots, like=x), self.order, derivative=derivative)


class MSplineBasis(_SplineBasis):
    r"""Evaluate the weighted sum of an M-spline family (see `Ramsay (1988)`_).
    Parameters
    ----------
    order : int
        Sets :attr:`MSplineBasis.order`.
    num_basis: int
        Sets :attr:`MSplineBasis.num_basis`.
    lower: float
        Sets :attr:`MSplineBasis.lower`.
    upper: float
        Sets :attr:`MSplineBasis.upper`.
    n_grid: int
        Number of evenly spaced points in :attr:`MSplineBasis.x`.
    grid: 1-D array-like
        Sets :attr:`MSplineBasis.x`. A PyTorch tensor or JAX array selects
        that backend (keeping its dtype and device).
    backend: {None, 'numpy', 'torch', 'jax'}
        Backend for :attr:`MSplineBasis.x` and the basis arrays. Inferred from
        `grid` if None, and NumPy if `grid` is not given.
    Attributes
    ----------
    order : int
        Order of spline, :math:`k` in notation of `Ramsay (1988)`_.
        Polynomials are of degree :math:`k - 1`.
    num_basis: int
        Number of members in spline family, denoted as :math:`n` in
        `Ramsay (1988)`_. Related to number of points :math:`q` in the mesh
        and the order :math:`k` by :math:`n = q - 2 + k`.
    lower: float
        Lower end of interval spanned by the splines (first point in mesh).
    upper: float
        Upper end of interval spanned by the splines (last point in mesh).
    x: array
        Points at which the spline family is evaluated.
    backend: str
        Name of the backend of :attr:`MSplineBasis.x` and the basis arrays.
    msplines : :class:`Msplines`
        An instance of :class:`Msplines` representing the spline family.
    basis_vectors : array
        The member splines evaluated at the grid points, of shape
        ``(len(x), num_basis)``.
    basis_derivatives : array
        First derivatives of the member splines at the grid points, of shape
        ``(len(x), num_basis)``.

    .. _`Ramsay (1988)`: https://www.jstor.org/stable/2245395
    """

    def _make_family(self, mesh):
        self.msplines = Msplines(order=self.order, mesh=mesh)
        return self.msplines
