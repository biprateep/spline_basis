import numba as nb
import numpy as np

from . import _utils

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

        `Ramsay (1988)`: https://www.jstor.org/stable/2245395
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
        np.ndarray
            The values of the M-spline evaluated at each x.

        """
        x = _utils.validate_x(x, self.lower, self.upper)

        return _calculate_M(x=x, i=i, k=self.order, n=self.n, knots=self.knots, invalid_i=invalid_i)

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
        np.ndarray
            The values of the first derivative of M-spline evaluated at each x.

        """
        x = _utils.validate_x(x, self.lower, self.upper)

        return _calculate_dM_dx(x=x, i=i, k=self.order, n=self.n, knots=self.knots, invalid_i=invalid_i)


@nb.jit(nopython=True)
def _ti_le_x_lt_tiplusk(x, ti, tiplusk, last):
    r"""Indices where :math:`t_i \le x < t_{i+k}`.
    Parameters
    ----------
    x : np.ndarray
    ti : float
        :math:`t_i`
    tiplusk : float
        :math:`t_{i+k}`
    last : float
        Last knot. If :math:`t_{i+k}` is the last knot the interval is closed,
        :math:`t_i \le x \le t_{i+k}`, so the splines do not vanish at the
        upper end of the mesh.
    Returns
    -------
    np.ndarray
        Array of booleans of same length as `x` indicating
        if `x` is in the support interval.
    """
    if tiplusk == last:
        return (ti <= x) & (x <= tiplusk)
    return (ti <= x) & (x < tiplusk)


@nb.jit(nopython=True)
def _calculate_M(x, i, k, n, knots, invalid_i="raise"):
    r"""Calculate M-splines at points `x` recursively."""
    if not (1 <= i <= n):
        if invalid_i == "raise":
            raise ValueError(f"invalid spline member `i` of {i}")
        elif invalid_i == "zero":
            return np.zeros_like(x)
        else:
            raise ValueError(f"invalid `invalid_i` of {invalid_i}")

    tiplusk = knots[i + k - 1]
    ti = knots[i - 1]
    if tiplusk == ti:
        return np.zeros_like(x)

    boolindex = _ti_le_x_lt_tiplusk(x, ti, tiplusk, knots[-1])
    if k == 1:
        values = 1.0 / (tiplusk - ti)
        res = np.where(boolindex, values, np.zeros_like(values))
        return res
    else:
        assert k > 1

        values = (
            k
            * (
                (x - ti) * _calculate_M(x, i, k - 1, n, knots)
                + (tiplusk - x) * _calculate_M(x, i + 1, k - 1, n, knots, invalid_i="zero")
            )
            / ((float(k) - 1) * (tiplusk - ti))
        )

        res = np.where(boolindex, values, np.zeros_like(values))

        return res


@nb.jit(nopython=True)
def _calculate_dM_dx(x, i, k, n, knots, invalid_i="raise"):
    r"""Calculate the derivatives of M-splines at points `x ` recursively"""
    if not (1 <= i <= n):
        if invalid_i == "raise":
            raise ValueError(f"invalid spline member `i` of {i}")
        elif invalid_i == "zero":
            return np.zeros_like(x)
        else:
            raise ValueError(f"invalid `invalid_i` of {invalid_i}")

    tiplusk = knots[i + k - 1]
    ti = knots[i - 1]
    if tiplusk == ti or k == 1:
        return np.zeros_like(x)
    else:
        assert k > 1
        boolindex = _ti_le_x_lt_tiplusk(x, ti, tiplusk, knots[-1])
        values = (
            k
            * (
                (x - ti) * _calculate_dM_dx(x, i, k - 1, n, knots)
                + _calculate_M(x, i, k - 1, n, knots)
                + (tiplusk - x) * _calculate_dM_dx(x, i + 1, k - 1, n, knots, invalid_i="zero")
                - _calculate_M(x, i + 1, k - 1, n, knots, invalid_i="zero")
            )
            / ((k - 1) * (tiplusk - ti))
        )

        res = np.where(
            boolindex,
            values,
            np.zeros_like(x),
        )

        return res


class MSplineBasis:
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
        Sets :attr:`MSplineBasis.x`.
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
    x: np.ndarray
        Points at which the spline family is evaluated.
    msplines : :class:`Msplines`
        An instance of :class:`Msplines` representing the spline family.
    basis_vectors : np.ndarray
        The member splines evaluated at the grid points, of shape
        ``(len(x), num_basis)``.
    basis_derivatives : np.ndarray
        First derivatives of the member splines at the grid points, of shape
        ``(len(x), num_basis)``.

    .. _`Ramsay (1988)`: https://www.jstor.org/stable/2245395
    """

    def __init__(self, order, num_basis, lower=None, upper=None, n_grid=None, grid=None):
        """See main class docstring."""
        self.order = _utils.validate_order(order)
        self.num_basis = _utils.validate_num_basis(num_basis, self.order)
        self.num_mesh_points = self.num_basis + 2 - self.order  # num_splines = num_mesh_points + 2 - order
        self.lower, self.upper, self.x = _utils.build_grid(lower, upper, n_grid, grid)

        self.mesh = np.linspace(self.lower, self.upper, self.num_mesh_points)
        self.msplines = Msplines(order=self.order, mesh=self.mesh)
        self.basis_vectors = np.array([self.msplines(self.x, i=i + 1) for i in range(self.num_basis)]).T
        self.basis_derivatives = np.array(
            [self.msplines.derivatives(self.x, i=i + 1) for i in range(self.num_basis)]
        ).T

    def __call__(self, weights, constant=0.0):
        r"""Weighted sum of spline family .
        Parameters
        ----------
        weights : array-like
            Weights for each member of the spline family.
        constant : float
            Constant offset to be added to the Spline family.
        Returns
        -------
        np.ndarray
            :math:`M_{\rm{total}}` for each point in the grid.
        """
        weights = _utils.validate_weights(weights, self.num_basis, constant)
        return constant + self.basis_vectors @ weights

    def derivatives(self, weights, constant=0.0):
        r"""Derivative of the weighted sum of the spline family.
        Parameters
        ----------
        weights : array-like
            Weights for each member of the spline family.
        constant : float
            Constant offset to be added to the derivative of the spline family.

        Returns
        -------
        np.ndarray
            Derivative of the weighted sum of the spline family evaluated
            at each point in the grid (with an optional constant offset).
        """
        weights = _utils.validate_weights(weights, self.num_basis, constant)
        return constant + self.basis_derivatives @ weights
