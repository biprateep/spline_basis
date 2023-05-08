import numba as nb
import numpy as np


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
        if not (isinstance(order, int) and order >= 1):
            raise ValueError(f"`order` not int >= 1: {order}")
        self.order = order

        if mesh is None:
            if lower is None or upper is None or num_knots is None:
                raise ValueError(
                    "if `mesh` is None, then `lower`, `upper`, and " "`num_knots` must be specified"
                )
            if not (isinstance(lower, (int, float)) and isinstance(upper, (int, float))):
                raise ValueError(f"`lower` and `upper` not int or float: {lower}, {upper}")
            if not (isinstance(num_knots, int) and num_knots >= 2 * self.order):
                raise ValueError(f"`num_knots` not int >= {2*self.order}: {num_knots}")
            self.lower = lower
            self.upper = upper
            self.mesh = np.linspace(self.lower, self.upper, num_knots - 2 * self.order + 2)

        if mesh is not None:
            self.mesh = np.array(mesh, dtype="float")
            if self.mesh.ndim != 1:
                raise ValueError(f"`mesh` not array-like of dimension 1: {mesh}")
            if len(self.mesh) < 2:
                raise ValueError(f"`mesh` not length >= 2: {mesh}")
            if not np.array_equal(self.mesh, np.unique(self.mesh)):
                raise ValueError(f"`mesh` elements not unique and sorted: {mesh}")
            self.lower = self.mesh[0]
            self.upper = self.mesh[-1]

        assert self.lower < self.upper

        self.knots = np.array(
            [self.lower] * self.order + list(self.mesh[1:-1]) + [self.upper] * self.order,
            dtype="float",
        )

        self.n = len(self.knots) - self.order
        assert self.n == len(self.mesh) - 2 + self.order

    def __call__(self, x, i, invalid_i="raise"):
        r"""Evaluate spline :math:`M_i` at point(s) x.
        Parameters
        ----------
        x : np.ndarray
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
        if not (isinstance(x, np.ndarray) and x.ndim == 1):
            raise ValueError("`x` is not np.ndarray of dimension 1")
        if (x < self.lower).any() or (x > self.upper).any():
            raise ValueError(f"`x` outside {self.lower} and {self.upper}: {x}")

        return _calculate_M(x=x, i=i, k=self.order, n=self.n, knots=self.knots, invalid_i=invalid_i)

    def derivatives(self, x, i, invalid_i="raise"):
        r"""Evaluate first derivative of spline :math:`M_i` at point(s) x.
        Parameters
        ----------
        x : np.ndarray
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

        if not (isinstance(x, np.ndarray) and x.ndim == 1):
            raise ValueError("`x` is not np.ndarray of dimension 1")
        if (x < self.lower).any() or (x > self.upper).any():
            raise ValueError(f"`x` outside {self.lower} and {self.upper}: {x}")

        return _calculate_dM_dx(x=x, i=i, k=self.order, n=self.n, knots=self.knots, invalid_i=invalid_i)


@nb.jit(nopython=True)
def _ti_le_x_lt_tiplusk(x, ti, tiplusk):
    r"""Indices where :math:`t_i \le x \le t_{i+k}`.
    Parameters
    ----------
    x : np.ndarray
    ti : float
        :math:`t_i`
    tiplusk : float
        :math:`t_{i+k}`
    Returns
    -------
    np.ndarray
        Array of booleans of same length as `x` indicating
        if :math:`t_i \le x \le t_{i+k}`.
    """
    key = (ti, tiplusk)
    val = (ti <= x) & (x < tiplusk)
    return val


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

    boolindex = _ti_le_x_lt_tiplusk(x, ti, tiplusk)
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
        boolindex = _ti_le_x_lt_tiplusk(x, ti, tiplusk)
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
        Sets :attr:`Isplines_total.order`.
    mesh : array-like
        Sets :attr:`Isplines_total.mesh`.
    x : np.ndarray
        Sets :attr:`Isplines_total.x`.
    Attributes
    ----------
    order : int
        See :attr:`Isplines.order`.
    mesh : np.ndarray
        See :attr:`Isplines.mesh`.
    n : int
        See :attr:`Isplines.n`.
    lower : float
        See :attr:`Isplines.lower`.
    upper : float
        See :attr:`Isplines.upper`.

    .. _`Ramsay (1988)`: https://www.jstor.org/stable/2245395
    """

    def __init__(self, order, num_basis, lower=None, upper=None, n_grid=None, grid=None):
        """See main class docstring."""
        if not (isinstance(order, int) and order >= 1):
            raise ValueError(f"`order` not int >= 1: {order}")
        self.order = order
        if not (isinstance(num_basis, int) and num_basis >= self.order - 2):
            raise ValueError(f"`num_basis` not int >= {self.order-2}: {order}")
        self.num_basis = num_basis
        self.num_mesh_points = num_basis + 2 - self.order  # num_splines = num_mesh_points + 2 - order

        if grid is None:
            if lower is None or upper is None or n_grid is None:
                raise ValueError(
                    "if `grid` is None, then `lower`, `upper`, and " "`n_grid` must be specified"
                )
            if not (isinstance(lower, (int, float)) and isinstance(upper, (int, float))):
                raise ValueError(f"`lower` and `upper` not int or float: {lower}, {upper}")
            if not (isinstance(n_grid, int) and n_grid >= 1):
                raise ValueError(f"`n_grid` not int >= 1: {n_grid}")
            self.lower = lower
            self.upper = upper
            self.x = np.linspace(self.lower, self.upper, n_grid)

        if grid is not None:
            self.grid = np.array(grid, dtype="float")
            if self.grid.ndim != 1:
                raise ValueError(f"`grid` not array-like of dimension 1: {grid}")
            if len(self.grid) < 1:
                raise ValueError(f"`grid` not length >= 1: {self.grid}")
            if not np.array_equal(self.grid, np.unique(self.grid)):
                raise ValueError(f"`grid` elements not unique and sorted: {self.grid}")
            self.lower = self.grid[0]
            self.upper = self.grid[-1]
            self.x = self.grid

        assert self.lower < self.upper

        self.mesh = np.linspace(self.lower, self.upper, self.num_mesh_points)
        self.msplines = Msplines(order=self.order, mesh=self.mesh)
        self.basis_vectors = np.array([self.msplines(self.x, i=i + 1) for i in range(self.num_basis)])

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
        if not (isinstance(weights, np.ndarray) and weights.ndim == 1):
            raise ValueError("`weights` is not np.ndarray of dimension 1")
        if len(weights) != self.num_basis:
            raise ValueError(f"`weights` not length {self.num_basis}: {weights}")

        res = constant + np.sum(weights[:, None] * self.basis_vectors, axis=1)

        return res

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
        if not (isinstance(weights, np.ndarray) and weights.ndim == 1):
            raise ValueError("`weights` is not np.ndarray of dimension 1")
        if len(weights) != self.num_basis:
            raise ValueError(f"`weights` not length {self.num_basis}: {weights}")
        if not (isinstance(constant, (int, float))):
            raise ValueError(f"`constant` not int or float: {constant}")

        basis_derivatives = np.array(
            [self.isplines.derivatives(self.x, i=i + 1) for i in range(self.num_basis)]
        )

        res = constant + np.sum(weights[:, None] * basis_derivatives, axis=1)

        return res
