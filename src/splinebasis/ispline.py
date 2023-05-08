import numba as nb
import numpy as np

from .mspline import Msplines, _calculate_dM_dx, _calculate_M


class Isplines:
    r"""Implements I-splines (see `Ramsay (1988)`_).
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
        Order of spline, :math:`k` in notation of `Ramsay (1988)`_. Note that
        the degree of the I-spline is equal to :math:`k`, while the
        associated M-spline has order :math:`k` but degree :math:`k - 1`.
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


    .. _`Ramsay (1988)`: https://www.jstor.org/stable/2245395
    """

    def __init__(
        self,
        order: int,
        lower: float = None,
        upper: float = None,
        num_knots: int = None,
        mesh: np.ndarray = None,
    ) -> np.ndarray:
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
        self._mspline_order = self.order + 1
        self._mspline_knots = np.array(
            [self.lower] * self._mspline_order + list(self.mesh[1:-1]) + [self.upper] * self._mspline_order,
            dtype="float",
        )
        self._mspline_n = len(self._mspline_knots) - self._mspline_order

        self.n = len(self.knots) - self.order
        assert self.n == len(self.mesh) - 2 + self.order

    def __call__(self, x, i):
        r"""Evaluate spline :math:`I_i` at point(s) x.
        Parameters
        ----------
        x : np.ndarray
            Points at which to evaluate the spline.
        i : int
            Spline member :math:`I_i`, where :math:`1 \le i \le`
            :attr:`Isplines.n`.
        Returns
        -------
        np.ndarray
            The values of the I-spline evaluated at each x.
        Note
        ----
        The spline is evaluated using the formula given in the
        `Praat manual`_, which corrects some errors in the formula
        provided by `Ramsay (1988)`_:
        .. math::
           I_i\left(x\right)
           =
           \begin{cases}
           0 & \rm{if\;} i > j, \\
           1 & \rm{if\;} i < j - k, \\
           \sum_{m=i+1}^j \left(t_{m+k+1} - t_m\right)
                          M_m\left(x \mid k + 1\right) / \left(k + 1 \right)
             & \rm{otherwise},
           \end{cases}
        where :math:`j` is the index such that :math:`t_j \le x < t_{j+1}`
        (the :math:`\left\{t_j\right\}` are the :attr:`Msplines.knots` for a
        M-spline of order :math:`k + 1`) and :math:`k` is
        :attr:`Isplines.order`.
            `Ramsay (1988)`: https://www.jstor.org/stable/2245395
            `Praat manual`: http://www.fon.hum.uva.nl/praat/manual/spline.html
        """

        if not (isinstance(x, np.ndarray) and x.ndim == 1):
            raise ValueError("`x` is not np.ndarray of dimension 1")
        if (x < self.lower).any() or (x > self.upper).any():
            raise ValueError(f"`x` outside {self.lower} and {self.upper}: {x}")

        return _calculate_I_or_dI(
            x,
            i,
            k=self.order,
            _mspline_n=self._mspline_n,
            _mspline_knots=self._mspline_knots,
            n=self.n,
            quantity="I",
        )

    def derivatives(self, x, i):
        r"""Evaluate first derivative of I spline at each x.
        Parameters
        ----------
        x : np.ndarray
            Points at which to evaluate the spline.
        i : int
            Same meaning as for :meth:`Isplines.I`.
        Returns
        -------
        np.ndarray
            Derivative of I-spline with respect to x.

        """
        if not (isinstance(x, np.ndarray) and x.ndim == 1):
            raise ValueError("`x` is not np.ndarray of dimension 1")
        if (x < self.lower).any() or (x > self.upper).any():
            raise ValueError(f"`x` outside {self.lower} and {self.upper}: {x}")

        return _calculate_I_or_dI(
            x,
            i,
            k=self.order,
            _mspline_n=self._mspline_n,
            _mspline_knots=self._mspline_knots,
            n=self.n,
            quantity="dI",
        )


@nb.jit(nopython=True)
def j(x, _mspline_knots):
    r"""np.ndarray: :math:`j` as defined in :meth:`Isplines.I`."""

    _j = np.searchsorted(_mspline_knots, x, "right")
    # assert (1 <= _j).all() and (_j <= len(knots)).all()
    # assert x.shape == _j.shape
    return _j


@nb.jit(nopython=True)
def _sum_terms_I(
    x,
    k,
    _mspline_n,
    _mspline_knots,
):
    r"""np.ndarray: sum terms for :meth:`Isplines.I`.
    Row `m - 1` has summation term for `m`.
    """
    _sum_terms_I_val = np.zeros((_mspline_n, len(x)))

    for m in range(1, _mspline_n + 1):
        _sum_terms_I_val[m - 1, :] = (
            (_mspline_knots[m + k] - _mspline_knots[m - 1])
            * _calculate_M(x, m, k + 1, _mspline_n, _mspline_knots)
            / (k + 1)
        )

    # _sum_terms_I_val = np.vstack(
    #     [
    #         (_mspline_knots[m + k] - _mspline_knots[m - 1])
    #         * _calculate_M(x, m, k + 1, _mspline_n, _mspline_knots)
    #         / (k + 1)
    #         for m in range(1, _mspline_n + 1)
    #     ]
    # )
    # assert _sum_terms_I_val.shape == (_mspline_n, len(x))
    return _sum_terms_I_val


@nb.jit(nopython=True)
def _sum_terms_dI_dx(x, k, _mspline_n, _mspline_knots):
    r"""np.ndarray: sum terms for :meth:`Isplines.dI_dx`.
    Row `m - 1` has summation term for `m`.
    """
    _sum_terms_dI_dx_val = np.zeros((_mspline_n, len(x)))
    for m in range(1, _mspline_n + 1):
        _sum_terms_dI_dx_val[m - 1, :] = (
            (_mspline_knots[m + k] - _mspline_knots[m - 1])
            * _calculate_dM_dx(x, m, k + 1, _mspline_n, _mspline_knots)
            / (k + 1)
        )

    # _sum_terms_dI_dx_val = np.array(
    #     [
    #         (_mspline_knots[m + k] - _mspline_knots[m - 1])
    #         * _calculate_dM_dx(x, m, k + 1, _mspline_n, _mspline_knots)
    #         / (k + 1)
    #         for m in range(1, _mspline_n + 1)
    #     ]
    # )
    # assert _sum_terms_dI_dx_val.shape == (_mspline_n, len(x))
    return _sum_terms_dI_dx_val


@nb.jit(nopython=True)
def _calculate_I_or_dI(x, i, k, _mspline_n, _mspline_knots, n, quantity):
    r"""Calculate :meth:`Isplines.I` or :meth:`Isplines.dI_dx`.
    Parameters
    ----------
    i : int
        Same meaning as for :meth:`Isplines.I`.
    quantity : {'I', 'dI'}
        Calculate :meth:`Isplines.I` or :meth:`Isplines.dI_dx`?
    Returns
    -------
    np.ndarray
        The return value of :meth:`Isplines.I` or :meth:`Isplines.dI_dx`.
    Note
    ----
    Most calculations for :meth:`Isplines.I` and :meth:`Isplines.dI_dx`
    are the same, so this method implements both.
    """
    if quantity == "I":
        sum_terms = _sum_terms_I(
            x,
            k,
            _mspline_n,
            _mspline_knots,
        )
        i_lt_jminusk = np.zeros_like(x) + 1.0
        # i_lt_jminusk = 2.0
    elif quantity == "dI":
        sum_terms = _sum_terms_dI_dx(
            x,
            k,
            _mspline_n,
            _mspline_knots,
        )
        i_lt_jminusk = np.zeros_like(x)
    else:
        raise ValueError(f"invalid `quantity` {quantity}")

    if not (1 <= i <= n):
        raise ValueError(f"invalid spline member `i` of {i}")

    # create `binary_terms` where entry (m - 1, x) is 1 if and only if
    # the corresponding `sum_terms` entry is part of the sum.

    binary_terms = np.zeros_like(sum_terms)

    for m in range(1, _mspline_n + 1):
        if m < i + 1:
            binary_terms[m - 1, :] = np.zeros_like(x)
        else:
            binary_terms[m - 1, :] = (m <= j(x, _mspline_knots)).astype(np.float64)

    # binary_terms = np.array(
    #     [
    #         np.zeros(len(x)) if m < i + 1 else (m <= j(x, _mspline_knots)).astype(int)
    #         for m in range(1, _mspline_n + 1)
    #     ]
    # )
    # assert binary_terms.shape == sum_terms.shape

    # compute sums from `sum_terms` and `binary_terms`
    sums = np.sum(sum_terms * binary_terms, axis=0)
    # assert sums.shape == x.shape

    # return value with sums, 0, or 1
    res = np.where(
        i > j(x, _mspline_knots), np.zeros_like(x), np.where(i < j(x, _mspline_knots) - k, i_lt_jminusk, sums)
    )

    return res


class ISplineBasis:
    r"""Evaluate the weighted sum of an I-spline family (see `Ramsay (1988)`_).
    Parameters
    ----------
    num_basis, lower=None, upper=None, n_grid=None, grid=None
    order : int
        Sets :attr:`ISplineBasis.order`.
    num_basis: int
        Sets :attr:`ISplineBasis.num_basis`.
    lower: float
        Sets :attr:`ISplineBasis.lower`.
    upper: float
        Sets :attr:`ISplineBasis.upper`.
    n_grid: int
        Sets :attr:`ISplineBasis.n_grid`.
    grid: 1-D array-like
        Sets :attr:`ISplineBasis.grid`.
    Attributes
    ----------
    order : int
        Order of spline, :math:`k` in notation of `Ramsay (1988)`_. Note that
        the degree of the I-spline is equal to :math:`k`, while the
        associated M-spline has order :math:`k` but degree :math:`k - 1`.
    num_basis: int
        Number of members in spline family, denoted as :math:`n` in
        `Ramsay (1988)`_. Related to number of points :math:`q` in the mesh
        and the order :math:`k` by :math:`n = q - 2 + k`.
    lower: float
        Lower end of interval spanned by the splines (first point in mesh).
    upper: float
        Upper end of interval spanned by the splines (last point in mesh).
    n_grid: int
        Number of points at which to evaluate the spline family.
    grid: 1-D array-like
        Points at which to evaluate the spline family.
    isplines : :class:`Isplines`
        An instance of :class:`Isplines` representing the spline family.
    basis_vectors : np.ndarray
        The member splines evaluated at the grid points.
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
        self.isplines = Isplines(order=self.order, mesh=self.mesh)
        self.basis_vectors = np.array([self.isplines(self.x, i=i + 1) for i in range(self.num_basis)]).T

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
            :math:`I_{\rm{total}}` for each point in the grid.
        """
        if not (isinstance(weights, np.ndarray) and weights.ndim == 1):
            raise ValueError("`weights` is not np.ndarray of dimension 1")
        if len(weights) != self.num_basis:
            raise ValueError(f"`weights` not length {self.num_basis}: {weights}")

        res = constant + np.sum(weights[None, :] * self.basis_vectors, axis=1)

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
        ).T

        res = constant + np.sum(weights[None, :] * basis_derivatives, axis=1)

        return res
