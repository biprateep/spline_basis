import numba as nb
import numpy as np

from . import _utils
from .mspline import _calculate_dM_dx, _calculate_M

__all__ = ["Isplines", "ISplineBasis"]


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
    ):
        """See main class docstring."""
        self.order = _utils.validate_order(order)
        self.lower, self.upper, self.mesh = _utils.build_mesh(self.order, lower, upper, num_knots, mesh)
        self.knots = _utils.build_knots(self.order, self.mesh)
        self._mspline_order = self.order + 1
        self._mspline_knots = _utils.build_knots(self._mspline_order, self.mesh)
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
        x = _utils.validate_x(x, self.lower, self.upper)

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
        x = _utils.validate_x(x, self.lower, self.upper)

        return _calculate_I_or_dI(
            x,
            i,
            k=self.order,
            _mspline_n=self._mspline_n,
            _mspline_knots=self._mspline_knots,
            n=self.n,
            quantity="dI",
        )

    def _evaluate_all(self, x, quantity):
        r"""All members :math:`I_1, \ldots, I_n` (`quantity='I'`) or their derivatives
        (`quantity='dI'`) at `x`, as an array of shape ``(n, len(x))``."""
        x = _utils.validate_x(x, self.lower, self.upper)
        return _calculate_all_I_or_dI(
            x,
            k=self.order,
            _mspline_n=self._mspline_n,
            _mspline_knots=self._mspline_knots,
            n=self.n,
            quantity=quantity,
        )


@nb.jit(nopython=True)
def j(x, _mspline_knots, k):
    r"""np.ndarray: :math:`j` as defined in :meth:`Isplines.__call__`.

    Points at the upper end of the mesh are assigned to the last non-empty
    knot interval, so that :math:`t_j \le x \le t_{j+1}` there.
    """
    return np.minimum(np.searchsorted(_mspline_knots, x, "right"), len(_mspline_knots) - (k + 1))


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
def _calculate_all_I_or_dI(x, k, _mspline_n, _mspline_knots, n, quantity):
    r"""Calculate :meth:`Isplines.__call__` or :meth:`Isplines.derivatives` for all members.
    Parameters
    ----------
    quantity : {'I', 'dI'}
        Calculate :meth:`Isplines.__call__` or :meth:`Isplines.derivatives`?
    Returns
    -------
    np.ndarray
        Array of shape ``(n, len(x))`` whose row `i - 1` is member `i`.
    Note
    ----
    The summation terms are shared by all members, so they are computed once
    and each member's sum is accumulated from :math:`i = n` down to 1.
    """
    if quantity == "I":
        sum_terms = _sum_terms_I(x, k, _mspline_n, _mspline_knots)
        i_lt_jminusk = 1.0
    elif quantity == "dI":
        sum_terms = _sum_terms_dI_dx(x, k, _mspline_n, _mspline_knots)
        i_lt_jminusk = 0.0
    else:
        raise ValueError(f"invalid `quantity` {quantity}")

    _j = j(x, _mspline_knots, k)
    res = np.zeros((n, len(x)))
    for col in range(len(x)):
        jc = _j[col]
        # sum of `sum_terms` over m = i + 1, ..., j; row `m - 1` has the term for `m`
        partial_sum = 0.0
        for i in range(n, 0, -1):
            if i + 1 <= jc:
                partial_sum += sum_terms[i, col]
            if i > jc:
                res[i - 1, col] = 0.0
            elif i < jc - k:
                res[i - 1, col] = i_lt_jminusk
            else:
                res[i - 1, col] = partial_sum

    return res


@nb.jit(nopython=True)
def _calculate_I_or_dI(x, i, k, _mspline_n, _mspline_knots, n, quantity):
    r"""Calculate :meth:`Isplines.__call__` or :meth:`Isplines.derivatives` for member `i`.
    Parameters
    ----------
    i : int
        Same meaning as for :meth:`Isplines.__call__`.
    quantity : {'I', 'dI'}
        Calculate :meth:`Isplines.__call__` or :meth:`Isplines.derivatives`?
    Returns
    -------
    np.ndarray
        The return value of :meth:`Isplines.__call__` or :meth:`Isplines.derivatives`.
    """
    if not (1 <= i <= n):
        raise ValueError(f"invalid spline member `i` of {i}")

    return _calculate_all_I_or_dI(x, k, _mspline_n, _mspline_knots, n, quantity)[i - 1]


class ISplineBasis:
    r"""Evaluate the weighted sum of an I-spline family (see `Ramsay (1988)`_).
    Parameters
    ----------
    order : int
        Sets :attr:`ISplineBasis.order`.
    num_basis: int
        Sets :attr:`ISplineBasis.num_basis`.
    lower: float
        Sets :attr:`ISplineBasis.lower`.
    upper: float
        Sets :attr:`ISplineBasis.upper`.
    n_grid: int
        Number of evenly spaced points in :attr:`ISplineBasis.x`.
    grid: 1-D array-like
        Sets :attr:`ISplineBasis.x`.
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
    x: np.ndarray
        Points at which the spline family is evaluated.
    isplines : :class:`Isplines`
        An instance of :class:`Isplines` representing the spline family.
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
        self.isplines = Isplines(order=self.order, mesh=self.mesh)
        self.basis_vectors = self.isplines._evaluate_all(self.x, "I").T
        self.basis_derivatives = self.isplines._evaluate_all(self.x, "dI").T

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
