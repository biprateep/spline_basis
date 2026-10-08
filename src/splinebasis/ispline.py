import numpy as np

from . import _core, _utils
from ._backend import get_backend
from ._basis import _SplineBasis

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
        self._mspline_order = self.order + 1
        self._mspline_knots = _utils.build_knots(self._mspline_order, self.mesh)
        self._mspline_n = len(self._mspline_knots) - self._mspline_order

        self.n = len(self.knots) - self.order
        assert self.n == len(self.mesh) - 2 + self.order

    def __call__(self, x, i, invalid_i="raise"):
        r"""Evaluate spline :math:`I_i` at point(s) x.
        Parameters
        ----------
        x : np.ndarray
            Points at which to evaluate the spline.
        i : int
            Spline member :math:`I_i`, where :math:`1 \le i \le`
            :attr:`Isplines.n`.
        invalid_i : {'raise', 'zero'}
            If `i` is invalid, do we raise an error or return 0?
        Returns
        -------
        array
            The values of the I-spline evaluated at each x.
        Note
        ----
        The spline is defined by the formula given in the
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
        :attr:`Isplines.order`. It is evaluated equivalently as
        :math:`I_i(x) = \sum_{m > i} B_m\left(x \mid k + 1\right)` with
        B-splines :math:`B_m` from de Boor's recursion.
            `Ramsay (1988)`: https://www.jstor.org/stable/2245395
            `Praat manual`: http://www.fon.hum.uva.nl/praat/manual/spline.html
        """
        be = get_backend(x)
        x = _utils.validate_x(be, x, self.lower, self.upper)
        if not _utils.validate_member(i, self.n, invalid_i):
            return be.zeros_like(x)
        return self._evaluate(be, x, derivative=False)[:, i - 1]

    def derivatives(self, x, i, invalid_i="raise"):
        r"""Evaluate first derivative of I spline at each x.
        Parameters
        ----------
        x : 1-D array-like
            Points at which to evaluate the spline.
        i : int
            Same meaning as for :meth:`Isplines.__call__`.
        invalid_i : {'raise', 'zero'}
            If `i` is invalid, do we raise an error or return 0?
        Returns
        -------
        array
            Derivative of I-spline with respect to x, which equals the M-spline
            :math:`M_i\left(x \mid k\right)`.

        """
        be = get_backend(x)
        x = _utils.validate_x(be, x, self.lower, self.upper)
        if not _utils.validate_member(i, self.n, invalid_i):
            return be.zeros_like(x)
        return self._evaluate(be, x, derivative=True)[:, i - 1]

    def evaluate_all(self, x):
        r"""Evaluate all members :math:`I_1, \ldots, I_n` at point(s) x.
        Parameters
        ----------
        x : 1-D array-like
            Points at which to evaluate the splines.
        Returns
        -------
        array
            Array of shape ``(len(x), n)`` whose column `i - 1` is :math:`I_i`.
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
            Array of shape ``(len(x), n)`` whose column `i - 1` is :math:`dI_i/dx`.
        """
        be = get_backend(x)
        return self._evaluate(be, _utils.validate_x(be, x, self.lower, self.upper), derivative=True)

    def _evaluate(self, be, x, derivative):
        return _core.isplines(
            be,
            x,
            be.constant(self.knots, like=x),
            self.order,
            be.constant(self._mspline_knots, like=x),
            derivative=derivative,
        )


class ISplineBasis(_SplineBasis):
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
        Sets :attr:`ISplineBasis.x`. A PyTorch tensor or JAX array selects
        that backend (keeping its dtype and device).
    backend: {None, 'numpy', 'torch', 'jax'}
        Backend for :attr:`ISplineBasis.x` and the basis arrays. Inferred from
        `grid` if None, and NumPy if `grid` is not given.
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
    x: array
        Points at which the spline family is evaluated.
    backend: str
        Name of the backend of :attr:`ISplineBasis.x` and the basis arrays.
    isplines : :class:`Isplines`
        An instance of :class:`Isplines` representing the spline family.
    basis_vectors : array
        The member splines evaluated at the grid points, of shape
        ``(len(x), num_basis)``.
    basis_derivatives : array
        First derivatives of the member splines at the grid points, of shape
        ``(len(x), num_basis)``.

    .. _`Ramsay (1988)`: https://www.jstor.org/stable/2245395
    """

    def _make_family(self, mesh):
        self.isplines = Isplines(order=self.order, mesh=mesh)
        return self.isplines
