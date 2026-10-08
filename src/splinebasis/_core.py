r"""Backend-agnostic evaluation of whole M-spline and I-spline families.

Every function takes a backend `be` from :mod:`splinebasis._backend`, points
`x` (a 1-D float array of that backend) and the knot sequence `knots` (an
array of the same backend and dtype), and returns an array of shape
``(len(x), n)`` whose column `i - 1` is member `i` of the family.

The B-splines are evaluated with de Boor's local recursion: at each point only
the `order` B-splines whose support contains it are non-zero, so the cost is
:math:`O(\mathrm{len}(x)\, k^2)` plus scattering into the dense output.
M-splines and I-splines follow from B-splines (`Ramsay (1988)`_):

.. math::
   M_i(x \mid k) = \frac{k}{t_{i+k} - t_i} B_i(x \mid k), \qquad
   I_i(x \mid k) = \sum_{m > i} B_m(x \mid k + 1), \qquad
   \frac{d I_i(x \mid k)}{dx} = M_i(x \mid k),

where the I-spline uses the knots of order :math:`k + 1` on the same mesh.

.. _`Ramsay (1988)`: https://www.jstor.org/stable/2245395
"""


def _local_bsplines(be, x, knots, order, derivative):
    r"""Non-zero B-splines of `order` (or their derivatives) at each point.

    Returns
    -------
    tuple
        `(columns, mu)`: `mu[r]` is the index of the knot interval
        :math:`t_\mu \le x_r < t_{\mu + 1}` (0-based; the upper end of the mesh
        belongs to the last non-empty interval), and `columns[s][r]` is
        :math:`B_{\mu - k + 1 + s}` at `x[r]` for `s` in `0 .. order - 1`
        (0-based member index).
    """
    n = knots.shape[0] - order
    mu = be.clip(be.searchsorted_right(knots, x) - 1, order - 1, n - 1)

    columns = [be.ones_like(x)]
    previous = None
    for r in range(1, order):
        previous = columns
        # `columns` holds the r B-splines of order r that are non-zero on interval mu
        columns = [be.zeros_like(x) for _ in range(r + 1)]
        for s in range(r):
            i = mu - r + 1 + s
            t_left, t_right = knots[i], knots[i + r]
            weight = previous[s] / (t_right - t_left)
            columns[s] = columns[s] + (t_right - x) * weight
            columns[s + 1] = (x - t_left) * weight

    if not derivative:
        return columns, mu

    if order == 1:
        return [be.zeros_like(x)], mu
    # dB_i(x|k)/dx = (k - 1) [B_i(x|k-1) / (t_{i+k-1} - t_i) - B_{i+1}(x|k-1) / (t_{i+k} - t_{i+1})]
    derivatives = []
    for s in range(order):
        i = mu - order + 1 + s
        value = be.zeros_like(x)
        if s >= 1:
            value = value + previous[s - 1] / (knots[i + order - 1] - knots[i])
        if s <= order - 2:
            value = value - previous[s] / (knots[i + order] - knots[i + 1])
        derivatives.append((order - 1) * value)
    return derivatives, mu


def _scatter(be, x, columns, mu, order, n):
    """Dense `(len(x), n)` array from the local `columns` on knot intervals `mu`."""
    indices = mu[:, None] - order + 1 + be.arange(order, like=x)[None, :]
    return be.scatter_columns(n, indices, be.stack_columns(columns))


def bsplines(be, x, knots, order, derivative=False):
    """All B-splines of `order` on `knots` (or their first derivatives) at `x`."""
    columns, mu = _local_bsplines(be, x, knots, order, derivative)
    return _scatter(be, x, columns, mu, order, knots.shape[0] - order)


def msplines(be, x, knots, order, derivative=False):
    """All M-splines of `order` on `knots` (or their first derivatives) at `x`."""
    columns, mu = _local_bsplines(be, x, knots, order, derivative)
    scaled = []
    for s, column in enumerate(columns):
        i = mu - order + 1 + s
        scaled.append(column * (order / (knots[i + order] - knots[i])))
    return _scatter(be, x, scaled, mu, order, knots.shape[0] - order)


def isplines(be, x, knots, order, mspline_knots, derivative=False):
    """All I-splines of `order` at `x`, or their first derivatives.

    Parameters
    ----------
    knots : array
        Knots of order `order` on the mesh (used for the derivatives).
    mspline_knots : array
        Knots of order `order + 1` on the same mesh.
    """
    if derivative:
        return msplines(be, x, knots, order)
    return be.reverse_cumsum(bsplines(be, x, mspline_knots, order + 1))[:, 1:]
