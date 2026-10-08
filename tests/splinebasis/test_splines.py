import numpy as np
import pytest

from splinebasis import ISplineBasis, Isplines, MSplineBasis, Msplines

MESH = [0.0, 0.2, 0.5, 0.7, 1.0]
ORDERS = [1, 2, 3, 4]


def _central_difference(f, x, h=1e-6):
    return (f(x + h) - f(x - h)) / (2 * h)


def _trapezoid(y, x):
    # np.trapz was renamed np.trapezoid in NumPy 2, so integrate by hand
    return np.sum((y[1:] + y[:-1]) * np.diff(x)) / 2


@pytest.mark.parametrize("order", ORDERS)
def test_msplines_integrate_to_one(order):
    """Each M-spline is a density over its support."""
    msplines = Msplines(order, mesh=MESH)
    x = np.linspace(0, 1, 200001)
    for i in range(1, msplines.n + 1):
        np.testing.assert_allclose(_trapezoid(msplines(x, i), x), 1.0, atol=1e-4)


@pytest.mark.parametrize("order", ORDERS)
def test_msplines_continuous_at_upper(order):
    """M-splines take their left limit at the upper end of the mesh rather than dropping to 0."""
    msplines = Msplines(order, mesh=MESH)
    for i in range(1, msplines.n + 1):
        at_upper = msplines(np.array([1.0]), i)[0]
        near_upper = msplines(np.array([1.0 - 1e-9]), i)[0]
        np.testing.assert_allclose(at_upper, near_upper, rtol=1e-6, atol=1e-6)
    assert msplines(np.array([1.0]), msplines.n)[0] > 0


@pytest.mark.parametrize("order", [2, 3, 4])
def test_mspline_derivatives_match_finite_differences(order):
    msplines = Msplines(order, mesh=MESH)
    x = np.linspace(0.01, 0.99, 50)  # odd hundredths, so no point sits on a mesh point (kink)
    for i in range(1, msplines.n + 1):
        expected = _central_difference(lambda y: msplines(y, i), x)
        np.testing.assert_allclose(msplines.derivatives(x, i), expected, rtol=1e-4, atol=1e-4)


@pytest.mark.parametrize("order", ORDERS)
def test_isplines_monotone_from_zero_to_one(order):
    isplines = Isplines(order, mesh=MESH)
    x = np.linspace(0, 1, 501)
    for i in range(1, isplines.n + 1):
        values = isplines(x, i)
        assert np.all(np.diff(values) >= -1e-12)
        np.testing.assert_allclose(values[[0, -1]], [0.0, 1.0], atol=1e-12)


@pytest.mark.parametrize("order", ORDERS)
def test_isplines_are_integrals_of_msplines(order):
    """dI_i/dx = M_i(x | k), and I_i is the running integral of M_i."""
    isplines = Isplines(order, mesh=MESH)
    msplines = Msplines(order, mesh=MESH)
    x = np.linspace(0, 1, 100001)
    for i in range(1, isplines.n + 1):
        np.testing.assert_allclose(isplines.derivatives(x, i), msplines(x, i), atol=1e-10)
        running_integral = np.concatenate(
            [[0.0], np.cumsum(np.diff(x) * (msplines(x, i)[1:] + msplines(x, i)[:-1]) / 2)]
        )
        np.testing.assert_allclose(isplines(x, i), running_integral, atol=1e-4)


def test_integer_x_accepted():
    msplines = Msplines(3, mesh=[0, 1, 2])
    isplines = Isplines(3, mesh=[0, 1, 2])
    np.testing.assert_allclose(msplines(np.array([0, 1, 2]), 2), msplines(np.array([0.0, 1.0, 2.0]), 2))
    np.testing.assert_allclose(isplines(np.array([0, 1, 2]), 2), isplines(np.array([0.0, 1.0, 2.0]), 2))


def test_invalid_member():
    msplines = Msplines(3, mesh=MESH)
    with pytest.raises(ValueError):
        msplines(np.array([0.5]), 0)
    np.testing.assert_array_equal(msplines(np.array([0.5]), msplines.n + 1, invalid_i="zero"), [0.0])
    with pytest.raises(ValueError):
        Isplines(3, mesh=MESH)(np.array([0.5]), 0)


def test_x_outside_range_raises():
    with pytest.raises(ValueError):
        Msplines(3, mesh=MESH)(np.array([1.5]), 1)
    with pytest.raises(ValueError):
        Isplines(3, mesh=MESH).derivatives(np.array([-0.1]), 1)


@pytest.mark.parametrize("basis_cls, family", [(MSplineBasis, "msplines"), (ISplineBasis, "isplines")])
def test_basis_shapes_and_values(basis_cls, family):
    basis = basis_cls(order=3, num_basis=6, lower=0, upper=1, n_grid=11)
    splines = getattr(basis, family)
    assert basis.basis_vectors.shape == (11, 6)
    assert basis.basis_derivatives.shape == (11, 6)

    weights = np.arange(1.0, 7.0)
    expected = sum(w * splines(basis.x, i + 1) for i, w in enumerate(weights))
    expected_derivatives = sum(w * splines.derivatives(basis.x, i + 1) for i, w in enumerate(weights))
    np.testing.assert_allclose(basis(weights, constant=2.0), 2.0 + expected, atol=1e-10)
    np.testing.assert_allclose(basis.derivatives(weights), expected_derivatives, atol=1e-10)


def test_isplinebasis_grid():
    grid = np.linspace(-5, 5, 20)
    basis = ISplineBasis(order=3, num_basis=8, grid=grid)
    np.testing.assert_array_equal(basis.x, grid)
    np.testing.assert_allclose(basis(np.ones(8)), basis.basis_vectors.sum(axis=1))


@pytest.mark.parametrize("basis_cls", [MSplineBasis, ISplineBasis])
def test_basis_accepts_numpy_scalars(basis_cls):
    basis = basis_cls(np.int64(3), np.int64(5), np.float32(0), np.float64(1), np.int64(11))
    assert basis.basis_vectors.shape == (11, 5)


@pytest.mark.parametrize("basis_cls", [MSplineBasis, ISplineBasis])
@pytest.mark.parametrize(
    "kwargs",
    [
        dict(order=3, num_basis=2, lower=0, upper=1, n_grid=11),
        dict(order=0, num_basis=2, lower=0, upper=1, n_grid=11),
        dict(order=3, num_basis=5, lower=1, upper=0, n_grid=11),
        dict(order=3, num_basis=5, lower=0, upper=1),
        dict(order=3, num_basis=5, grid=[0.0, 0.5, 0.2]),
    ],
)
def test_basis_invalid_arguments(basis_cls, kwargs):
    with pytest.raises(ValueError):
        basis_cls(**kwargs)


def test_basis_invalid_weights():
    basis = ISplineBasis(order=3, num_basis=5, lower=0, upper=1, n_grid=11)
    with pytest.raises(ValueError):
        basis(np.ones(4))
    with pytest.raises(ValueError):
        basis.derivatives(np.ones(5), constant="1")
