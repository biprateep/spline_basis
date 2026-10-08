"""PyTorch and JAX backends agree with NumPy and support autodiff; skipped if not installed."""

import numpy as np
import pytest

from splinebasis import ISplineBasis, Isplines, MSplineBasis, Msplines

MESH = [0.0, 0.2, 0.5, 0.7, 1.0]
X = np.linspace(0, 1, 41)


@pytest.fixture(params=["torch", "jax"])
def backend(request):
    """(name, to_array, to_numpy) for a backend, in float64."""
    if request.param == "torch":
        torch = pytest.importorskip("torch")
        return "torch", lambda a: torch.as_tensor(np.asarray(a), dtype=torch.float64), lambda a: a.numpy()
    jax = pytest.importorskip("jax")
    jax.config.update("jax_enable_x64", True)
    import jax.numpy as jnp

    return "jax", lambda a: jnp.asarray(np.asarray(a), dtype=jnp.float64), np.asarray


@pytest.mark.parametrize("family_cls", [Msplines, Isplines])
@pytest.mark.parametrize("order", [1, 2, 3, 4])
def test_family_matches_numpy(backend, family_cls, order):
    name, to_array, to_numpy = backend
    family = family_cls(order, mesh=MESH)
    x = to_array(X)
    for method in ("evaluate_all", "derivatives_all"):
        result = getattr(family, method)(x)
        assert type(result).__module__.split(".")[0] in (name, "jaxlib")
        np.testing.assert_allclose(to_numpy(result), getattr(family, method)(X), atol=1e-12)
    np.testing.assert_allclose(to_numpy(family(x, 2)), family(X, 2), atol=1e-12)
    np.testing.assert_allclose(to_numpy(family.derivatives(x, 2)), family.derivatives(X, 2), atol=1e-12)


@pytest.mark.parametrize("basis_cls", [MSplineBasis, ISplineBasis])
def test_basis_weights_select_backend(backend, basis_cls):
    name, to_array, to_numpy = backend
    basis = basis_cls(order=3, num_basis=6, lower=0, upper=1, n_grid=11)
    weights = np.arange(1.0, 7.0)
    result = basis(to_array(weights), constant=to_array(2.0))
    assert type(result).__module__.split(".")[0] in (name, "jaxlib")
    np.testing.assert_allclose(to_numpy(result), basis(weights, constant=2.0), atol=1e-12)
    np.testing.assert_allclose(to_numpy(basis.derivatives(to_array(weights))), basis.derivatives(weights))


@pytest.mark.parametrize("basis_cls", [MSplineBasis, ISplineBasis])
def test_basis_backend_from_grid_and_override(backend, basis_cls):
    name, to_array, to_numpy = backend
    reference = basis_cls(order=3, num_basis=6, grid=X)
    from_grid = basis_cls(order=3, num_basis=6, grid=to_array(X))
    assert from_grid.backend == name
    np.testing.assert_allclose(to_numpy(from_grid.basis_vectors), reference.basis_vectors, atol=1e-12)
    # NumPy weights on a non-NumPy basis give a result on the basis' backend
    assert type(from_grid(np.ones(6))).__module__.split(".")[0] in (name, "jaxlib")

    override = basis_cls(order=3, num_basis=6, lower=0, upper=1, n_grid=41, backend=name)
    assert override.backend == name
    # built from Python bounds, so in the default float dtype of the backend (float32 for torch)
    np.testing.assert_allclose(
        to_numpy(override.basis_vectors), reference.basis_vectors, rtol=1e-5, atol=1e-6
    )


def test_torch_autograd():
    torch = pytest.importorskip("torch")
    isplines = Isplines(3, mesh=MESH)
    x = torch.linspace(0.01, 0.99, 25, dtype=torch.float64, requires_grad=True)
    isplines.evaluate_all(x)[:, 2].sum().backward()
    np.testing.assert_allclose(x.grad.numpy(), isplines.derivatives(x.detach().numpy(), 3), atol=1e-10)

    basis = ISplineBasis(order=3, num_basis=6, lower=0, upper=1, n_grid=11)
    weights = torch.ones(6, requires_grad=True)
    basis(weights).sum().backward()
    np.testing.assert_allclose(weights.grad.numpy(), basis.basis_vectors.sum(axis=0), rtol=1e-6)


def test_torch_keeps_dtype():
    torch = pytest.importorskip("torch")
    msplines = Msplines(3, mesh=MESH)
    assert msplines.evaluate_all(torch.linspace(0, 1, 5, dtype=torch.float32)).dtype == torch.float32
    assert msplines.evaluate_all(torch.tensor([0, 1])).dtype == torch.get_default_dtype()
    basis = ISplineBasis(order=3, num_basis=6, lower=0, upper=1, n_grid=11)
    assert basis(torch.ones(6, dtype=torch.float32)).dtype == torch.float32


def test_jax_grad_and_jit():
    jax = pytest.importorskip("jax")
    jax.config.update("jax_enable_x64", True)
    import jax.numpy as jnp

    isplines = Isplines(3, mesh=MESH)
    x = jnp.linspace(0.01, 0.99, 25)
    grad = jax.vmap(jax.grad(lambda xi: isplines.evaluate_all(xi[None])[0, 2]))(x)
    np.testing.assert_allclose(np.asarray(grad), isplines.derivatives(np.asarray(x), 3), atol=1e-10)

    jitted = jax.jit(isplines.evaluate_all)
    np.testing.assert_allclose(np.asarray(jitted(x)), isplines.evaluate_all(np.asarray(x)), atol=1e-12)

    basis = MSplineBasis(order=3, num_basis=6, lower=0, upper=1, n_grid=11)
    loss = jax.jit(lambda w: jnp.sum(basis(w) ** 2))
    assert np.isfinite(float(loss(jnp.ones(6))))


def test_mixing_backends_raises():
    torch = pytest.importorskip("torch")
    pytest.importorskip("jax")
    import jax.numpy as jnp

    from splinebasis._backend import get_backend

    with pytest.raises(TypeError):
        get_backend(torch.zeros(1), jnp.zeros(1))


def test_unknown_backend_raises():
    with pytest.raises(ValueError):
        ISplineBasis(order=3, num_basis=6, lower=0, upper=1, n_grid=11, backend="tensorflow")
