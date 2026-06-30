"""Tests for the PLU-decomposed invertible linear bijector (PLUB)."""

import jax
from jax import numpy as jnp, random as jr
import pytest

from miniflows.plu import PLU


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_plub(dim: int, seed: int = 0) -> PLU:
    return PLU(dim, rng=jr.key(seed))


# ---------------------------------------------------------------------------
# Shape / finiteness
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("dim", [2, 4, 8])
def test_plub_forward_shape_and_finite(dim):
    m = _make_plub(dim)
    x = jr.normal(jr.key(1), (dim,))
    y, ld = m.fwd_logdet(x)
    assert y.shape == (dim,)
    assert ld.shape == ()
    assert jnp.all(jnp.isfinite(y))
    assert jnp.isfinite(ld)


@pytest.mark.parametrize("dim", [2, 4, 8])
def test_plub_inverse_shape_and_finite(dim):
    m = _make_plub(dim)
    y = jr.normal(jr.key(2), (dim,))
    x, ld = m.inv_logdet(y)
    assert x.shape == (dim,)
    assert ld.shape == ()
    assert jnp.all(jnp.isfinite(x))
    assert jnp.isfinite(ld)


# ---------------------------------------------------------------------------
# Roundtrip: forward then inverse recovers input
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("dim", [2, 4, 8])
def test_plub_roundtrip_fwd_inv(dim):
    m = _make_plub(dim, seed=10)
    x = jr.normal(jr.key(3), (dim,))
    y, _ = m.fwd_logdet(x)
    xr, _ = m.inv_logdet(y)
    assert jnp.allclose(xr, x, atol=1e-5), f"max err={jnp.abs(xr - x).max()}"


@pytest.mark.parametrize("dim", [2, 4, 8])
def test_plub_roundtrip_inv_fwd(dim):
    m = _make_plub(dim, seed=20)
    y = jr.normal(jr.key(4), (dim,))
    x, _ = m.inv_logdet(y)
    yr, _ = m.fwd_logdet(x)
    assert jnp.allclose(yr, y, atol=1e-5), f"max err={jnp.abs(yr - y).max()}"


# ---------------------------------------------------------------------------
# Forward/inverse log-dets cancel
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("dim", [2, 4, 8])
def test_plub_logdets_cancel(dim):
    m = _make_plub(dim, seed=30)
    x = jr.normal(jr.key(5), (dim,))
    _, ld_fwd = m.fwd_logdet(x)
    _, ld_inv = m.inv_logdet(m.fwd_logdet(x)[0])
    assert jnp.allclose(ld_fwd + ld_inv, 0.0, atol=1e-6)


# ---------------------------------------------------------------------------
# Log-det matches autodiff (via Jacobian determinant)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("dim", [2, 4])
def test_plub_logdet_matches_autodiff(dim):
    m = _make_plub(dim, seed=40)
    x = jr.normal(jr.key(6), (dim,))
    _, ld = m.fwd_logdet(x)
    J = jax.jacobian(lambda x: m.fwd_logdet(x)[0])(x)
    expected = jnp.log(jnp.abs(jnp.linalg.det(J)))
    assert jnp.allclose(ld, expected, atol=1e-5), f"ld={ld}, expected={expected}"


# ---------------------------------------------------------------------------
# Log-det formula check: should equal sum of logs diagonal of U
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("dim", [2, 4, 8])
def test_plub_logdet_equals_logs_sum(dim):
    m = _make_plub(dim, seed=50)
    x = jr.normal(jr.key(7), (dim,))
    _, ld = m.fwd_logdet(x)
    assert jnp.allclose(ld, m.logs.sum(), atol=1e-6)


# ---------------------------------------------------------------------------
# Inverse log-det equals negative forward log-det
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("dim", [2, 4, 8])
def test_plub_inverse_logdet_negation(dim):
    m = _make_plub(dim, seed=60)
    x = jr.normal(jr.key(8), (dim,))
    _, fwd_ld = m.fwd_logdet(x)
    _, inv_ld = m.inv_logdet(m.fwd_logdet(x)[0])
    assert jnp.allclose(inv_ld, -fwd_ld, atol=1e-6)


# ---------------------------------------------------------------------------
# Determinism: same key gives same output
# ---------------------------------------------------------------------------


def test_plub_determinism():
    m1 = _make_plub(4, seed=0)
    m2 = _make_plub(4, seed=0)
    x = jr.normal(jr.key(9), (4,))
    y1, ld1 = m1.fwd_logdet(x)
    y2, ld2 = m2.fwd_logdet(x)
    assert jnp.allclose(y1, y2)
    assert jnp.allclose(ld1, ld2)
