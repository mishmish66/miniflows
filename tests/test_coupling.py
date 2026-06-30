"""Tests for the affine-coupling layer (AffineCoupling).

The layer's primary interface is `fwd_logdet` / `inv_logdet`, mirroring the
RQ-spline `SplineCoupling`.  `s` is a log-scale, so the forward applies
`x -> x * exp(s) + t` on the transformed dims with log-det `s.sum()`.
"""

import jax
from jax import numpy as jnp, random as jr
import equinox as eqx
import pytest

from miniflows.coupling_aff import AffineCoupling


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make(flow_dim: int, cond_dim: int = 0, seed: int = 0) -> AffineCoupling:
    """Build an AffineCoupling using mlp_dims (handles cond_dim == 0 too)."""
    in_dim, out_dim = AffineCoupling.mlp_dims(flow_dim, cond_dim)
    mlp = eqx.nn.MLP(in_dim, out_dim, width_size=32, depth=2, key=jr.key(seed))
    # first flow_dim // 2 dims are identity (pass-through / MLP input)
    id_mask = jnp.arange(flow_dim) < (flow_dim // 2)
    return AffineCoupling(mlp=mlp, id_mask=id_mask)


# ---------------------------------------------------------------------------
# mlp_dims static helper
# ---------------------------------------------------------------------------


def test_mlp_dims_unconditional():
    flow_dim = 4
    in_dim, out_dim = AffineCoupling.mlp_dims(flow_dim, 0)
    assert in_dim == flow_dim // 2
    assert out_dim == (flow_dim - flow_dim // 2) * 2


def test_mlp_dims_conditional_includes_cond():
    in_dim, out_dim = AffineCoupling.mlp_dims(4, 3)
    assert in_dim == 4 // 2 + 3
    assert out_dim == (4 - 4 // 2) * 2


# ---------------------------------------------------------------------------
# Unconditional — shape / finiteness
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("flow_dim", [4, 6])
def test_uncond_fwd_shape_finite(flow_dim):
    m = _make(flow_dim)
    x = jr.normal(jr.key(1), (flow_dim,))
    y, ld = m.fwd_logdet(x, None)
    assert y.shape == (flow_dim,)
    assert jnp.all(jnp.isfinite(y))
    assert jnp.isfinite(ld)


@pytest.mark.parametrize("flow_dim", [4, 6])
def test_uncond_inv_shape_finite(flow_dim):
    m = _make(flow_dim)
    z = jr.normal(jr.key(2), (flow_dim,))
    x, ld = m.inv_logdet(z, None)
    assert x.shape == (flow_dim,)
    assert jnp.all(jnp.isfinite(x))
    assert jnp.isfinite(ld)


# ---------------------------------------------------------------------------
# Unconditional — roundtrip / log-dets cancel / autodiff / id-dims passthrough
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("flow_dim", [4, 6])
def test_uncond_roundtrip(flow_dim):
    m = _make(flow_dim, seed=10)
    x = jr.normal(jr.key(3), (flow_dim,))
    y, _ = m.fwd_logdet(x, None)
    xr, _ = m.inv_logdet(y, None)
    assert jnp.allclose(xr, x, atol=1e-5), f"max err={jnp.abs(xr - x).max()}"


@pytest.mark.parametrize("flow_dim", [4, 6])
def test_uncond_logdets_cancel(flow_dim):
    m = _make(flow_dim, seed=20)
    x = jr.normal(jr.key(4), (flow_dim,))
    y, ld_fwd = m.fwd_logdet(x, None)
    _, ld_inv = m.inv_logdet(y, None)
    assert jnp.allclose(ld_fwd + ld_inv, 0.0, atol=1e-5)


@pytest.mark.parametrize("flow_dim", [4, 6])
def test_uncond_logdet_matches_autodiff(flow_dim):
    m = _make(flow_dim, seed=30)
    x = 0.5 * jr.normal(jr.key(5), (flow_dim,))
    _, ld = m.fwd_logdet(x, None)
    J = jax.jacobian(lambda x: m.fwd_logdet(x, None)[0])(x)
    expected = jnp.log(jnp.abs(jnp.linalg.det(J)))
    assert jnp.allclose(ld, expected, atol=1e-5), f"ld={ld}, expected={expected}"


@pytest.mark.parametrize("flow_dim", [4, 6])
def test_uncond_id_dims_unchanged(flow_dim):
    m = _make(flow_dim, seed=40)
    x = jr.normal(jr.key(6), (flow_dim,))
    y, _ = m.fwd_logdet(x, None)
    assert jnp.allclose(y[m.id_mask], x[m.id_mask], atol=1e-6)


# ---------------------------------------------------------------------------
# Conditional — shape / roundtrip / log-dets / autodiff / varies with c
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("flow_dim,cond_dim", [(4, 2), (6, 3)])
def test_cond_fwd_inv_shape_finite(flow_dim, cond_dim):
    m = _make(flow_dim, cond_dim)
    x = jr.normal(jr.key(7), (flow_dim,))
    c = jr.normal(jr.key(8), (cond_dim,))
    y, ldf = m.fwd_logdet(x, c)
    xr, ldi = m.inv_logdet(y, c)
    assert y.shape == (flow_dim,) and xr.shape == (flow_dim,)
    assert jnp.all(jnp.isfinite(y)) and jnp.all(jnp.isfinite(xr))
    assert jnp.isfinite(ldf) and jnp.isfinite(ldi)


@pytest.mark.parametrize("flow_dim,cond_dim", [(4, 2), (6, 3)])
def test_cond_roundtrip(flow_dim, cond_dim):
    m = _make(flow_dim, cond_dim, seed=50)
    x = jr.normal(jr.key(11), (flow_dim,))
    c = jr.normal(jr.key(12), (cond_dim,))
    y, _ = m.fwd_logdet(x, c)
    xr, _ = m.inv_logdet(y, c)
    assert jnp.allclose(xr, x, atol=1e-5), f"max err={jnp.abs(xr - x).max()}"


@pytest.mark.parametrize("flow_dim,cond_dim", [(4, 2), (6, 3)])
def test_cond_logdets_cancel(flow_dim, cond_dim):
    m = _make(flow_dim, cond_dim, seed=60)
    x = jr.normal(jr.key(13), (flow_dim,))
    c = jr.normal(jr.key(14), (cond_dim,))
    y, ld_fwd = m.fwd_logdet(x, c)
    _, ld_inv = m.inv_logdet(y, c)
    assert jnp.allclose(ld_fwd + ld_inv, 0.0, atol=1e-5)


@pytest.mark.parametrize("flow_dim,cond_dim", [(4, 2), (6, 3)])
def test_cond_logdet_matches_autodiff(flow_dim, cond_dim):
    m = _make(flow_dim, cond_dim, seed=70)
    x = 0.5 * jr.normal(jr.key(15), (flow_dim,))
    c = jr.normal(jr.key(16), (cond_dim,))
    _, ld = m.fwd_logdet(x, c)
    J = jax.jacobian(lambda x: m.fwd_logdet(x, c)[0])(x)
    expected = jnp.log(jnp.abs(jnp.linalg.det(J)))
    assert jnp.allclose(ld, expected, atol=1e-5), f"ld={ld}, expected={expected}"


@pytest.mark.parametrize("flow_dim,cond_dim", [(4, 2), (6, 3)])
def test_cond_output_varies_with_c(flow_dim, cond_dim):
    m = _make(flow_dim, cond_dim, seed=80)
    x = jr.normal(jr.key(17), (flow_dim,))
    c1 = jr.normal(jr.key(18), (cond_dim,))
    c2 = jr.normal(jr.key(19), (cond_dim,))
    y1, _ = m.fwd_logdet(x, c1)
    y2, _ = m.fwd_logdet(x, c2)
    # transformed dims must differ; identity dims stay the same
    assert not jnp.allclose(y1[~m.id_mask], y2[~m.id_mask])
    assert jnp.allclose(y1[m.id_mask], y2[m.id_mask])


# ---------------------------------------------------------------------------
# MLP-output-shape validation
# ---------------------------------------------------------------------------


def _make_bad(flow_dim: int = 4, seed: int = 0) -> AffineCoupling:
    """Layer whose MLP emits the wrong number of params."""
    in_dim, out_dim = AffineCoupling.mlp_dims(flow_dim, 0)
    mlp = eqx.nn.MLP(in_dim, out_dim + 1, width_size=8, depth=1, key=jr.key(seed))
    id_mask = jnp.arange(flow_dim) < (flow_dim // 2)
    return AffineCoupling(mlp=mlp, id_mask=id_mask)


def test_fwd_rejects_bad_mlp_output():
    m = _make_bad()
    with pytest.raises(ValueError):
        m.fwd_logdet(jr.normal(jr.key(1), (4,)), None)


def test_inv_rejects_bad_mlp_output():
    m = _make_bad()
    with pytest.raises(ValueError):
        m.inv_logdet(jr.normal(jr.key(1), (4,)), None)
