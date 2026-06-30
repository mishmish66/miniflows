"""Tests for the RQ-spline coupling layer (SplineCoupling)."""

import jax
from jax import numpy as jnp, random as jr
import equinox as eqx
import pytest

from miniflows.coupling_nsf import SplineCoupling

N_BINS = 8


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_unconditional(
    flow_dim: int, n_bins: int = N_BINS, seed: int = 0
) -> SplineCoupling:
    """Build an unconditional SplineCoupling using mlp_dims."""
    in_dim, out_dim = SplineCoupling.mlp_dims(flow_dim, 0, n_bins)
    key = jr.key(seed)
    mlp = eqx.nn.MLP(in_dim, out_dim, width_size=32, depth=2, key=key)
    # id_mask: first flow_dim//2 dims are identity (pass-through / MLP input)
    id_mask = jnp.arange(flow_dim) < (flow_dim // 2)
    return SplineCoupling(mlp=mlp, id_mask=id_mask, n_bins=n_bins)


def _make_conditional(
    flow_dim: int, cond_dim: int, n_bins: int = N_BINS, seed: int = 0
) -> SplineCoupling:
    """Build a conditional SplineCoupling.

    Note: SplineCoupling.mlp_dims has a known discrepancy for cond_dim > 0
    (it subtracts cond_dim from the output dimension, which doesn't match
    what the forward/inverse pass expects).  We therefore compute the MLP
    dims manually here:
      - MLP input  = flow_dim // 2 + cond_dim  (id-dims concatenated with c)
      - MLP output = (flow_dim - flow_dim // 2) * (n_bins * 3 - 1)
    """
    id_half = flow_dim // 2
    tform_dim = flow_dim - id_half
    in_dim = id_half + cond_dim
    out_dim = tform_dim * (n_bins * 3 - 1)
    key = jr.key(seed)
    mlp = eqx.nn.MLP(in_dim, out_dim, width_size=32, depth=2, key=key)
    id_mask = jnp.arange(flow_dim) < id_half
    return SplineCoupling(mlp=mlp, id_mask=id_mask, n_bins=n_bins)


# ---------------------------------------------------------------------------
# mlp_dims static helper
# ---------------------------------------------------------------------------


def test_mlp_dims_unconditional():
    flow_dim, n_bins = 4, N_BINS
    in_dim, out_dim = SplineCoupling.mlp_dims(flow_dim, 0, n_bins)
    assert in_dim == flow_dim // 2
    assert out_dim == (flow_dim - flow_dim // 2) * (n_bins * 3 - 1)


def test_mlp_dims_returns_ints():
    in_dim, out_dim = SplineCoupling.mlp_dims(6, 0, N_BINS)
    assert isinstance(in_dim, int)
    assert isinstance(out_dim, int)


# ---------------------------------------------------------------------------
# Unconditional — shape / finiteness
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("flow_dim", [4, 6])
def test_spline_coupling_uncond_fwd_shape_finite(flow_dim):
    m = _make_unconditional(flow_dim)
    x = jr.normal(jr.key(1), (flow_dim,))
    y, ld = m.fwd_logdet(x, None)
    assert y.shape == (flow_dim,)
    assert jnp.all(jnp.isfinite(y))
    assert jnp.all(jnp.isfinite(ld))


@pytest.mark.parametrize("flow_dim", [4, 6])
def test_spline_coupling_uncond_inv_shape_finite(flow_dim):
    m = _make_unconditional(flow_dim)
    z = jr.normal(jr.key(2), (flow_dim,))
    x, ld = m.inv_logdet(z, None)
    assert x.shape == (flow_dim,)
    assert jnp.all(jnp.isfinite(x))
    assert jnp.all(jnp.isfinite(ld))


# ---------------------------------------------------------------------------
# Unconditional — roundtrip
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("flow_dim", [4, 6])
def test_spline_coupling_uncond_roundtrip(flow_dim):
    m = _make_unconditional(flow_dim, seed=10)
    x = jr.normal(jr.key(3), (flow_dim,))
    y, _ = m.fwd_logdet(x, None)
    xr, _ = m.inv_logdet(y, None)
    assert jnp.allclose(xr, x, atol=1e-4), f"max err={jnp.abs(xr - x).max()}"


# ---------------------------------------------------------------------------
# Unconditional — forward/inverse log-dets cancel
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("flow_dim", [4, 6])
def test_spline_coupling_uncond_logdets_cancel(flow_dim):
    m = _make_unconditional(flow_dim, seed=20)
    x = jr.normal(jr.key(4), (flow_dim,))
    y, ld_fwd = m.fwd_logdet(x, None)
    _, ld_inv = m.inv_logdet(y, None)
    # ld_fwd and ld_inv are per-transformed-dim arrays; sum over dims
    assert jnp.allclose(ld_fwd.sum() + ld_inv.sum(), 0.0, atol=1e-4)


# ---------------------------------------------------------------------------
# Unconditional — log-det matches autodiff Jacobian determinant
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("flow_dim", [4, 6])
def test_spline_coupling_uncond_logdet_matches_autodiff(flow_dim):
    m = _make_unconditional(flow_dim, seed=30)
    # Use a small input so spline is inside [low, high] = [-5, 5]
    x = 0.5 * jr.normal(jr.key(5), (flow_dim,))
    _, ld = m.fwd_logdet(x, None)
    J = jax.jacobian(lambda x: m.fwd_logdet(x, None)[0])(x)
    expected = jnp.log(jnp.abs(jnp.linalg.det(J)))
    assert jnp.allclose(ld.sum(), expected, atol=1e-4), (
        f"ld.sum()={ld.sum()}, expected={expected}"
    )


# ---------------------------------------------------------------------------
# Unconditional — identity mask dims are truly passed through
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("flow_dim", [4, 6])
def test_spline_coupling_uncond_id_dims_unchanged(flow_dim):
    m = _make_unconditional(flow_dim, seed=40)
    x = jr.normal(jr.key(6), (flow_dim,))
    y, _ = m.fwd_logdet(x, None)
    assert jnp.allclose(y[m.id_mask], x[m.id_mask], atol=1e-6)


# ---------------------------------------------------------------------------
# Conditional — shape / finiteness
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("flow_dim,cond_dim", [(4, 2), (6, 3)])
def test_spline_coupling_cond_fwd_shape_finite(flow_dim, cond_dim):
    m = _make_conditional(flow_dim, cond_dim)
    x = jr.normal(jr.key(7), (flow_dim,))
    c = jr.normal(jr.key(8), (cond_dim,))
    y, ld = m.fwd_logdet(x, c)
    assert y.shape == (flow_dim,)
    assert jnp.all(jnp.isfinite(y))
    assert jnp.all(jnp.isfinite(ld))


@pytest.mark.parametrize("flow_dim,cond_dim", [(4, 2), (6, 3)])
def test_spline_coupling_cond_inv_shape_finite(flow_dim, cond_dim):
    m = _make_conditional(flow_dim, cond_dim)
    z = jr.normal(jr.key(9), (flow_dim,))
    c = jr.normal(jr.key(10), (cond_dim,))
    x, ld = m.inv_logdet(z, c)
    assert x.shape == (flow_dim,)
    assert jnp.all(jnp.isfinite(x))
    assert jnp.all(jnp.isfinite(ld))


# ---------------------------------------------------------------------------
# Conditional — roundtrip
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("flow_dim,cond_dim", [(4, 2), (6, 3)])
def test_spline_coupling_cond_roundtrip(flow_dim, cond_dim):
    m = _make_conditional(flow_dim, cond_dim, seed=50)
    x = jr.normal(jr.key(11), (flow_dim,))
    c = jr.normal(jr.key(12), (cond_dim,))
    y, _ = m.fwd_logdet(x, c)
    xr, _ = m.inv_logdet(y, c)
    assert jnp.allclose(xr, x, atol=1e-4), f"max err={jnp.abs(xr - x).max()}"


# ---------------------------------------------------------------------------
# Conditional — forward/inverse log-dets cancel
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("flow_dim,cond_dim", [(4, 2), (6, 3)])
def test_spline_coupling_cond_logdets_cancel(flow_dim, cond_dim):
    m = _make_conditional(flow_dim, cond_dim, seed=60)
    x = jr.normal(jr.key(13), (flow_dim,))
    c = jr.normal(jr.key(14), (cond_dim,))
    y, ld_fwd = m.fwd_logdet(x, c)
    _, ld_inv = m.inv_logdet(y, c)
    assert jnp.allclose(ld_fwd.sum() + ld_inv.sum(), 0.0, atol=1e-4)


# ---------------------------------------------------------------------------
# Conditional — log-det matches autodiff Jacobian determinant
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("flow_dim,cond_dim", [(4, 2), (6, 3)])
def test_spline_coupling_cond_logdet_matches_autodiff(flow_dim, cond_dim):
    m = _make_conditional(flow_dim, cond_dim, seed=70)
    x = 0.5 * jr.normal(jr.key(15), (flow_dim,))
    c = jr.normal(jr.key(16), (cond_dim,))
    _, ld = m.fwd_logdet(x, c)
    J = jax.jacobian(lambda x: m.fwd_logdet(x, c)[0])(x)
    expected = jnp.log(jnp.abs(jnp.linalg.det(J)))
    assert jnp.allclose(ld.sum(), expected, atol=1e-4), (
        f"ld.sum()={ld.sum()}, expected={expected}"
    )


# ---------------------------------------------------------------------------
# Conditional — output changes with conditioning
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("flow_dim,cond_dim", [(4, 2), (6, 3)])
def test_spline_coupling_cond_output_varies_with_c(flow_dim, cond_dim):
    m = _make_conditional(flow_dim, cond_dim, seed=80)
    x = jr.normal(jr.key(17), (flow_dim,))
    c1 = jr.normal(jr.key(18), (cond_dim,))
    c2 = jr.normal(jr.key(19), (cond_dim,))
    y1, _ = m.fwd_logdet(x, c1)
    y2, _ = m.fwd_logdet(x, c2)
    # The transformed dims must differ; identity dims stay the same
    assert not jnp.allclose(y1[~m.id_mask], y2[~m.id_mask])
    assert jnp.allclose(y1[m.id_mask], y2[m.id_mask])


# ---------------------------------------------------------------------------
# MLP-output-shape validation
# ---------------------------------------------------------------------------

def _make_bad(flow_dim: int = 4, n_bins: int = N_BINS, seed: int = 0) -> SplineCoupling:
    """Layer whose MLP emits the wrong number of params."""
    in_dim, out_dim = SplineCoupling.mlp_dims(flow_dim, 0, n_bins)
    mlp = eqx.nn.MLP(in_dim, out_dim + 1, width_size=8, depth=1, key=jr.key(seed))
    id_mask = jnp.arange(flow_dim) < (flow_dim // 2)
    return SplineCoupling(mlp=mlp, id_mask=id_mask, n_bins=n_bins)


def test_fwd_rejects_bad_mlp_output():
    m = _make_bad()
    with pytest.raises(ValueError):
        m.fwd_logdet(jr.normal(jr.key(1), (4,)), None)


def test_inv_rejects_bad_mlp_output():
    m = _make_bad()
    with pytest.raises(ValueError):
        m.inv_logdet(jr.normal(jr.key(1), (4,)), None)
