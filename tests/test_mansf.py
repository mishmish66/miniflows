"""Tests for the masked-autoregressive spline flow (ARSpline)"""

import jax
from jax import numpy as jnp, random as jr

from miniflows.causal_mlp import CausalMLP
from miniflows.mansf import ARSpline

N_BINS = 8


def make_arspline(key, dim: int = 2, width: int = 64, depth: int = 2) -> ARSpline:
    nparams = 3 * N_BINS - 1
    net = CausalMLP(
        num_ranks=dim,
        in_rank_dim="scalar",
        out_rank_dim=nparams,
        width=width,
        depth=depth,
        rng=key,
    )
    return ARSpline(net=net)


def test_arspline_roundtrip():
    m = make_arspline(jr.key(0))
    x = jr.normal(jr.key(1), (2,))
    z, _ = m.fwd_logdet(x)
    xr, _ = m.inv_logdet(z)
    assert jnp.allclose(xr, x, atol=1e-4)


def test_arspline_forward_inverse_logdets_cancel():
    m = make_arspline(jr.key(0))
    x = jr.normal(jr.key(1), (2,))
    z, ld = m.fwd_logdet(x)
    _, ldi = m.inv_logdet(z)
    assert jnp.allclose(ld + ldi, 0.0, atol=1e-4)


def test_arspline_logdet_matches_autodiff():
    m = make_arspline(jr.key(0))
    x = jr.normal(jr.key(2), (2,))
    _, ld = m.fwd_logdet(x)
    J = jax.jacobian(lambda x: m.fwd_logdet(x)[0])(x)
    assert jnp.allclose(ld, jnp.log(jnp.abs(jnp.linalg.det(J))), atol=1e-4)


def test_arspline_forward_is_autoregressive():
    dim = 4
    m = make_arspline(jr.key(0), dim=dim)
    x = jr.normal(jr.key(3), (dim,))
    # z_i depends on x_{<=i} only -> forward jacobian is lower triangular.
    J = jax.jacobian(lambda x: m.fwd_logdet(x)[0])(x)
    upper = jnp.triu(jnp.abs(J), k=1)
    assert jnp.all(upper < 1e-6)
    assert jnp.all(jnp.abs(jnp.diag(J)) > 0)
