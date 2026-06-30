"""Tests for the masked autoregressive CausalLinear / CausalMLP."""

import jax
from jax import numpy as jnp, random as jr
import pytest

from miniflows.causal_mlp import CausalLinear, CausalMLP


def test_causal_linear_respects_ranks():
    # out 0 (rank 0) reads nothing; out 1 (rank 1) reads the two rank-0 inputs.
    lay = CausalLinear([0, 0, 1], [0, 1], rng=jr.key(0))
    x = jnp.array([1.0, 1.0, 1.0])
    J = jax.jacobian(lay)(x)
    assert jnp.allclose(J[0], 0.0)
    assert jnp.all(jnp.abs(J[1, :2]) > 0)
    assert jnp.allclose(J[1, 2], 0.0)


def test_causal_linear_shapes():
    lay = CausalLinear([0, 1, 2], [0, 1, 2, 3], rng=jr.key(0))
    out = lay(jnp.ones(3))
    assert out.shape == (4,)


def test_causal_mlp_output_shape_vector_out():
    m = CausalMLP(
        num_ranks=4,
        in_rank_dim="scalar",
        out_rank_dim=5,
        width=16,
        depth=2,
        rng=jr.key(0),
    )
    assert m(jnp.arange(4.0), None).shape == (4, 5)


def test_causal_mlp_output_shape_scalar_out():
    m = CausalMLP(
        num_ranks=4,
        in_rank_dim="scalar",
        out_rank_dim="scalar",
        width=16,
        depth=2,
        rng=jr.key(0),
    )
    assert m(jnp.arange(4.0), None).shape == (4,)


def test_causal_mlp_is_strictly_autoregressive():
    dim = 5
    m = CausalMLP(
        num_ranks=dim,
        in_rank_dim="scalar",
        out_rank_dim=3,
        width=24,
        depth=3,
        rng=jr.key(1),
    )
    x = jr.normal(jr.key(2), (dim,))
    # J[i, j, k] = d(param i,j) / d(x k); coordinate i must use only x_{<i}.
    J = jax.jacobian(lambda x: m(x, None))(x)
    dep = jnp.abs(J).sum(1)  # (dim, dim): coord i vs input k
    expected = jnp.tril(jnp.ones((dim, dim)), k=-1)  # strict lower-triangular
    assert jnp.allclose((dep > 1e-7).astype(float), expected)


def test_causal_mlp_conditioning_changes_output():
    dim = 3
    m = CausalMLP(
        num_ranks=dim,
        in_rank_dim="scalar",
        out_rank_dim=2,
        width=16,
        depth=2,
        cond_dim=4,
        rng=jr.key(3),
    )
    x = jr.normal(jr.key(4), (dim,))
    c1 = jr.normal(jr.key(5), (4,))
    c2 = jr.normal(jr.key(6), (4,))
    out1, out2 = m(x, c1), m(x, c2)
    assert out1.shape == (dim, 2)
    # coordinate 0 reads nothing, but later coordinates must respond to c
    assert not jnp.allclose(out1[1:], out2[1:])


def test_causal_mlp_requires_two_ranks():
    with pytest.raises(ValueError):
        CausalMLP(
            num_ranks=1,
            in_rank_dim="scalar",
            out_rank_dim=2,
            width=8,
            depth=1,
            rng=jr.key(0),
        )
