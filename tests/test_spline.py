"""Tests for the rational-quadratic spline primitives."""

import jax
from jax import numpy as jnp, random as jr
import pytest

from miniflows.spline import RationalQuadratic, RationalQuadraticSpline

N_BINS = 8


def _random_segment(key):
    e0, e1 = jnp.exp(0.5 * jr.normal(key, (2,)))
    return RationalQuadratic(e0=e0, e1=e1)


def _random_params(key, n_bins=N_BINS):
    return jr.normal(key, (3 * n_bins - 1,))


def test_segment_maps_unit_endpoints():
    seg = RationalQuadratic(e0=jnp.array(1.3), e1=jnp.array(0.7))
    assert jnp.allclose(seg.fwd(jnp.array(0.0)), 0.0, atol=1e-6)
    assert jnp.allclose(seg.fwd(jnp.array(1.0)), 1.0, atol=1e-6)


def test_segment_roundtrip():
    for s in range(5):
        seg = _random_segment(jr.key(s))
        z = jnp.linspace(0.01, 0.99, 25)
        w = jax.vmap(seg.fwd)(z)
        zr = jax.vmap(seg.inverse)(w)
        assert jnp.allclose(zr, z, atol=1e-4)


def test_segment_dydx_matches_autodiff():
    seg = _random_segment(jr.key(0))
    for z in jnp.linspace(0.05, 0.95, 10):
        ad = jax.grad(lambda z: seg.fwd(z))(z)
        assert jnp.allclose(ad, seg.dydx(z), atol=1e-4)


def test_segment_monotone():
    seg = _random_segment(jr.key(1))
    y = jax.vmap(seg.fwd)(jnp.linspace(0.0, 1.0, 50))
    assert jnp.all(jnp.diff(y) > 0)


@pytest.mark.parametrize("bad_len", [9, 10, 12])
def test_decode_rejects_bad_param_length(bad_len):
    with pytest.raises(ValueError):
        RationalQuadraticSpline.decode(jnp.zeros(bad_len))


def test_decode_accepts_valid_length():
    spline = RationalQuadraticSpline.decode(_random_params(jr.key(0)))
    # knots partition [0,1] in both axes
    assert jnp.allclose(spline.k_x0s[0], 0.0)
    assert jnp.allclose(spline.k_x1s[-1], 1.0, atol=1e-5)
    assert jnp.allclose(spline.k_y0s[0], 0.0)
    assert jnp.allclose(spline.k_y1s[-1], 1.0, atol=1e-5)


def test_spline_roundtrip_inside_unit():
    spline = RationalQuadraticSpline.decode(_random_params(jr.key(3)))
    x = jnp.linspace(0.01, 0.99, 50)
    y, ld = jax.vmap(spline.fwd_logdet)(x)
    xr, ldi = jax.vmap(spline.inv_logdet)(y)
    assert jnp.allclose(xr, x, atol=1e-4)
    assert jnp.allclose(ld + ldi, 0.0, atol=1e-4)


def test_spline_logdet_matches_autodiff():
    spline = RationalQuadraticSpline.decode(_random_params(jr.key(4)))
    for x in jnp.linspace(0.05, 0.95, 12):
        dy = jax.grad(lambda x: spline.fwd_logdet(x)[0])(x)
        _, ld = spline.fwd_logdet(x)
        assert jnp.allclose(jnp.log(dy), ld, atol=1e-4)


def test_spline_tails_are_identity():
    spline = RationalQuadraticSpline.decode(_random_params(jr.key(5)))
    for x in [jnp.array(-2.0), jnp.array(3.5)]:
        y, ld = spline.fwd_logdet(x)
        assert jnp.allclose(y, x, atol=1e-6)
        assert jnp.allclose(ld, 0.0, atol=1e-6)


def test_spline_is_monotone_on_unit():
    spline = RationalQuadraticSpline.decode(_random_params(jr.key(6)))
    y, _ = jax.vmap(spline.fwd_logdet)(jnp.linspace(0.0, 1.0, 100))
    assert jnp.all(jnp.diff(y) > 0)
