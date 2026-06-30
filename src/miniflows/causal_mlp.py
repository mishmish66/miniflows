from collections.abc import Callable
from typing import Literal
from itertools import pairwise

from jax import numpy as jnp
import jax
from jax import random as jr

import equinox as eqx
from jaxtyping import Float, Array, Key, Int


class CausalLinear(eqx.Module):
    """Masked linear with per-unit autoregressive ranks."""

    w_flat: Float[Array, " n"]
    bias: Float[Array, " out"]
    unmasked_idxs: tuple[Int[Array, " n"], ...]

    in_dim: int = eqx.field(static=True)
    out_dim: int = eqx.field(static=True)

    def __init__(
        self,
        in_ranks: list[int],
        out_ranks: list[int],
        *,
        rng: Key[Array, ""],
    ):
        self.in_dim = len(in_ranks)
        self.out_dim = len(out_ranks)

        # mask[i, j] is True when output j may read input i, laid out as
        # (in_dim, out_dim) so it indexes the weight matrix directly.
        mask = jnp.array(in_ranks)[:, None] < jnp.array(out_ranks)[None, :]
        fan_in = mask.sum(axis=0)
        lim = 1.0 / jnp.sqrt(fan_in.clip(min=1))

        rng, wkey, bkey = jr.split(rng, 3)

        w_unif = jr.uniform(
            wkey, (len(in_ranks), len(out_ranks)), minval=-1.0, maxval=1.0
        )
        w_full = w_unif * lim[None, :]

        self.unmasked_idxs = jnp.nonzero(mask)
        self.w_flat = w_full[self.unmasked_idxs]
        self.bias = jr.uniform(bkey, (len(out_ranks),))

    def __call__(self, x: Float[Array, " in"]) -> Float[Array, " out"]:
        w_full = (
            jnp.zeros((self.in_dim, self.out_dim))
            .at[self.unmasked_idxs]
            .set(self.w_flat)
        )
        return jnp.einsum("i,io->o", x, w_full) + self.bias


class CausalMLP(eqx.Module):
    """Autoregressive MLP over a rank axis: x of shape (l, di) -> (l, do)."""

    layers: list[CausalLinear]
    activation: Callable[[Array], Array] = eqx.field(static=True)
    num_ranks: int = eqx.field(static=True)
    cond_dim: int | None = eqx.field(static=True)
    in_rank_dim: int | Literal["scalar"] = eqx.field(static=True)
    out_rank_dim: int | Literal["scalar"] = eqx.field(static=True)

    def __init__(
        self,
        num_ranks: int,
        in_rank_dim: int | Literal["scalar"],
        out_rank_dim: int | Literal["scalar"],
        width: int,
        depth: int,
        *,
        cond_dim: int | None = None,
        activation: Callable[[Array], Array] = jax.nn.gelu,
        rng: Key[Array, ""],
    ):
        if num_ranks < 2:
            msg = f"need num_ranks >= 2 for autoregressive structure, got {num_ranks}"
            raise ValueError(msg)

        self.in_rank_dim = in_rank_dim
        self.out_rank_dim = out_rank_dim
        self.num_ranks = num_ranks
        self.cond_dim = cond_dim

        if in_rank_dim == "scalar":
            in_rank_dim = 1
        if out_rank_dim == "scalar":
            out_rank_dim = 1

        # each layer is a (dim, per_row) grid flattened row-major: row r is a
        # contiguous block of `per_row` units all carrying rank r.
        per_row = [in_rank_dim] + [width] * depth
        ranks = [[r for r in range(num_ranks) for _ in range(p)] for p in per_row]

        if cond_dim is not None:
            ranks[0].extend([0] * cond_dim)

        # Hidden layers read inclusively (a rank-r unit may use rank-r inputs).
        # CausalLinear compares in_rank < out_rank, so bumping the out-ranks by
        # one turns that strict `<` into an inclusive `<=`.
        layers = []
        for ranks_in, ranks_out in pairwise(ranks):
            rng, key = jr.split(rng)
            incl = [r + 1 for r in ranks_out]
            layers.append(CausalLinear(ranks_in, incl, rng=key))

        # Final read-out stays strict so coordinate r depends only on x_{<r}, and
        # emits `out_rank_dim` units per coordinate (row-major: row r is its block).
        rng, key = jr.split(rng)
        ranks_out = [r for r in range(num_ranks) for _ in range(out_rank_dim)]
        layers.append(CausalLinear(ranks[-1], ranks_out, rng=key))

        self.layers = layers
        self.activation = activation

    def __call__(
        self, x: Float[Array, "l di"] | Float[Array, " l"], c: Float[Array, " c"] | None
    ) -> Float[Array, "l do"] | Float[Array, " l"]:
        # scalar input is already the flat (l,) vector CausalLinear wants;
        # otherwise flatten the (l, di) grid row-major.
        x = x if self.in_rank_dim == "scalar" else x.reshape(-1)
        if c is None:
            h = x
        else:
            h = jnp.concat([x, c])

        for i, lay in enumerate(self.layers):
            h = lay(h)
            if i < len(self.layers) - 1:
                h = self.activation(h)
        # scalar output stays (l,); otherwise unflatten to (l, do).
        if self.out_rank_dim == "scalar":
            return h
        return h.reshape(self.num_ranks, self.out_rank_dim)  # (l*do,) -> (l, do)
