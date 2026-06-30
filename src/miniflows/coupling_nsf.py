"""GLOW-style neural spline coupling flow"""

import jax
from jax import numpy as jnp
from jax.lax import stop_gradient as sg

import equinox as eqx
from jaxtyping import Float, Array, Bool

from miniflows.spline import spline_fwd, spline_inv


class SplineCoupling(eqx.Module):
    """One RQ-spline coupling layer"""

    mlp: eqx.nn.MLP
    id_mask: Bool[Array, " d"]

    n_bins: int = eqx.field(static=True)

    low: float = eqx.field(static=True, default=-5.0)
    high: float = eqx.field(static=True, default=5.0)

    @staticmethod
    def mlp_dims(flow_dim: int, cond_dim: int, n_bins: int) -> tuple[int, int]:
        in_dim = flow_dim // 2 + cond_dim
        out_dim = (flow_dim - flow_dim // 2) * (n_bins * 3 - 1)
        return in_dim, out_dim

    def fwd_logdet(self, x: Float[Array, " d"], c: Float[Array, " c"] | None):
        tform_dim = sg(~self.id_mask).sum()
        if c is None:
            h = x[self.id_mask]
        else:
            h = jnp.concat([x[self.id_mask], c])

        mlp_out = self.mlp(h)

        param_dim = tform_dim * (self.n_bins * 3 - 1)
        if mlp_out.shape != (param_dim,):
            msg = f"cond_mlp output {mlp_out.shape} but needed {(param_dim,)}"
            raise ValueError(msg)

        params = mlp_out.reshape(tform_dim, -1)
        for_tform = x[~self.id_mask]
        tformed, ld = jax.vmap(spline_fwd, in_axes=(0, 0, None, None))(
            for_tform, params, jnp.array(self.low), jnp.array(self.high)
        )
        return x.at[~self.id_mask].set(tformed), ld

    def inv_logdet(self, z: Float[Array, " d"], c: Float[Array, " c"] | None):
        tform_dim = sg(~self.id_mask).sum()
        if c is None:
            h = z[self.id_mask]
        else:
            h = jnp.concat([z[self.id_mask], c])
        mlp_out = self.mlp(h)

        param_dim = tform_dim * (self.n_bins * 3 - 1)
        if mlp_out.shape != (param_dim,):
            msg = f"cond_mlp output {mlp_out.shape} but needed {(param_dim,)}"
            raise ValueError(msg)

        params = mlp_out.reshape(tform_dim, -1)
        for_tform = z[~self.id_mask]
        tformed, ld = jax.vmap(spline_inv, in_axes=(0, 0, None, None))(
            for_tform, params, jnp.array(self.low), jnp.array(self.high)
        )

        return z.at[~self.id_mask].set(tformed), ld
