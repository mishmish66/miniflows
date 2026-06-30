from jaxtyping import Float, Array

import jax
from jax import numpy as jnp
import equinox as eqx

from miniflows.spline import spline_fwd, spline_inv
from miniflows.causal_mlp import CausalMLP


class ARSpline(eqx.Module):
    """Masked-autoregressive RQ spline layer"""

    net: CausalMLP
    dim: int = eqx.field(static=True)
    n_bins: int = eqx.field(static=True)
    low: float = eqx.field(static=True, default=-5.0)
    high: float = eqx.field(static=True, default=5.0)

    def fwd_logdet(self, x: Float[Array, " d"], c: Float[Array, " c"] | None = None):
        params = self.net(x, c)  # (dim,) scalar-per-row -> (dim, n_params)
        z, ld = jax.vmap(spline_fwd, in_axes=(0, 0, None, None))(
            x, params, jnp.array(self.low), jnp.array(self.high)
        )
        return z, ld.sum()

    def inv_logdet(self, z: Float[Array, " d"], c: Float[Array, " c"] | None = None):
        x = jnp.zeros(self.dim)
        for i in range(self.dim):
            params = self.net(x, c)  # (dim, n_params)
            xi, _ = spline_inv(
                z[i], params[i], jnp.array(self.low), jnp.array(self.high)
            )
            x = x.at[i].set(xi)
        # log-det of the inverse is minus that of the forward at the solved x
        params = self.net(x, c)
        _, ld = jax.vmap(spline_fwd, in_axes=(0, 0, None, None))(
            x, params, jnp.array(self.low), jnp.array(self.high)
        )
        return x, -ld.sum()
