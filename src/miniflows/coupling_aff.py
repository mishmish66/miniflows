"""Affine coupling flow (RealNVP).

@inproceedings{dinh_density_2017,
    title = {Density estimation using {Real} {NVP}},
    url = {https://openreview.net/forum?id=HkpbnH9lx},
    booktitle = {5th {International} {Conference} on {Learning} {Representations} ({ICLR} 2017)},
    publisher = {OpenReview.net},
    author = {Dinh, Laurent and Sohl-Dickstein, Jascha and Bengio, Samy},
    year = {2017},
}
"""

from jax import numpy as jnp
from jax.lax import stop_gradient as sg

import equinox as eqx
from jaxtyping import Float, Array, Bool


class AffineCoupling(eqx.Module):
    """One RealNVP affine coupling layer."""

    mlp: eqx.nn.MLP
    id_mask: Bool[Array, " d"]
    min_scale: float = eqx.field(static=True, default=1e-3)

    @staticmethod
    def mlp_dims(flow_dim: int, cond_dim: int) -> tuple[int, int]:
        in_dim = flow_dim // 2 + cond_dim
        out_dim = (flow_dim - flow_dim // 2) * 2
        return in_dim, out_dim

    def fwd_logdet(
        self, x: Float[Array, " d"], c: Float[Array, " c"] | None = None
    ) -> tuple[Float[Array, " d"], Float[Array, ""]]:
        tform_dim = sg(~self.id_mask).sum()
        if c is None:
            h = x[self.id_mask]
        else:
            h = jnp.concat([x[self.id_mask], c])

        mlp_out = self.mlp(h)

        param_dim = tform_dim * 2
        if mlp_out.shape != (param_dim,):
            msg = f"cond_mlp output {mlp_out.shape} but needed {(param_dim,)}"
            raise ValueError(msg)

        s_raw, t = jnp.split(mlp_out, 2)
        s = s_raw / (1 + jnp.abs(s_raw / jnp.log(self.min_scale)))

        tformed = x[~self.id_mask] * jnp.exp(s) + t
        return x.at[~self.id_mask].set(tformed), s.sum()

    def inv_logdet(
        self, z: Float[Array, " d"], c: Float[Array, " c"] | None = None
    ) -> tuple[Float[Array, " d"], Float[Array, ""]]:
        tform_dim = sg(~self.id_mask).sum()
        if c is None:
            h = z[self.id_mask]
        else:
            h = jnp.concat([z[self.id_mask], c])

        mlp_out = self.mlp(h)

        param_dim = tform_dim * 2
        if mlp_out.shape != (param_dim,):
            msg = f"cond_mlp output {mlp_out.shape} but needed {(param_dim,)}"
            raise ValueError(msg)

        s_raw, t = jnp.split(mlp_out, 2)
        s = s_raw / (1 + jnp.abs(s_raw / jnp.log(self.min_scale)))

        tformed = (z[~self.id_mask] - t) * jnp.exp(-s)
        return z.at[~self.id_mask].set(tformed), -s.sum()
