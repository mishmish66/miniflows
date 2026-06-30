# Mini Flows

This is a really small equinox flows library that only depends on equinox.
The scope of this is very small, it's really just meant to give some basic neural bijections.

The interface for these classes is through the fwd_logdet/inv_logdet functions.
To actually define a flow model usually our bijections would have to be combined with a base density.

## Modules

### Supporting characters
- CausalLinear autoregressive Linear which maintains causal dependency between inputs and outputs
- CausalMLP: autoregressive MLP as per MADE

### Bijectors
- PLU: PLU decomposed linear bijector
- CouplingAff: Invertible affine coupling transform with an MLP
- CouplingSpline: Invertible spline coupling transform with an MLP
- ARSpline: Invertible autoregressive spline with a causal MLP
- ARAffine: Invertible autoregressive affine with a causal MLP
