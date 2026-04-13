# Spatial MRF Method

## Model Setup

We model one time slice at a time and focus only on spatial dependence across brain regions.

For a fixed gene, let:

- `B` be the number of brain regions
- `x_b` be the observed summary statistic for region `b`
- `z_b in {-1, +1}` be the latent expression state for region `b`
- `W` be a symmetric nonnegative region-to-region weight matrix

The binary spatial MRF uses an energy of the form

`E(z) = -sum_b theta_b z_b - beta sum_{b < b'} W_{bb'} z_b z_b'`

where:

- `theta_b` is the unary evidence from the data at region `b`
- `beta` controls the overall strength of spatial smoothing
- `W_{bb'}` determines how strongly regions `b` and `b'` are coupled

Larger positive weights encourage connected regions to share the same latent state.

## Unary Term

In the prototype implementation, the unary evidence is represented directly by a per-region score vector `theta`. This keeps the model flexible: `theta` can come from z-scores, log fold changes, posterior log-odds from a separate model, or another summary statistic.

## Inference

The current code uses Iterated Conditional Modes (ICM), a simple deterministic coordinate-ascent method.

For each region `b`, we compute the local field

`h_b = theta_b + beta sum_{b' != b} W_{bb'} z_b'`

and update

`z_b = +1 if h_b >= 0, else -1`

The algorithm cycles through all regions until no states change or the maximum number of iterations is reached.

## Why This Version

This is a good first project implementation because it:

- directly encodes weighted spatial smoothing
- is easy to inspect and debug
- lets us compare uniform versus biologically weighted graphs immediately
- avoids committing too early to a more complicated emission model

## Natural Extensions

Possible next steps include:

- replacing the simple unary score with a fitted likelihood model
- allowing more than two latent states
- estimating `beta` from data rather than fixing it manually
- comparing ICM to Gibbs sampling or mean-field approximations
