# Spatial MRF Quickstart

## What Is Implemented

The current prototype includes:

- a binary spatial MRF with latent states in `{-1, +1}`
- a weighted region-to-region graph
- ICM inference for deterministic state updates
- a demo that compares weighted versus uniform spatial smoothing

## Key Files

- `src/spatial_mrf/model.py`: main MRF implementation
- `src/spatial_mrf/utils.py`: weight-matrix validation helpers
- `scripts/demo_spatial_mrf.py`: synthetic example and visualization
- `docs/spatial-mrf-method.md`: mathematical description of the model

## Suggested Local Setup

```bash
python -m venv .venv
. .venv/bin/activate
pip install -r requirements.txt
pip install -e .
python scripts/demo_spatial_mrf.py
```

On Windows PowerShell, the activation command is usually:

```powershell
.\.venv\Scripts\Activate.ps1
```

## How To Adapt This To Real Data

1. Build or import a symmetric brain-region weight matrix `W`.
2. Compute one unary score per region for a chosen gene.
3. Create `BinarySpatialMRF(weight_matrix=W, beta=...)`.
4. Call `fit(theta)` to infer region states.
5. Compare results under weighted and uniform graphs.

## Interpreting Inputs

- `theta[b]` should be positive when the data favors the `+1` state at region `b`
- `theta[b]` should be negative when the data favors the `-1` state
- larger `beta` means stronger spatial smoothing
- larger `W[b, b']` means stronger coupling between regions `b` and `b'`

## Good Next Steps

- decide what the latent states mean biologically
- define how unary evidence is computed from expression data
- choose a source for the connectivity matrix
- add evaluation code for weighted versus uniform comparisons
- add parameter sweeps over `beta`
