# 4761 Project: Anatomy-Weighted HMRF

This repository contains a Python implementation of an anatomy-weighted Hidden Markov Random Field (AW-HMRF) workflow for spatial transcriptomics data stored as `AnnData` (`.h5ad`) files. The main packaged workflow is a single-run experiment script that:

- loads an input `.h5ad`
- builds an anatomy-weighted spatial kNN graph
- initializes clusters with `kmeans` or `leiden`
- treats the initialization as a baseline
- runs AW-HMRF refinement
- evaluates baseline and HMRF outputs
- saves tables, plots, state assignments, and optionally an output `.h5ad`

## What Each File Is

### Main packaged workflow

- `scripts/run_aw_hmrf_once.py`
  Main entry point. Runs one full experiment on one input `.h5ad` and writes outputs to a parameter-named folder under `outputs/`.

### Core model code

- `src/spatial_mrf/model_HMRF.py`
  AW-HMRF model implementation with Gaussian emissions, EM/ICM optimization, configurable initialization, and energy tracking.

- `src/spatial_mrf/utils.py`
  Utility functions for building the anatomy-weighted graph, remapping clusters to atlas labels, aligning colors, and saving spatial plots.

- `src/spatial_mrf/evaluation.py`
  Evaluation helpers and energy-history plotting.

- `src/spatial_mrf/metrics.py`
  Metric implementations: ARI, NMI, purity, homogeneity, completeness, silhouette, and spatial consistency.

### Other scripts

- `scripts/check_dependencies.py`
  Small dependency checker.

- `scripts/demo_spatial_mrf.py`
  Lightweight toy demo for the older binary spatial MRF code path.

### Data and notebooks

- `data/adata_sub.h5ad`
  Default sample input used by the packaged AW-HMRF runner.

- `data/adata_HPF.h5ad`, `data/Zhuang_1080_embedded.h5ad`
  Other local `.h5ad` inputs used in experiments.

- `notebooks/`
  Exploratory notebooks for preprocessing, analysis, and plotting. These are not required to run the packaged script.

  Also, notebooks hippocampus_eval.ipynb and Eval_whole_section.ipynb provide analysis and visualization of gene expression features of AW-HMRF clustering and parcellation, which is worth playing around with.

## System Requirements

### Python version

- Python `>= 3.10`

### Core runtime dependencies

From `pyproject.toml` and `requirements.txt`:

- `numpy>=1.26`
- `pandas>=2.2`
- `matplotlib>=3.8`
- `scipy>=1.12`
- `scikit-learn>=1.4`

### Extra dependencies needed for the packaged AW-HMRF runner

The main runner also uses:

- `anndata>=0.10`
- `scanpy>=1.10`
- `python-igraph>=0.11`
- `leidenalg>=0.10`

These are listed in `requirements-notebooks.txt` because the same environment is also used by the notebooks.

### Hardware

No GPU is required. Current experiments are CPU-based.

Recommended:

- for sample-size runs such as `data/adata_sub.h5ad`: a normal laptop or workstation is fine
- for larger `.h5ad` files: a machine with more RAM is recommended because the workflow loads the full `AnnData`, embedding matrix, and spatial graph in memory

Practical guidance:

- sample experiments: 8 GB to 16 GB RAM is usually enough
- larger experiments: 32 GB RAM or more is safer

## Installation

There is no compilation step. This is a Python project.

Create an environment and install dependencies:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -U pip
pip install -e .
pip install -r requirements-notebooks.txt
```

If you only want the base package code and not the full AW-HMRF runner, `pip install -e .` is enough. For the packaged runner, `requirements-notebooks.txt` is effectively required because it includes `scanpy` and `leidenalg`.

## Sample Test Run

This is the simplest end-to-end run on the included sample input:

```bash
python3 scripts/run_aw_hmrf_once.py \
  --beta 20 \
  --label-key parcellation_division \
  --n-regions 15 \
  --init-method kmeans \
  --k 8 \
  --alpha 0.2 \
  --max-em-iter 10 \
  --max-icm-iter 5 \
  --random-state 0 \
  --save-adata
```

This uses:

- input file: `data/adata_sub.h5ad`
- embedding key: `X_pca`
- coordinate key: `spatial`
- label key: `parcellation_division`

### Expected sample output

The command creates a folder under `outputs/` named from the run parameters, for example:

```text
outputs/hmrf_label-parcellation_division_beta-20_K-15_init-kmeans_k-8_alpha-0p2_em-10_icm-5_seed-0/
```

Inside it, the main saved files are:
- `config.json`
- `metrics_summary.csv`
- `metrics_summary.json`
- `baseline_kmeans/`
- `hmrf/`
- optional `adata_with_hmrf.h5ad`

Inside `baseline_kmeans/` or `hmrf/`, you will find:

- `metrics.csv`
- `metrics.json`
- `confusion_counts.csv`
- `confusion_normalized.csv`
- `confusion_heatmap.png`
- `per_region_accuracy.csv`
- `per_region_accuracy.png`
- `spatial.png` or HMRF spatial plot image
- `*_states.csv`
- `*_states.npy`
- `label_mapping.json`

The HMRF folder also includes:

- `hmrf_result.npz`
- `energy_history.png`

## How To Run on Another Input File

The packaged runner supports an explicit input path:

```bash
python3 scripts/run_aw_hmrf_once.py \
  --adata-path /path/to/input_data.h5ad \
  --beta 20 \
  --label-key parcellation_division \
  --n-regions 20 \
  --init-method leiden \
  --leiden-resolution 0.6 \
  --k 12 \
  --alpha 0.2 \
  --max-em-iter 10 \
  --max-icm-iter 5 \
  --random-state 0 \
  --save-adata
```

Important assumptions for the current packaged workflow:

- the input `.h5ad` must contain `adata.obsm["X_pca"]`
- the input `.h5ad` must contain `adata.obsm["spatial"]`
- the label column passed by `--label-key` must exist in `adata.obs`

## Parameters

### Required

- `--beta`
  Spatial coupling strength for the HMRF.

### Commonly changed

- `--adata-path`
  Input `.h5ad` path. Default is `data/adata_sub.h5ad`.

- `--label-key`
  Ground-truth atlas label column in `adata.obs`. Also used as the remap and color-alignment reference.

- `--n-regions`
  Number of HMRF regions/clusters.

- `--init-method {kmeans,leiden}`
  Initialization method for the baseline and for the initial HMRF state assignment.

- `--leiden-resolution`
  Leiden resolution. Only used when `--init-method leiden`.

- `--k`
  Number of neighbors used in the anatomy-weighted kNN graph.

- `--alpha`
  Penalty weight for edges crossing different atlas labels or involving missing labels.

- `--max-em-iter`
  Maximum number of EM iterations.

- `--max-icm-iter`
  Maximum number of ICM sweeps inside each EM iteration.

- `--random-state`
  Random seed for reproducibility.

- `--save-adata`
  If set, save an output `.h5ad` with predicted columns written into `obs` and visualization colors written into `uns`.

### Advanced

- `--tol`
  Convergence threshold based on fraction of changed states.

- `--energy-tol`
  Optional convergence threshold based on energy improvement.

- `--spot-size`
  Marker size for spatial plots.

- `--output-root`
  Root directory for all run outputs.

## Running Larger Experiments

For larger files, the same script can be used, but you should be more careful with parameter choices and memory:

```bash
python3 scripts/run_aw_hmrf_once.py \
  --adata-path /path/to/large_data.h5ad \
  --beta 20 \
  --label-key parcellation_division \
  --n-regions 20 \
  --init-method kmeans \
  --k 8 \
  --alpha 0.2 \
  --max-em-iter 10 \
  --max-icm-iter 5 \
  --random-state 0 \
  --output-root outputs/large_runs \
  --save-adata
```

Suggestions for larger runs:

- start with `kmeans` initialization before trying `leiden`
- keep `max-em-iter` and `max-icm-iter` modest at first
- test on a subset before scaling up
- confirm that the target `label_key`, `X_pca`, and `spatial` keys exist in the larger file
- avoid saving too many very large `.h5ad` outputs unless you need them

## Batch Experiments

The packaged runner is designed to be called from shell loops. Example:

```bash
for k in 5 8 12
do
  for alpha in 0.1 0.2 0.3
  do
    python3 scripts/run_aw_hmrf_once.py \
      --beta 20 \
      --label-key parcellation_division \
      --n-regions 20 \
      --init-method kmeans \
      --k $k \
      --alpha $alpha \
      --max-em-iter 10 \
      --max-icm-iter 5 \
      --random-state 0 \
      --save-adata
  done
done
```

## How To Verify a Run

After a run finishes, the easiest checks are:

1. Confirm that a new parameter-named folder appears under `outputs/`.
2. Open `metrics_summary.csv` to compare the initialization baseline against HMRF.
3. Inspect `hmrf/energy_history.png` for convergence behavior.
4. Inspect the spatial plot and confusion outputs in `baseline_*` and `hmrf/`.

## Notes

- This repository does not currently provide a separate unit-test suite for the AW-HMRF workflow.
- The closest thing to a smoke test is a successful run of `scripts/run_aw_hmrf_once.py` on `data/adata_sub.h5ad`.
- The notebooks are useful for analysis and figure generation, but the main packaged functionality is the single-run experiment script plus the model/evaluation utilities it calls.
