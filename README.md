# 4761 Project: Spatial MRF / Anatomy-Weighted HMRF (AW-HMRF)

This repository contains a Python implementation of spatial Markov Random Field models for spatial transcriptomics, with a focus on an **Anatomy-Weighted Hidden Markov Random Field (AW-HMRF)** for MERFISH/Allen ABC Atlas-style data stored as **`AnnData` (`.h5ad`)**.

There are two “entry points” you can run:

- **Notebook workflows** under `notebooks/` (recommended for exploration + figures)
- A **single-run experiment script** `scripts/run_aw_hmrf_once.py` (recommended for reproducible runs + saving outputs)

There is **no compilation step**; this is a pure Python project.

## Repository Structure (what every file/folder is)

### Core package (`src/spatial_mrf/`)

- `src/spatial_mrf/model.py`: **Binary** spatial MRF baseline (`BinarySpatialMRF`) with ICM inference.
- `src/spatial_mrf/model_HMRF.py`: **Multi-class AW-HMRF** (`AW_HMRF`) using Gaussian emissions and an EM-style loop with an ICM E-step.
- `src/spatial_mrf/utils.py`: Helper utilities used by the runner (graph building, Hungarian remapping, plotting helpers). (This file currently mixes several utilities in one place.)
- `src/spatial_mrf/metrics.py`: Metric helpers (ARI, NMI, purity, silhouette, spatial consistency).
- `src/spatial_mrf/evaluation.py`: Evaluation wrapper + energy-history plotting.
- `src/spatial_mrf/__init__.py`: Package exports.

**Generated / local-only (not meant to be committed):**

- `src/spatial_mrf/__pycache__/`: Python bytecode cache.
- `src/spatial_mrf.egg-info/`: editable install metadata (created by `pip install -e .`).

### Scripts (`scripts/`)

- `scripts/run_aw_hmrf_once.py`: Main reproducible runner. Loads an input `.h5ad`, builds an anatomy-weighted spatial kNN graph, runs a baseline clustering, runs AW-HMRF refinement, evaluates, and saves outputs.
- `scripts/demo_spatial_mrf.py`: Small toy demo for the **binary** MRF (`BinarySpatialMRF`).
- `scripts/check_dependencies.py`: Quick import/version check for common dependencies.

### Notebooks (`notebooks/`)

Notebooks are used for exploration, preprocessing, and plots. Common ones:

- `notebooks/MERFISH.ipynb`: Load MERFISH / Allen ABC Atlas cached files, build an `AnnData` for a slice/section, and explore.
- `notebooks/exploreAllen.ipynb`: Explore the Allen `abc_atlas_access` cache/manifests and inspect metadata.
- `notebooks/aw_hmrf_toy.ipynb` (or `aw_hmrf.ipynb`): Toy end-to-end sanity check for AW-HMRF.
- `notebooks/aw_hmrf_data.ipynb`, `notebooks/Eval_whole_section.ipynb`, `notebooks/hippocampus_eval.ipynb`: Analysis/visualization notebooks (project-specific).

Jupyter also creates:

- `notebooks/.ipynb_checkpoints/`: auto-saved checkpoints (ignored by `.gitignore`).

### Data (`data/`)

- `data/merfish_download.py`: Minimal example of using `abc_atlas_access` to fetch cached files.
- `data/abc_atlas/`: Local Allen ABC Atlas cache directory (large; ignored by `.gitignore`).
- `data/*.h5ad`: Local experiment artifacts / intermediate `AnnData` files (ignored by `.gitignore`).

### Docs (`docs/`)

- `docs/project-proposal.md`: project proposal/notes.
- `docs/spatial-mrf-method.md`: math writeup for the spatial MRF idea.
- `docs/mrf-quickstart.md`: quickstart notes.

## System Requirements

### Python

- **Python >= 3.10** (tested with newer Python as well)

### Hardware

- CPU-only (no GPU required).
- RAM guidance (very approximate):
  - **Toy / small slice**: 8–16 GB
  - **Large sections / many cells**: 32 GB+ recommended (graph + embeddings + AnnData can be big)

## Dependencies and Installation

### Step-by-step installation (venv + pip)

From the repo root, run these commands in order (each on its own line):

1. `python3 -m venv .venv`

2. `source .venv/bin/activate`

3. `pip install -U pip`

4. `pip install -r requirements.txt`

5. `pip install -r requirements-notebooks.txt`

6. `pip install -e .`

### Optional: register your venv as a Jupyter kernel

Steps:

1. `source .venv/bin/activate`

2. `pip install -U ipykernel`

3. `python -m ipykernel install --user --name 4761-project --display-name "Python (4761-project)"`

**Why this helps:** it makes the “correct Python” show up as a selectable kernel in Jupyter, so notebooks reliably use your project’s `.venv` (instead of a different system Python where packages may be missing).

Then in JupyterLab: Kernel → Change Kernel → **Python (4761-project)**.

## Step-by-step: run everything (in order)

### 1) Install dependencies

Complete the install steps in **Dependencies and Installation** above.

### 2) Download the MERFISH/Allen data (cache it locally)

Launch Jupyter and run the download/cache cells in:

- `notebooks/exploreAllen.ipynb` (recommended starting point), or `notebooks/MERFISH.ipynb`.

Launch JupyterLab from your activated virtual environment: `jupyter lab`

Downloads will be cached under `data/abc_atlas/`.

### 3) Preprocess into an `.h5ad` that AW-HMRF can use

In `notebooks/MERFISH.ipynb`, create/save an `.h5ad` that contains:

- `adata.obsm["spatial"]` (coordinates)
- `adata.obsm["X_pca"]` (embedding)
- `adata.obs[<label-key>]` (optional but recommended for evaluation)

See the **“Download + preprocessing”** section below for a concrete recipe.

### 4) Run AW-HMRF

You can run AW-HMRF either:

- **In a notebook** (good for plots): use `notebooks/aw_hmrf_data.ipynb` / `notebooks/Eval_whole_section.ipynb` (project-specific), or
- **Via the reproducible script** (good for saving outputs):
  - Run `python scripts/run_aw_hmrf_once.py` and pass `--adata-path` pointing to your processed `.h5ad`, plus HMRF parameters like `--beta` and `--n-regions`.

## Quick “does it run?” test (sample inputs)

### 1) Dependency smoke check

Run `python scripts/check_dependencies.py` from your activated environment to confirm imports are available.

### 2) Toy binary MRF demo (fast)

Run `python scripts/demo_spatial_mrf.py` from your activated environment.

Expected behavior: prints inferred states for a tiny toy graph and opens/saves a matplotlib visualization window.

### 3) Toy AW-HMRF demo (notebook)

Run `notebooks/aw_hmrf_toy.ipynb` (or `notebooks/aw_hmrf.ipynb`) top-to-bottom.
Expected behavior: KMeans baseline vs HMRF results improve as \(\\beta\) increases on the toy grid.

## Running the AW-HMRF runner script (reproducible experiment)

### Minimal run on the default sample `.h5ad`

Run `scripts/run_aw_hmrf_once.py` on the default sample input (`data/adata_sub.h5ad`) and set parameters such as:

- `--beta`
- `--label-key`
- `--n-regions`
- `--init-method`
- `--k`, `--alpha`
- `--max-em-iter`, `--max-icm-iter`
- `--random-state`
- `--save-adata` (optional)

This assumes your input `.h5ad` contains:

- `adata.obsm["X_pca"]` (embedding; default key is `X_pca`)
- `adata.obsm["spatial"]` (coordinates; default key is `spatial`)
- `adata.obs[<label-key>]` (ground truth / atlas label column)

### Outputs

The script writes a parameter-named directory under `outputs/` containing:

- run config + summary metrics (`metrics_summary.csv/json`)
- baseline folder (`baseline_kmeans/` or baseline Leiden if chosen)
- `hmrf/` folder including `hmrf_result.npz` and `energy_history.png`
- optional output `.h5ad` if `--save-adata` is set

## Running on real MERFISH / large `.h5ad` files

### Recommended workflow

1. Use `notebooks/MERFISH.ipynb` to select a **single brain section** (or a subset).
2. Ensure you have:
   - embeddings in `adata.obsm["X_pca"]` (or update the runner to a different key)
   - coordinates in `adata.obsm["spatial"]`
3. Run `scripts/run_aw_hmrf_once.py` on that `.h5ad` first.
4. Scale up gradually:
   - increase cell count
   - tune graph \(k\), \\(\\alpha\\), and \\(\\beta\\)
   - keep `max-em-iter` / `max-icm-iter` modest until you trust runtime

### Public “large data” files and where to get them (Allen MERFISH / ABC Atlas)

This project uses the Allen Institute **ABC Atlas MERFISH (Zhuang ABCA)** releases. These are public and are accessed via the `abc_atlas_access` Python package (see `requirements-notebooks.txt`).

- **Dataset directory**: `Zhuang-ABCA-1`
  - **Expression matrices** (large `.h5ad`; exact names can vary by manifest):
    - `Zhuang-ABCA-1-log2.h5ad`
    - `Zhuang-ABCA-1-raw.h5ad`
  - **Metadata**:
    - `cell_metadata.csv`

- **CCF-aligned directory**: `Zhuang-ABCA-1-CCF`
  - **Metadata** (includes CCF/parcellation label columns used as ground truth):
    - `cell_metadata.csv`

**How to fetch/download**:

- Use `notebooks/exploreAllen.ipynb`, `notebooks/MERFISH.ipynb`, or `data/merfish_download.py`.
- Allen tutorial/reference: `https://alleninstitute.github.io/abc_atlas_access/notebooks/zhuang_merfish_tutorial.html`

Downloaded files are cached under `data/abc_atlas/` (ignored by git).

### Download + preprocessing (make an `.h5ad` you can run HMRF on)

You ultimately need a single `.h5ad` that contains:

- `adata.obsm["spatial"]`: spatial coordinates per cell (x,y or x,y,z)
- `adata.obsm["X_pca"]`: an embedding matrix (e.g. PCA)
- `adata.obs[<label-key>]`: atlas labels to evaluate against (optional but recommended)

#### Step 1: download/cache the Allen files

If you already have the Allen cache files in this repo under `data/abc_atlas/`, you do **not** need to download anything.

To “get the data” into your notebook session, do the following:

1. Activate your environment, then start JupyterLab: `source .venv/bin/activate` then `jupyter lab`
2. Open `notebooks/exploreAllen.ipynb` (or `notebooks/MERFISH.ipynb`)
3. Run the first cells that create an `AbcProjectCache` pointing at `../data/abc_atlas`
4. Run the cells that load the latest manifest and list available directories/matrices (this verifies the cache is being found)

If the cache is missing, those same notebooks will download it automatically via `abc_atlas_access` into `data/abc_atlas/`.

#### Step 2: build an `AnnData` for a single section (recommended)

Working section-by-section keeps memory manageable.

In `notebooks/MERFISH.ipynb`, select a target section (e.g. one `brain_section_label`), load only those cells from the expression matrix into memory, and write spatial coordinates into `adata.obsm["spatial"]`.

#### Step 3: preprocessing + PCA embedding (`X_pca`)

In `notebooks/MERFISH.ipynb`, run standard Scanpy preprocessing (normalize + log1p + HVGs + PCA) to create `adata.obsm["X_pca"]`.

#### Step 4: save your processed `.h5ad`

Save the processed AnnData to a local file under `data/` (for example `data/Zhuang_1080_embedded.h5ad`).

You can now run `scripts/run_aw_hmrf_once.py` and point `--adata-path` at that file.

### Key parameters (runner)

- `--beta`: spatial coupling strength (higher → smoother labels).
- `--n-regions`: number of clusters/states \(K\).
- `--k`: neighbors in kNN graph.
- `--alpha`: downweight/penalize edges crossing atlas labels or missing labels (see `build_anatomy_weighted_knn_graph`).
- `--init-method`: `{kmeans, leiden}` baseline/initialization.
- `--max-em-iter`, `--max-icm-iter`: runtime vs quality knobs.
- `--tol`, `--energy-tol`: convergence thresholds.

## Notes / limitations

- This repo currently has **no formal unit test suite**. The recommended smoke tests are the toy demos/notebooks and a successful `run_aw_hmrf_once.py` run.
- Some local data under `data/` (Allen cache + `.h5ad` artifacts) can be very large and is ignored by `.gitignore`.

