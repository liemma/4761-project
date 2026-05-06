from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scanpy as sc
from sklearn.cluster import KMeans


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from spatial_mrf.evaluation import evaluate_all, plot_energy_history
from spatial_mrf.model_HMRF import AW_HMRF
from spatial_mrf.utils import (
    align_visualization_colors,
    build_anatomy_weighted_knn_graph,
    plot_hmrf_spatial_results,
    remap_clusters_to_atlas,
)


DEFAULT_ADATA_PATH = PROJECT_ROOT / "data" / "adata_sub.h5ad"
DEFAULT_OUTPUT_ROOT = PROJECT_ROOT / "outputs"
DEFAULT_EMBEDDING_KEY = "X_pca"
DEFAULT_COORD_KEY = "spatial"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run one AW-HMRF configuration on adata_sub.h5ad and save outputs.",
    )
    parser.add_argument(
        "--adata-path",
        type=Path,
        default=DEFAULT_ADATA_PATH,
        help="Path to the input AnnData .h5ad file. Defaults to data/adata_sub.h5ad.",
    )
    parser.add_argument("--label-key", default="parcellation_division")
    parser.add_argument("--beta", type=float, required=True)
    parser.add_argument("--n-regions", type=int, default=15)
    parser.add_argument("--init-method", choices=["kmeans", "leiden"], default="kmeans")
    parser.add_argument("--leiden-resolution", type=float, default=1.0)
    parser.add_argument("--k", type=int, default=8)
    parser.add_argument("--alpha", type=float, default=0.2)
    parser.add_argument("--max-em-iter", type=int, default=10)
    parser.add_argument("--max-icm-iter", type=int, default=5)
    parser.add_argument("--tol", type=float, default=1e-3)
    parser.add_argument("--energy-tol", type=float, default=None)
    parser.add_argument("--random-state", type=int, default=0)
    parser.add_argument("--spot-size", type=float, default=0.034)
    parser.add_argument(
        "--save-adata",
        action="store_true",
        help="Save the AnnData object with predicted columns written into obs/uns.",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=DEFAULT_OUTPUT_ROOT,
        help="Root output directory. A parameter-named run directory will be created inside it.",
    )
    return parser.parse_args()


def slugify(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]+", "-", value).strip("-")


def format_value(value: object) -> str:
    if value is None:
        return "none"
    if isinstance(value, float):
        text = f"{value:.6g}"
        return text.replace(".", "p")
    return slugify(str(value))


def build_run_dir(args: argparse.Namespace) -> Path:
    parts = [
        f"label-{format_value(args.label_key)}",
        f"beta-{format_value(args.beta)}",
        f"K-{format_value(args.n_regions)}",
        f"init-{format_value(args.init_method)}",
        f"k-{format_value(args.k)}",
        f"alpha-{format_value(args.alpha)}",
        f"em-{format_value(args.max_em_iter)}",
        f"icm-{format_value(args.max_icm_iter)}",
        f"seed-{format_value(args.random_state)}",
    ]
    if args.init_method == "leiden":
        parts.append(f"leiden-res-{format_value(args.leiden_resolution)}")
    return args.output_root / ("hmrf_" + "_".join(parts))


def ensure_reference_colors(adata, label_key: str) -> None:
    color_key = f"{label_key}_colors"
    if color_key in adata.uns:
        return

    labels = adata.obs[label_key].astype("category")
    adata.obs[label_key] = labels
    n_categories = len(labels.cat.categories)
    cmap = plt.get_cmap("tab20")
    colors = [matplotlib.colors.to_hex(cmap(i % 20)) for i in range(n_categories)]
    adata.uns[color_key] = colors


def save_metrics(results: dict[str, float], output_dir: Path) -> None:
    pd.DataFrame([results]).to_csv(output_dir / "metrics.csv", index=False)
    json_ready = {
        key: (value.item() if hasattr(value, "item") else value)
        for key, value in results.items()
    }
    with open(output_dir / "metrics.json", "w", encoding="utf-8") as fh:
        json.dump(json_ready, fh, indent=2)


def save_metrics_summary(rows: list[dict[str, object]], output_dir: Path) -> None:
    df = pd.DataFrame(rows)
    df.to_csv(output_dir / "metrics_summary.csv", index=False)
    json_ready = []
    for row in df.to_dict(orient="records"):
        json_ready.append(
            {
                key: (value.item() if hasattr(value, "item") else value)
                for key, value in row.items()
            }
        )
    with open(output_dir / "metrics_summary.json", "w", encoding="utf-8") as fh:
        json.dump(json_ready, fh, indent=2)


def save_model_result(result, output_dir: Path) -> None:
    np.savez_compressed(
        output_dir / "hmrf_result.npz",
        states=result.states,
        mu=result.mu,
        sigma=result.sigma,
        energy_history=np.asarray(result.energy_history, dtype=float),
        converged=np.asarray(result.converged),
        n_iter=np.asarray(result.n_iter),
    )


def save_states(states: np.ndarray, output_dir: Path, stem: str) -> None:
    np.save(output_dir / f"{stem}_states.npy", np.asarray(states, dtype=int))
    pd.Series(np.asarray(states, dtype=int), name="state").to_csv(
        output_dir / f"{stem}_states.csv",
        index_label="cell_index",
    )


def save_confusion_outputs(adata, label_key: str, pred_key: str, output_dir: Path) -> None:
    tab = pd.crosstab(adata.obs[pred_key], adata.obs[label_key])
    tab.to_csv(output_dir / "confusion_counts.csv")

    tab_norm = tab.div(tab.sum(axis=1).replace(0, np.nan), axis=0).fillna(0.0)
    tab_norm.to_csv(output_dir / "confusion_normalized.csv")

    fig, ax = plt.subplots(figsize=(10, 8))
    im = ax.imshow(tab_norm.values, aspect="auto", cmap="viridis")
    ax.set_title("Cluster to Region Normalized Confusion")
    ax.set_xlabel(label_key)
    ax.set_ylabel(pred_key)
    ax.set_xticks(np.arange(len(tab_norm.columns)))
    ax.set_xticklabels(tab_norm.columns, rotation=90)
    ax.set_yticks(np.arange(len(tab_norm.index)))
    ax.set_yticklabels(tab_norm.index)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(output_dir / "confusion_heatmap.png", dpi=200, bbox_inches="tight")
    plt.close(fig)

    per_region_acc = tab.max(axis=0) / tab.sum(axis=0)
    per_region_acc.sort_values().to_csv(output_dir / "per_region_accuracy.csv", header=["accuracy"])

    fig, ax = plt.subplots(figsize=(10, 4))
    per_region_acc.sort_values().plot(kind="bar", ax=ax, color="#2c7fb8")
    ax.set_ylabel("Accuracy")
    ax.set_title("Per-region classification accuracy")
    fig.tight_layout()
    fig.savefig(output_dir / "per_region_accuracy.png", dpi=200, bbox_inches="tight")
    plt.close(fig)


def assign_prediction_outputs(
    adata,
    cluster_col: str,
    states: np.ndarray,
    label_key: str,
) -> tuple[str, dict[str, str]]:
    adata.obs[cluster_col] = pd.Categorical(np.asarray(states, dtype=int).astype(str))
    mapping = remap_clusters_to_atlas(adata, cluster_col, label_key)
    named_col = f"{cluster_col}_named"
    adata.obs[named_col] = adata.obs[named_col].astype("category")
    align_visualization_colors(adata, named_col, label_key)
    return named_col, mapping


def save_mapping(mapping: dict[str, str], output_dir: Path, filename: str = "label_mapping.json") -> None:
    with open(output_dir / filename, "w", encoding="utf-8") as fh:
        json.dump(mapping, fh, indent=2, sort_keys=True)


def save_spatial_plot(
    adata,
    color_key: str,
    title: str,
    output_path: Path,
    spot_size: float,
) -> None:
    sc.pl.spatial(
        adata,
        color=color_key,
        title=title,
        spot_size=spot_size,
        frameon=False,
        show=False,
    )
    fig = plt.gcf()
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def compute_initial_states(
    embedding: np.ndarray,
    adata,
    args: argparse.Namespace,
) -> np.ndarray:
    if args.init_method == "kmeans":
        kmeans = KMeans(
            n_clusters=args.n_regions,
            n_init=10,
            random_state=args.random_state,
        )
        return kmeans.fit_predict(embedding)

    init_adata = sc.AnnData(X=np.asarray(embedding))
    sc.pp.neighbors(
        init_adata,
        n_neighbors=args.k,
        use_rep="X",
        random_state=args.random_state,
    )
    sc.tl.leiden(
        init_adata,
        resolution=args.leiden_resolution,
        random_state=args.random_state,
        key_added="init_leiden",
    )
    codes = pd.Categorical(init_adata.obs["init_leiden"]).codes.astype(int)
    n_clusters = int(np.unique(codes).size)
    if n_clusters > args.n_regions:
        raise ValueError(
            f"Leiden initialization produced {n_clusters} clusters, which exceeds n_regions={args.n_regions}. "
            "Increase n_regions or lower --leiden-resolution."
        )
    return codes


def validate_inputs(adata, label_key: str) -> None:
    if label_key not in adata.obs:
        raise KeyError(f"label_key '{label_key}' not found in adata.obs")
    if DEFAULT_EMBEDDING_KEY not in adata.obsm:
        raise KeyError(f"embedding_key '{DEFAULT_EMBEDDING_KEY}' not found in adata.obsm")
    if DEFAULT_COORD_KEY not in adata.obsm:
        raise KeyError(f"coord_key '{DEFAULT_COORD_KEY}' not found in adata.obsm")


def main() -> None:
    args = parse_args()
    run_dir = build_run_dir(args)
    run_dir.mkdir(parents=True, exist_ok=True)

    config = {
        "adata_path": str(args.adata_path),
        "label_key": args.label_key,
        "embedding_key": DEFAULT_EMBEDDING_KEY,
        "coord_key": DEFAULT_COORD_KEY,
        "beta": args.beta,
        "n_regions": args.n_regions,
        "init_method": args.init_method,
        "leiden_resolution": args.leiden_resolution,
        "k": args.k,
        "alpha": args.alpha,
        "max_em_iter": args.max_em_iter,
        "max_icm_iter": args.max_icm_iter,
        "tol": args.tol,
        "energy_tol": args.energy_tol,
        "random_state": args.random_state,
        "spot_size": args.spot_size,
        "save_adata": args.save_adata,
        "output_dir": str(run_dir),
    }
    with open(run_dir / "config.json", "w", encoding="utf-8") as fh:
        json.dump(config, fh, indent=2)

    adata = sc.read_h5ad(args.adata_path)
    validate_inputs(adata, args.label_key)
    ensure_reference_colors(adata, args.label_key)

    W_aw = build_anatomy_weighted_knn_graph(
        adata,
        coord_key=DEFAULT_COORD_KEY,
        label_key=args.label_key,
        k=args.k,
        alpha=args.alpha,
    )
    embedding = adata.obsm[DEFAULT_EMBEDDING_KEY]
    init_states = compute_initial_states(embedding, adata, args)

    baseline_dir = run_dir / f"baseline_{args.init_method}"
    hmrf_dir = run_dir / "hmrf"
    baseline_dir.mkdir(parents=True, exist_ok=True)
    hmrf_dir.mkdir(parents=True, exist_ok=True)

    baseline_col = f"baseline_{args.init_method}"
    baseline_named_col, baseline_mapping = assign_prediction_outputs(
        adata,
        baseline_col,
        init_states,
        args.label_key,
    )
    save_states(init_states, baseline_dir, "baseline")
    save_mapping(baseline_mapping, baseline_dir)
    save_spatial_plot(
        adata,
        baseline_named_col,
        title=f"Initialization Baseline ({args.init_method})",
        output_path=baseline_dir / "spatial.png",
        spot_size=args.spot_size,
    )
    baseline_metrics = evaluate_all(
        adata,
        label_key=args.label_key,
        pred_key=baseline_named_col,
        embedding_key=DEFAULT_EMBEDDING_KEY,
        W=W_aw,
    )
    save_metrics(baseline_metrics, baseline_dir)
    save_confusion_outputs(adata, args.label_key, baseline_named_col, baseline_dir)

    model = AW_HMRF(
        n_regions=args.n_regions,
        beta=args.beta,
        max_em_iter=args.max_em_iter,
        max_icm_iter=args.max_icm_iter,
        tol=args.tol,
        energy_tol=args.energy_tol,
        random_state=args.random_state,
    )
    result = model.fit(embedding, W_aw, initial_states=init_states)
    results = {float(args.beta): result}

    save_model_result(result, hmrf_dir)
    save_states(result.states, hmrf_dir, "hmrf")

    plot_energy_history(
        result,
        save_path=hmrf_dir / "energy_history.png",
        show=False,
    )

    plot_hmrf_spatial_results(
        adata,
        results,
        label_key=args.label_key,
        spot_size=args.spot_size,
        output_dir=hmrf_dir,
        show=False,
    )

    pred_key = f"AW_HMRF_beta_{float(args.beta)}_named"
    metrics = evaluate_all(
        adata,
        label_key=args.label_key,
        pred_key=pred_key,
        embedding_key=DEFAULT_EMBEDDING_KEY,
        W=W_aw,
    )
    save_metrics(metrics, hmrf_dir)
    save_confusion_outputs(adata, args.label_key, pred_key, hmrf_dir)
    hmrf_mapping = remap_clusters_to_atlas(adata, f"AW_HMRF_beta_{float(args.beta)}", args.label_key)
    save_mapping(hmrf_mapping, hmrf_dir)

    save_metrics_summary(
        [
            {
                "result_name": f"baseline_{args.init_method}",
                "n_clusters": int(np.unique(init_states).size),
                **baseline_metrics,
            },
            {
                "result_name": "hmrf",
                "n_clusters": int(np.unique(result.states).size),
                **metrics,
            },
        ],
        run_dir,
    )

    if args.save_adata:
        adata.write(run_dir / "adata_with_hmrf.h5ad")

    print(f"Run complete. Outputs saved to: {run_dir}")


if __name__ == "__main__":
    main()
