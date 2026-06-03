from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Iterable

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import (
    auc,
    average_precision_score,
    confusion_matrix,
    precision_recall_curve,
    roc_curve,
)

from src.models.datasets import SMILES_COL, TARGET_COL


SPLIT_ORDER = ("train_core", "early_stop", "validation", "test")
PLOT_DPI = 160


def _split_names_present(values: Iterable[str]) -> list[str]:
    present = set(values)
    ordered = [name for name in SPLIT_ORDER if name in present]
    extras = sorted(present.difference(SPLIT_ORDER))
    return ordered + extras


def _json_float(value: Any) -> float | None:
    if pd.isna(value):
        return None

    return float(value)


def _target_summary(series: pd.Series) -> Dict[str, float | int | None]:
    return {
        "count": int(series.count()),
        "mean": _json_float(series.mean()),
        "std": _json_float(series.std()),
        "min": _json_float(series.min()),
        "median": _json_float(series.median()),
        "max": _json_float(series.max()),
    }


def build_dataset_summary(
    dataset_name: str,
    task_type: str,
    raw_df: pd.DataFrame,
    valid_df: pd.DataFrame,
    rejected_df: pd.DataFrame,
    splits: Dict[str, pd.DataFrame],
) -> Dict[str, Any]:
    """
    Summarize dataset quality, target distribution, and split composition.
    """

    split_summaries: Dict[str, Any] = {}

    for split_name in _split_names_present(splits.keys()):
        split_df = splits[split_name]
        target = split_df[TARGET_COL]
        split_summary: Dict[str, Any] = {
            "n_molecules": int(len(split_df)),
            "n_scaffolds": int(split_df["scaffold"].nunique()),
            "target": _target_summary(target),
        }

        if task_type == "classification":
            class_counts = target.astype(int).value_counts().sort_index()
            split_summary["class_counts"] = {
                str(int(label)): int(count)
                for label, count in class_counts.items()
            }
            split_summary["positive_fraction"] = _json_float(target.mean())

        split_summaries[split_name] = split_summary

    summary: Dict[str, Any] = {
        "dataset": dataset_name,
        "task_type": task_type,
        "n_raw_rows": int(len(raw_df)),
        "n_valid_rows": int(len(valid_df)),
        "n_rejected_rows": int(len(rejected_df)),
        "rejected_fraction": (
            float(len(rejected_df) / len(raw_df))
            if len(raw_df) > 0
            else 0.0
        ),
        "overall_target": _target_summary(valid_df[TARGET_COL]),
        "splits": split_summaries,
    }

    if task_type == "classification":
        class_counts = valid_df[TARGET_COL].astype(int).value_counts().sort_index()
        summary["class_counts"] = {
            str(int(label)): int(count)
            for label, count in class_counts.items()
        }
        summary["positive_fraction"] = _json_float(valid_df[TARGET_COL].mean())

    return summary


def write_dataset_summary(
    summary: Dict[str, Any],
    path: Path,
) -> None:
    path.write_text(
        json.dumps(summary, indent=2),
        encoding="utf-8",
    )


def collect_split_predictions(
    model,
    task_type: str,
    splits: Dict[str, pd.DataFrame],
    featurizer,
) -> pd.DataFrame:
    """
    Build a tidy prediction table for model diagnostics and plots.
    """

    frames = []

    for split_name in _split_names_present(splits.keys()):
        split_df = splits[split_name]
        X = featurizer.transform(split_df[SMILES_COL].tolist())
        y_true = split_df[TARGET_COL].to_numpy()

        report = pd.DataFrame(
            {
                "split": split_name,
                "smiles": split_df[SMILES_COL].to_numpy(),
                "scaffold": split_df["scaffold"].to_numpy(),
                "y_true": y_true,
            }
        )

        if task_type == "regression":
            y_pred = model.predict(X)
            report["y_pred"] = y_pred
            report["residual"] = y_pred - y_true
            report["absolute_error"] = np.abs(report["residual"])

        elif task_type == "classification":
            y_proba = model.predict_proba(X)[:, 1]
            y_pred = (y_proba >= 0.5).astype(int)
            report["y_pred"] = y_pred
            report["y_proba"] = y_proba
            report["correct"] = (y_pred == y_true.astype(int)).astype(int)

        else:
            raise ValueError(f"Unsupported task_type: {task_type}")

        frames.append(report)

    return pd.concat(frames, ignore_index=True)


def metrics_to_frame(
    metrics: Dict[str, Dict[str, float]],
) -> pd.DataFrame:
    rows = []

    for split_name in _split_names_present(metrics.keys()):
        for metric_name, value in metrics[split_name].items():
            rows.append(
                {
                    "split": split_name,
                    "metric": metric_name,
                    "value": float(value),
                }
            )

    return pd.DataFrame(rows)


def _save_figure(
    fig: plt.Figure,
    path: Path,
) -> Path:
    fig.tight_layout()
    fig.savefig(
        path,
        dpi=PLOT_DPI,
        bbox_inches="tight",
    )
    plt.close(fig)
    return path


def _bar_labels(ax: plt.Axes) -> None:
    for container in ax.containers:
        ax.bar_label(container, fmt="%.0f", fontsize=8)


def plot_dataset_split_overview(
    splits: Dict[str, pd.DataFrame],
    output_dir: Path,
) -> Path:
    split_names = _split_names_present(splits.keys())
    n_molecules = [len(splits[name]) for name in split_names]
    n_scaffolds = [splits[name]["scaffold"].nunique() for name in split_names]

    fig, axes = plt.subplots(1, 2, figsize=(11, 4))

    axes[0].bar(split_names, n_molecules, color="#3b82f6")
    axes[0].set_title("Molecules by split")
    axes[0].set_ylabel("Count")
    axes[0].tick_params(axis="x", rotation=25)
    _bar_labels(axes[0])

    axes[1].bar(split_names, n_scaffolds, color="#10b981")
    axes[1].set_title("Scaffolds by split")
    axes[1].set_ylabel("Count")
    axes[1].tick_params(axis="x", rotation=25)
    _bar_labels(axes[1])

    return _save_figure(fig, output_dir / "dataset_split_overview.png")


def plot_target_distribution(
    splits: Dict[str, pd.DataFrame],
    task_type: str,
    output_dir: Path,
) -> Path:
    split_names = _split_names_present(splits.keys())

    if task_type == "regression":
        fig, ax = plt.subplots(figsize=(8, 5))

        for split_name in split_names:
            values = splits[split_name][TARGET_COL].astype(float)
            ax.hist(
                values,
                bins=30,
                alpha=0.38,
                density=True,
                label=split_name,
            )

        ax.set_title("Target distribution by split")
        ax.set_xlabel("Target")
        ax.set_ylabel("Density")
        ax.legend()

    elif task_type == "classification":
        class_labels = [0, 1]
        counts = np.array(
            [
                [
                    int((splits[split_name][TARGET_COL].astype(int) == label).sum())
                    for label in class_labels
                ]
                for split_name in split_names
            ]
        )

        totals = counts.sum(axis=1, keepdims=True)
        proportions = np.divide(
            counts,
            totals,
            out=np.zeros_like(counts, dtype=float),
            where=totals != 0,
        )

        fig, ax = plt.subplots(figsize=(8, 5))
        x = np.arange(len(split_names))
        width = 0.35

        ax.bar(
            x - width / 2,
            proportions[:, 0],
            width,
            label="class 0",
            color="#64748b",
        )
        ax.bar(
            x + width / 2,
            proportions[:, 1],
            width,
            label="class 1",
            color="#ef4444",
        )
        ax.set_title("Class balance by split")
        ax.set_xlabel("Split")
        ax.set_ylabel("Fraction")
        ax.set_xticks(x)
        ax.set_xticklabels(split_names, rotation=25)
        ax.set_ylim(0, 1)
        ax.legend()

    else:
        raise ValueError(f"Unsupported task_type: {task_type}")

    return _save_figure(fig, output_dir / "target_distribution_by_split.png")


def plot_metrics_by_split(
    metrics: Dict[str, Dict[str, float]],
    output_dir: Path,
) -> Path:
    metrics_df = metrics_to_frame(metrics)
    split_names = _split_names_present(metrics_df["split"])
    metric_names = list(metrics_df["metric"].drop_duplicates())

    ncols = 2
    nrows = int(np.ceil(len(metric_names) / ncols))
    fig, axes = plt.subplots(
        nrows,
        ncols,
        figsize=(11, max(3.5, 3.1 * nrows)),
        squeeze=False,
    )

    for ax, metric_name in zip(axes.ravel(), metric_names):
        values = (
            metrics_df[metrics_df["metric"] == metric_name]
            .set_index("split")
            .reindex(split_names)["value"]
        )
        ax.bar(split_names, values, color="#6366f1")
        ax.set_title(metric_name)
        ax.tick_params(axis="x", rotation=25)

        finite_values = values[np.isfinite(values)]
        if len(finite_values) and finite_values.min() >= 0 and finite_values.max() <= 1:
            ax.set_ylim(0, 1)

        for idx, value in enumerate(values):
            if pd.notna(value):
                ax.text(
                    idx,
                    value,
                    f"{value:.3f}",
                    ha="center",
                    va="bottom",
                    fontsize=8,
                )

    for ax in axes.ravel()[len(metric_names):]:
        ax.axis("off")

    fig.suptitle("Model metrics by split", y=1.02)

    return _save_figure(fig, output_dir / "metrics_by_split.png")


def plot_regression_diagnostics(
    predictions: pd.DataFrame,
    output_dir: Path,
) -> Path:
    split_names = _split_names_present(predictions["split"])
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    y_min = float(
        min(predictions["y_true"].min(), predictions["y_pred"].min())
    )
    y_max = float(
        max(predictions["y_true"].max(), predictions["y_pred"].max())
    )

    for split_name in split_names:
        split_df = predictions[predictions["split"] == split_name]
        plot_df = split_df

        if len(plot_df) > 2500:
            plot_df = plot_df.sample(n=2500, random_state=42)

        axes[0].scatter(
            plot_df["y_true"],
            plot_df["y_pred"],
            s=12,
            alpha=0.45,
            label=split_name,
        )

        axes[1].hist(
            split_df["residual"],
            bins=35,
            alpha=0.38,
            density=True,
            label=split_name,
        )

    axes[0].plot([y_min, y_max], [y_min, y_max], color="#111827", linewidth=1)
    axes[0].set_title("Predicted vs actual")
    axes[0].set_xlabel("Actual")
    axes[0].set_ylabel("Predicted")
    axes[0].legend(fontsize=8)

    axes[1].axvline(0, color="#111827", linewidth=1)
    axes[1].set_title("Residual distribution")
    axes[1].set_xlabel("Predicted - actual")
    axes[1].set_ylabel("Density")
    axes[1].legend(fontsize=8)

    return _save_figure(fig, output_dir / "regression_diagnostics.png")


def plot_classification_curves(
    predictions: pd.DataFrame,
    output_dir: Path,
) -> Path:
    split_names = _split_names_present(predictions["split"])
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    axes[0].plot([0, 1], [0, 1], color="#94a3b8", linestyle="--", linewidth=1)

    for split_name in split_names:
        split_df = predictions[predictions["split"] == split_name]

        if split_df["y_true"].nunique() < 2:
            continue

        fpr, tpr, _ = roc_curve(split_df["y_true"].astype(int), split_df["y_proba"])
        precision, recall, _ = precision_recall_curve(
            split_df["y_true"].astype(int),
            split_df["y_proba"],
        )
        average_precision = average_precision_score(
            split_df["y_true"].astype(int),
            split_df["y_proba"],
        )

        axes[0].plot(
            fpr,
            tpr,
            label=f"{split_name} AUC={auc(fpr, tpr):.3f}",
        )
        axes[1].plot(
            recall,
            precision,
            label=f"{split_name} AUPRC={average_precision:.3f}",
        )

    axes[0].set_title("ROC curves")
    axes[0].set_xlabel("False positive rate")
    axes[0].set_ylabel("True positive rate")
    axes[0].set_xlim(0, 1)
    axes[0].set_ylim(0, 1)
    axes[0].legend(fontsize=8)

    axes[1].set_title("Precision-recall curves")
    axes[1].set_xlabel("Recall")
    axes[1].set_ylabel("Precision")
    axes[1].set_xlim(0, 1)
    axes[1].set_ylim(0, 1)
    axes[1].legend(fontsize=8)

    return _save_figure(fig, output_dir / "classification_curves.png")


def plot_classification_confusion_matrices(
    predictions: pd.DataFrame,
    output_dir: Path,
) -> Path:
    split_names = _split_names_present(predictions["split"])
    ncols = 2
    nrows = int(np.ceil(len(split_names) / ncols))
    fig, axes = plt.subplots(
        nrows,
        ncols,
        figsize=(9, max(3.8, 3.8 * nrows)),
        squeeze=False,
    )

    for ax, split_name in zip(axes.ravel(), split_names):
        split_df = predictions[predictions["split"] == split_name]
        cm = confusion_matrix(
            split_df["y_true"].astype(int),
            split_df["y_pred"].astype(int),
            labels=[0, 1],
        )

        image = ax.imshow(cm, cmap="Blues")
        ax.set_title(split_name)
        ax.set_xlabel("Predicted")
        ax.set_ylabel("Actual")
        ax.set_xticks([0, 1])
        ax.set_yticks([0, 1])

        for row in range(cm.shape[0]):
            for col in range(cm.shape[1]):
                ax.text(
                    col,
                    row,
                    str(cm[row, col]),
                    ha="center",
                    va="center",
                    color="#111827",
                    fontsize=10,
                )

        fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)

    for ax in axes.ravel()[len(split_names):]:
        ax.axis("off")

    fig.suptitle("Confusion matrices by split", y=1.02)

    return _save_figure(fig, output_dir / "classification_confusion_matrices.png")


def save_training_plots(
    splits: Dict[str, pd.DataFrame],
    task_type: str,
    metrics: Dict[str, Dict[str, float]],
    predictions: pd.DataFrame,
    artifact_dir: Path,
) -> list[Path]:
    """
    Save dataset exploration and model diagnostic plots for a training run.
    """

    output_dir = artifact_dir / "plots"
    output_dir.mkdir(parents=True, exist_ok=True)

    paths = [
        plot_dataset_split_overview(splits, output_dir),
        plot_target_distribution(splits, task_type, output_dir),
        plot_metrics_by_split(metrics, output_dir),
    ]

    if task_type == "regression":
        paths.append(plot_regression_diagnostics(predictions, output_dir))

    elif task_type == "classification":
        paths.extend(
            [
                plot_classification_curves(predictions, output_dir),
                plot_classification_confusion_matrices(predictions, output_dir),
            ]
        )

    else:
        raise ValueError(f"Unsupported task_type: {task_type}")

    return paths
