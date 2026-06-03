import pandas as pd

from src.models.datasets import SMILES_COL, TARGET_COL
from src.models.plots import (
    build_dataset_summary,
    metrics_to_frame,
    save_training_plots,
)


def _split_frame(
    split_idx: int,
    targets: list[float],
) -> pd.DataFrame:
    return pd.DataFrame(
        {
            SMILES_COL: [f"C{split_idx}{idx}" for idx in range(len(targets))],
            TARGET_COL: targets,
            "scaffold": [
                f"scaffold_{split_idx}_{idx}"
                for idx in range(len(targets))
            ],
        }
    )


def test_build_dataset_summary_includes_quality_and_split_details():
    splits = {
        "train_core": _split_frame(0, [0, 1]),
        "early_stop": _split_frame(1, [1, 1]),
        "validation": _split_frame(2, [0, 1]),
        "test": _split_frame(3, [0, 0]),
    }
    valid_df = pd.concat(splits.values(), ignore_index=True)
    raw_df = pd.concat(
        [
            valid_df[[SMILES_COL, TARGET_COL]],
            pd.DataFrame({SMILES_COL: ["bad"], TARGET_COL: [1]}),
        ],
        ignore_index=True,
    )
    rejected_df = pd.DataFrame({SMILES_COL: ["bad"], TARGET_COL: [1]})

    summary = build_dataset_summary(
        dataset_name="Example",
        task_type="classification",
        raw_df=raw_df,
        valid_df=valid_df,
        rejected_df=rejected_df,
        splits=splits,
    )

    assert summary["n_raw_rows"] == 9
    assert summary["n_valid_rows"] == 8
    assert summary["n_rejected_rows"] == 1
    assert summary["class_counts"] == {"0": 4, "1": 4}
    assert summary["splits"]["train_core"]["n_scaffolds"] == 2
    assert summary["splits"]["early_stop"]["class_counts"] == {"1": 2}


def test_metrics_to_frame_flattens_split_metrics_in_order():
    metrics = {
        "test": {"rmse": 1.0},
        "train_core": {"rmse": 0.5},
    }

    frame = metrics_to_frame(metrics)

    assert frame["split"].tolist() == ["train_core", "test"]
    assert frame["metric"].tolist() == ["rmse", "rmse"]


def test_save_training_plots_writes_regression_diagnostics(tmp_path):
    splits = {
        "train_core": _split_frame(0, [0.0, 1.0, 2.0]),
        "early_stop": _split_frame(1, [0.5, 1.5]),
        "validation": _split_frame(2, [1.0, 2.0]),
        "test": _split_frame(3, [1.5, 2.5]),
    }
    predictions = []

    for split_name, split_df in splits.items():
        frame = split_df.rename(columns={TARGET_COL: "y_true"})[
            [SMILES_COL, "scaffold", "y_true"]
        ].copy()
        frame["split"] = split_name
        frame["y_pred"] = frame["y_true"] + 0.1
        frame["residual"] = 0.1
        frame["absolute_error"] = 0.1
        predictions.append(frame)

    metrics = {
        split_name: {"mae": 0.1, "rmse": 0.1, "r2": 0.9}
        for split_name in splits
    }

    paths = save_training_plots(
        splits=splits,
        task_type="regression",
        metrics=metrics,
        predictions=pd.concat(predictions, ignore_index=True),
        artifact_dir=tmp_path,
    )

    assert {path.name for path in paths} == {
        "dataset_split_overview.png",
        "target_distribution_by_split.png",
        "metrics_by_split.png",
        "regression_diagnostics.png",
    }
    assert all(path.exists() and path.stat().st_size > 0 for path in paths)


def test_save_training_plots_writes_classification_diagnostics(tmp_path):
    splits = {
        "train_core": _split_frame(0, [0, 1, 0, 1]),
        "early_stop": _split_frame(1, [0, 1]),
        "validation": _split_frame(2, [0, 1]),
        "test": _split_frame(3, [0, 1]),
    }
    predictions = []

    for split_name, split_df in splits.items():
        frame = split_df.rename(columns={TARGET_COL: "y_true"})[
            [SMILES_COL, "scaffold", "y_true"]
        ].copy()
        frame["split"] = split_name
        frame["y_proba"] = [0.2 if value == 0 else 0.8 for value in frame["y_true"]]
        frame["y_pred"] = (frame["y_proba"] >= 0.5).astype(int)
        frame["correct"] = 1
        predictions.append(frame)

    metrics = {
        split_name: {
            "accuracy": 1.0,
            "precision": 1.0,
            "recall": 1.0,
            "f1": 1.0,
            "roc_auc": 1.0,
            "auprc": 1.0,
        }
        for split_name in splits
    }

    paths = save_training_plots(
        splits=splits,
        task_type="classification",
        metrics=metrics,
        predictions=pd.concat(predictions, ignore_index=True),
        artifact_dir=tmp_path,
    )

    assert {path.name for path in paths} == {
        "dataset_split_overview.png",
        "target_distribution_by_split.png",
        "metrics_by_split.png",
        "classification_curves.png",
        "classification_confusion_matrices.png",
    }
    assert all(path.exists() and path.stat().st_size > 0 for path in paths)
