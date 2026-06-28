"""Train reproducible reference baselines on the saved XGBoost scaffold splits.

The XGBoost training pipeline writes the exact molecules assigned to each split
to ``predictions.csv``.  Reusing that manifest prevents a baseline comparison
from silently changing the held-out test set when dataset-library versions
change.  Regression tasks receive a training-mean dummy regressor;
classification tasks receive both a majority-class dummy classifier and an
L2-regularized logistic regression on the same saved molecular features.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.dummy import DummyClassifier, DummyRegressor
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from src.models.metrics import classification_metrics, regression_metrics


SPLIT_ORDER = ("train_core", "early_stop", "validation", "test")


def _load_bundle_inputs(
    dataset_name: str,
    model_root: Path,
) -> tuple[dict[str, Any], pd.DataFrame, Any]:
    bundle_dir = model_root / dataset_name
    metadata_path = bundle_dir / "metadata.json"
    predictions_path = bundle_dir / "predictions.csv"
    featurizer_path = bundle_dir / "featurizer.joblib"

    required = (metadata_path, predictions_path, featurizer_path)
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise FileNotFoundError(
            "The trained bundle is incomplete; missing: " + ", ".join(missing)
        )

    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    predictions = pd.read_csv(predictions_path)
    featurizer = joblib.load(featurizer_path)

    expected_splits = set(SPLIT_ORDER)
    actual_splits = set(predictions["split"].unique())
    if actual_splits != expected_splits:
        raise ValueError(
            f"Unexpected split manifest for {dataset_name}: {actual_splits}"
        )

    return metadata, predictions, featurizer


def _evaluate_regressor(
    model: Any,
    predictions: pd.DataFrame,
    featurizer: Any | None = None,
) -> dict[str, dict[str, float]]:
    results: dict[str, dict[str, float]] = {}
    for split_name in SPLIT_ORDER:
        split_df = predictions[predictions["split"] == split_name]
        if featurizer is None:
            X = np.zeros((len(split_df), 1), dtype=np.float32)
        else:
            X = featurizer.transform(split_df["smiles"].tolist())
        y_true = split_df["y_true"].to_numpy(dtype=float)
        results[split_name] = regression_metrics(y_true, model.predict(X))
    return results


def _positive_probability(model: Any, X: Any) -> np.ndarray:
    probabilities = model.predict_proba(X)
    classes = np.asarray(model.classes_)
    positive_columns = np.flatnonzero(classes == 1)
    if len(positive_columns) != 1:
        return np.zeros(X.shape[0], dtype=float)
    return probabilities[:, positive_columns[0]]


def _evaluate_classifier(
    model: Any,
    predictions: pd.DataFrame,
    featurizer: Any | None = None,
) -> dict[str, dict[str, float]]:
    results: dict[str, dict[str, float]] = {}
    for split_name in SPLIT_ORDER:
        split_df = predictions[predictions["split"] == split_name]
        if featurizer is None:
            X = np.zeros((len(split_df), 1), dtype=np.float32)
        else:
            X = sparse.csr_matrix(
                featurizer.transform(split_df["smiles"].tolist())
            )
        y_true = split_df["y_true"].to_numpy(dtype=int)
        y_pred = model.predict(X).astype(int)
        y_proba = _positive_probability(model, X)
        results[split_name] = classification_metrics(
            y_true,
            y_pred,
            y_proba,
        )
    return results


def train_baselines(
    dataset_name: str,
    model_root: Path,
    artifact_root: Path,
) -> dict[str, Any]:
    metadata, predictions, featurizer = _load_bundle_inputs(
        dataset_name,
        model_root,
    )
    task_type = metadata["task_type"]
    train_df = predictions[predictions["split"] == "train_core"]
    y_train = train_df["y_true"].to_numpy()

    output_dir = artifact_root / dataset_name
    output_dir.mkdir(parents=True, exist_ok=True)

    if task_type == "regression":
        dummy = DummyRegressor(strategy="mean")
        dummy.fit(np.zeros((len(train_df), 1)), y_train.astype(float))
        model_metrics = {
            "mean_regressor": _evaluate_regressor(dummy, predictions)
        }
        fitted_models = {"mean_regressor": dummy}
    elif task_type == "classification":
        y_train = y_train.astype(int)
        dummy = DummyClassifier(strategy="most_frequent")
        dummy.fit(np.zeros((len(train_df), 1)), y_train)

        X_train = sparse.csr_matrix(
            featurizer.transform(train_df["smiles"].tolist())
        )
        logistic = Pipeline(
            steps=[
                ("scale", StandardScaler(with_mean=False)),
                (
                    "classifier",
                    LogisticRegression(
                        solver="liblinear",
                        max_iter=3000,
                        random_state=int(metadata.get("random_seed", 42)),
                    ),
                ),
            ]
        )
        logistic.fit(X_train, y_train)

        model_metrics = {
            "majority_classifier": _evaluate_classifier(dummy, predictions),
            "logistic_regression": _evaluate_classifier(
                logistic,
                predictions,
                featurizer,
            ),
        }
        fitted_models = {
            "majority_classifier": dummy,
            "logistic_regression": logistic,
        }
    else:
        raise ValueError(f"Unsupported task type: {task_type}")

    result = {
        "dataset": dataset_name,
        "task_type": task_type,
        "random_seed": int(metadata.get("random_seed", 42)),
        "split_source": str(model_root / dataset_name / "predictions.csv"),
        "split_sizes": {
            name: int((predictions["split"] == name).sum())
            for name in SPLIT_ORDER
        },
        "models": model_metrics,
    }
    (output_dir / "metrics.json").write_text(
        json.dumps(result, indent=2),
        encoding="utf-8",
    )
    for model_name, model in fitted_models.items():
        joblib.dump(model, output_dir / f"{model_name}.joblib")

    return result


def _summary_rows(result: dict[str, Any]) -> list[dict[str, Any]]:
    metric_name = "rmse" if result["task_type"] == "regression" else "roc_auc"
    return [
        {
            "dataset": result["dataset"],
            "task_type": result["task_type"],
            "baseline": model_name,
            "test_metric": metric_name,
            "test_value": split_metrics["test"][metric_name],
        }
        for model_name, split_metrics in result["models"].items()
    ]


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Train reference baselines on saved scaffold splits."
    )
    parser.add_argument(
        "--dataset",
        action="append",
        help="Dataset name; repeat for multiple datasets. Defaults to all bundles.",
    )
    parser.add_argument(
        "--model-root",
        type=Path,
        default=Path("models/admet_xgboost"),
    )
    parser.add_argument(
        "--artifact-root",
        type=Path,
        default=Path("models/baselines"),
    )
    args = parser.parse_args()

    datasets = args.dataset or sorted(
        path.name
        for path in args.model_root.iterdir()
        if path.is_dir() and (path / "metadata.json").is_file()
    )
    results = [
        train_baselines(name, args.model_root, args.artifact_root)
        for name in datasets
    ]
    summary = pd.DataFrame(
        row
        for result in results
        for row in _summary_rows(result)
    )
    args.artifact_root.mkdir(parents=True, exist_ok=True)
    summary.to_csv(args.artifact_root / "summary.csv", index=False)
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
