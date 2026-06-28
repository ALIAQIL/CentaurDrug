import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import pytest

from src.models.train_baseline import train_baselines


class _SimpleFeaturizer:
    def transform(self, smiles: list[str]) -> np.ndarray:
        return np.asarray(
            [[value.count("C"), value.count("N")] for value in smiles],
            dtype=np.float32,
        )


def _write_bundle(
    model_root: Path,
    dataset_name: str,
    task_type: str,
    predictions: pd.DataFrame,
) -> None:
    bundle_dir = model_root / dataset_name
    bundle_dir.mkdir(parents=True)
    (bundle_dir / "metadata.json").write_text(
        json.dumps({"task_type": task_type, "random_seed": 7}),
        encoding="utf-8",
    )
    predictions.to_csv(bundle_dir / "predictions.csv", index=False)
    joblib.dump(_SimpleFeaturizer(), bundle_dir / "featurizer.joblib")


def test_regression_baseline_uses_only_train_core_mean(tmp_path: Path) -> None:
    predictions = pd.DataFrame(
        {
            "split": [
                "train_core",
                "train_core",
                "early_stop",
                "early_stop",
                "validation",
                "validation",
                "test",
                "test",
            ],
            "smiles": ["C", "CC", "CCC", "CCCC", "N", "NN", "CN", "CCN"],
            "y_true": [1.0, 3.0, 20.0, 22.0, 30.0, 32.0, 4.0, 0.0],
        }
    )
    model_root = tmp_path / "models"
    artifact_root = tmp_path / "baselines"
    _write_bundle(model_root, "regression", "regression", predictions)

    result = train_baselines("regression", model_root, artifact_root)

    assert result["split_sizes"]["train_core"] == 2
    assert result["models"]["mean_regressor"]["test"]["rmse"] == pytest.approx(2.0)
    assert (artifact_root / "regression" / "mean_regressor.joblib").is_file()
    assert (artifact_root / "regression" / "metrics.json").is_file()


def test_classification_baseline_writes_dummy_and_logistic_models(
    tmp_path: Path,
) -> None:
    split_names = ("train_core", "early_stop", "validation", "test")
    predictions = pd.DataFrame(
        [
            {"split": split, "smiles": smiles, "y_true": label}
            for split in split_names
            for smiles, label in (("C", 0), ("CC", 0), ("N", 1), ("NN", 1))
        ]
    )
    model_root = tmp_path / "models"
    artifact_root = tmp_path / "baselines"
    _write_bundle(model_root, "classification", "classification", predictions)

    result = train_baselines("classification", model_root, artifact_root)

    models = result["models"]
    assert models["majority_classifier"]["test"]["roc_auc"] == pytest.approx(0.5)
    assert models["logistic_regression"]["test"]["roc_auc"] == pytest.approx(1.0)
    assert (artifact_root / "classification" / "majority_classifier.joblib").is_file()
    assert (artifact_root / "classification" / "logistic_regression.joblib").is_file()


def test_baseline_rejects_an_incomplete_split_manifest(tmp_path: Path) -> None:
    predictions = pd.DataFrame(
        {
            "split": ["train_core", "test"],
            "smiles": ["C", "N"],
            "y_true": [0, 1],
        }
    )
    model_root = tmp_path / "models"
    _write_bundle(model_root, "classification", "classification", predictions)

    with pytest.raises(ValueError, match="Unexpected split manifest"):
        train_baselines("classification", model_root, tmp_path / "baselines")
