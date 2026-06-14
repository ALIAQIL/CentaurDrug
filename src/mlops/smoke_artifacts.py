from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Iterable

import joblib
import numpy as np

from src.tools.evaluator import MODEL_PATHS


class SmokeFeaturizer:
    def transform(self, smiles: Iterable[str]):
        rows = [[float(len(str(value))), 1.0] for value in smiles]
        return np.asarray(rows, dtype=np.float32)


class ConstantRegressor:
    def __init__(self, value: float):
        self.value = float(value)

    def predict(self, X):
        return np.full(shape=(len(X),), fill_value=self.value, dtype=np.float32)


class ConstantClassifier:
    def __init__(self, probability: float):
        self.probability = float(probability)

    def predict_proba(self, X):
        probabilities = np.full(
            shape=(len(X),),
            fill_value=self.probability,
            dtype=np.float32,
        )
        return np.column_stack([1.0 - probabilities, probabilities])


SMOKE_MODEL_CONFIG = {
    "solubility": {
        "task_type": "regression",
        "prediction_unit": "smoke_log_solubility",
        "model": ConstantRegressor(-2.0),
    },
    "lipophilicity": {
        "task_type": "regression",
        "prediction_unit": "smoke_logd",
        "model": ConstantRegressor(2.0),
    },
    "ames": {
        "task_type": "classification",
        "positive_class_name": "toxic_or_mutagenic",
        "model": ConstantClassifier(0.1),
    },
    "herg": {
        "task_type": "classification",
        "positive_class_name": "hERG_risk",
        "model": ConstantClassifier(0.1),
    },
    "cyp3a4": {
        "task_type": "classification",
        "positive_class_name": "CYP3A4_inhibitor",
        "model": ConstantClassifier(0.1),
    },
}


def create_smoke_model_artifacts(output_dir: str | Path) -> Path:
    root = Path(output_dir)
    root.mkdir(parents=True, exist_ok=True)

    for short_name, dataset_dir in MODEL_PATHS.items():
        config = SMOKE_MODEL_CONFIG[short_name]
        artifact_dir = root / dataset_dir
        artifact_dir.mkdir(parents=True, exist_ok=True)

        metadata = {
            "project": "centaurdrug",
            "dataset": dataset_dir,
            "task_type": config["task_type"],
            "prediction_unit": config.get("prediction_unit"),
            "positive_class_name": config.get("positive_class_name"),
            "features": {
                "morgan_radius": 2,
                "morgan_n_bits": 2048,
            },
            "smoke_artifact": True,
        }

        joblib.dump(config["model"], artifact_dir / "model.joblib")
        joblib.dump(SmokeFeaturizer(), artifact_dir / "featurizer.joblib")
        (artifact_dir / "metadata.json").write_text(
            json.dumps(metadata, indent=2),
            encoding="utf-8",
        )

    return root


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Create lightweight ADMET model artifacts for container smoke tests."
    )
    parser.add_argument("--output-dir", required=True)

    args = parser.parse_args()
    output_dir = create_smoke_model_artifacts(args.output_dir)
    print(json.dumps({"status": "ok", "output_dir": str(output_dir)}, indent=2))


if __name__ == "__main__":
    main()
