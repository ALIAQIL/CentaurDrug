from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, Optional

import joblib
from rdkit import DataStructs

from src.models.featurizers import morgan_fingerprint_for_ad
from src.models.validation import validate_smiles


class ADMETPredictor:
    """
    Load one trained ADMET artifact directory and serve predictions.
    """

    def __init__(
        self,
        artifact_dir: str | Path,
    ):
        self.artifact_dir = Path(artifact_dir)

        self.model = joblib.load(self.artifact_dir / "model.joblib")
        self.featurizer = joblib.load(
            self.artifact_dir / "featurizer.joblib"
        )

        self.metadata = json.loads(
            (self.artifact_dir / "metadata.json").read_text(
                encoding="utf-8"
            )
        )

        fps_path = self.artifact_dir / "training_fps.joblib"

        if fps_path.exists():
            self.training_fps = joblib.load(fps_path)
        else:
            self.training_fps = None

        self.task_type = self.metadata["task_type"]
        self.dataset = self.metadata["dataset"]

        self.prediction_unit = self.metadata.get("prediction_unit")
        self.positive_class_name = self.metadata.get(
            "positive_class_name"
        )

        features = self.metadata.get("features", {})

        self.radius = int(features.get("morgan_radius", 2))
        self.n_bits = int(features.get("morgan_n_bits", 2048))

    def applicability_domain_score(
        self,
        canonical_smiles: str,
    ) -> Optional[float]:
        """
        Compute max Tanimoto similarity to training molecules.

        Higher = closer to training chemistry.
        """

        if not self.training_fps:
            return None

        query_fp = morgan_fingerprint_for_ad(
            canonical_smiles,
            radius=self.radius,
            n_bits=self.n_bits,
        )

        similarities = DataStructs.BulkTanimotoSimilarity(
            query_fp,
            self.training_fps,
        )

        if not similarities:
            return None

        return float(max(similarities))

    def predict(
        self,
        smiles: Any,
        ad_threshold: float = 0.35,
    ) -> Dict[str, Any]:
        """
        Production-style inference contract.
        """

        validation = validate_smiles(smiles)

        if not validation.is_valid:
            return {
                "status": "rejected",
                "reason": validation.rejection_reason,
                "original_smiles": validation.original_smiles,
                "dataset": self.dataset,
            }

        X = self.featurizer.transform([validation.canonical_smiles])

        ad_score = self.applicability_domain_score(
            validation.canonical_smiles
        )

        if ad_score is None:
            in_domain = None
        else:
            in_domain = bool(ad_score >= ad_threshold)

        if self.task_type == "regression":
            prediction = float(self.model.predict(X)[0])

            return {
                "status": "ok",
                "dataset": self.dataset,
                "task_type": self.task_type,
                "original_smiles": validation.original_smiles,
                "canonical_smiles": validation.canonical_smiles,
                "prediction": prediction,
                "prediction_unit": self.prediction_unit,
                "ad_score": ad_score,
                "in_applicability_domain": in_domain,
                "ad_threshold": ad_threshold,
            }

        if self.task_type == "classification":
            probability_positive = float(
                self.model.predict_proba(X)[0, 1]
            )

            prediction = int(probability_positive >= 0.5)

            return {
                "status": "ok",
                "dataset": self.dataset,
                "task_type": self.task_type,
                "positive_class_name": self.positive_class_name,
                "original_smiles": validation.original_smiles,
                "canonical_smiles": validation.canonical_smiles,
                "probability_positive": probability_positive,
                "prediction": prediction,
                "threshold": 0.5,
                "ad_score": ad_score,
                "in_applicability_domain": in_domain,
                "ad_threshold": ad_threshold,
            }

        raise ValueError(
            f"Unsupported task_type in metadata: {self.task_type}"
        )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run ADMET prediction from artifact directory."
    )

    parser.add_argument(
        "--artifact-dir",
        required=True,
    )

    parser.add_argument(
        "--smiles",
        required=True,
    )

    parser.add_argument(
        "--ad-threshold",
        type=float,
        default=0.35,
    )

    args = parser.parse_args()

    predictor = ADMETPredictor(args.artifact_dir)

    result = predictor.predict(
        smiles=args.smiles,
        ad_threshold=args.ad_threshold,
    )

    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()