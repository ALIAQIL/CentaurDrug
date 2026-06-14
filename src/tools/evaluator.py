from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict

from src.models.predict import ADMETPredictor
from src.models.validation import validate_smiles
from src.tools.filters import (
    brenk_filter,
    is_valid_smiles,
    lipinski_filter,
    pains_filter,
    qed_score,
    veber_filter,
)


DEFAULT_MODEL_ROOT = Path("models/admet_xgboost")


MODEL_PATHS = {
    "solubility": "Solubility_AqSolDB",
    "lipophilicity": "Lipophilicity_AstraZeneca",
    "ames": "AMES",
    "herg": "hERG",
    "cyp3a4": "CYP3A4_Veith",
}

REQUIRED_ARTIFACT_FILES = (
    "model.joblib",
    "featurizer.joblib",
    "metadata.json",
)


class ModelArtifactError(RuntimeError):
    pass


def validate_model_artifacts(model_root: str | Path = DEFAULT_MODEL_ROOT) -> Dict[str, Any]:
    root = Path(model_root)
    missing: Dict[str, list[str]] = {}

    for short_name, dataset_dir in MODEL_PATHS.items():
        artifact_dir = root / dataset_dir
        missing_files = [
            filename
            for filename in REQUIRED_ARTIFACT_FILES
            if not (artifact_dir / filename).exists()
        ]

        if missing_files:
            missing[short_name] = [
                str(artifact_dir / filename)
                for filename in missing_files
            ]

    return {
        "status": "ok" if not missing else "error",
        "model_root": str(root),
        "required_models": sorted(MODEL_PATHS.keys()),
        "missing": missing,
    }


def assert_model_artifacts_ready(model_root: str | Path = DEFAULT_MODEL_ROOT) -> None:
    result = validate_model_artifacts(model_root)

    if result["status"] != "ok":
        raise ModelArtifactError(
            "ADMET model artifacts are missing or incomplete: "
            f"{result['missing']}"
        )


def evaluate_rules(smiles: str) -> dict:
    if not is_valid_smiles(smiles):
        return {
            "valid": False,
            "decision": "reject",
            "reason": "Invalid SMILES",
        }

    lipinski = lipinski_filter(smiles)
    veber = veber_filter(smiles)
    pains = pains_filter(smiles)
    brenk = brenk_filter(smiles)
    qed = qed_score(smiles)

    passed = (
        lipinski["passed"]
        and veber["passed"]
        and pains["passed"]
        and brenk["passed"]
        and qed >= 0.35
    )

    return {
        "valid": True,
        "lipinski": lipinski,
        "veber": veber,
        "pains": pains,
        "brenk": brenk,
        "qed": qed,
        "qed_threshold": 0.35,
        "decision": "pass" if passed else "reject",
    }


class ADMETPanelEvaluator:
    def __init__(
        self,
        model_root: str | Path = DEFAULT_MODEL_ROOT,
        require_all_models: bool = True,
    ):
        self.model_root = Path(model_root)
        self.predictors: Dict[str, ADMETPredictor] = {}

        if require_all_models:
            assert_model_artifacts_ready(self.model_root)

        for short_name, dataset_dir in MODEL_PATHS.items():
            artifact_dir = self.model_root / dataset_dir

            if artifact_dir.exists():
                self.predictors[short_name] = ADMETPredictor(artifact_dir)

        if require_all_models and set(self.predictors) != set(MODEL_PATHS):
            missing = sorted(set(MODEL_PATHS).difference(self.predictors))
            raise ModelArtifactError(
                f"Could not load required ADMET predictors: {missing}"
            )

    def predict_admet(self, smiles: str) -> Dict[str, Any]:
        predictions = {}

        for name, predictor in self.predictors.items():
            try:
                predictions[name] = predictor.predict(smiles)
            except Exception as exc:
                predictions[name] = {
                    "status": "error",
                    "error": str(exc),
                }

        return predictions

    def interpret_admet(self, admet_predictions: Dict[str, Any]) -> Dict[str, Any]:
        risks = []
        guidance = []

        solubility = admet_predictions.get("solubility", {})
        if solubility.get("status") == "ok":
            value = solubility.get("prediction")
            if value is not None and value < -4.0:
                risks.append("poor_solubility")
                guidance.append(
                    "Improve aqueous solubility while preserving the core scaffold."
                )

        lipophilicity = admet_predictions.get("lipophilicity", {})
        if lipophilicity.get("status") == "ok":
            value = lipophilicity.get("prediction")
            if value is not None and value > 4.0:
                risks.append("high_lipophilicity")
                guidance.append(
                    "Reduce excessive lipophilicity to improve developability."
                )

        ames = admet_predictions.get("ames", {})
        if ames.get("status") == "ok":
            probability = ames.get("probability_positive")
            if probability is not None and probability >= 0.5:
                risks.append("ames_mutagenicity_risk")
                guidance.append(
                    "Avoid structural alerts associated with mutagenicity."
                )

        herg = admet_predictions.get("herg", {})
        if herg.get("status") == "ok":
            probability = herg.get("probability_positive")
            if probability is not None and probability >= 0.5:
                risks.append("herg_cardiotoxicity_risk")
                guidance.append(
                    "Reduce hERG liability, for example by lowering lipophilicity or avoiding strongly basic exposed amines."
                )

        cyp3a4 = admet_predictions.get("cyp3a4", {})
        if cyp3a4.get("status") == "ok":
            probability = cyp3a4.get("probability_positive")
            if probability is not None and probability >= 0.5:
                risks.append("cyp3a4_inhibition_risk")
                guidance.append(
                    "Reduce CYP3A4 inhibition liability to lower drug-drug interaction risk."
                )

        out_of_domain_models = []

        for model_name, prediction in admet_predictions.items():
            if (
                prediction.get("status") == "ok"
                and prediction.get("in_applicability_domain") is False
            ):
                out_of_domain_models.append(model_name)

        if out_of_domain_models:
            risks.append("out_of_applicability_domain")
            guidance.append(
                f"Predictions for {out_of_domain_models} are outside the applicability domain and should not be over-trusted."
            )

        if not risks:
            status = "acceptable"

        elif any(
            risk in risks
            for risk in [
                "ames_mutagenicity_risk",
                "herg_cardiotoxicity_risk",
            ]
        ):
            status = "needs_optimization_high_priority"

        else:
            status = "needs_optimization"

        return {
            "status": status,
            "main_risks": risks,
            "llm_guidance": guidance,
        }

    def evaluate_molecule(self, smiles: str) -> Dict[str, Any]:
        validation = validate_smiles(smiles)

        if not validation.is_valid:
            return {
                "status": "rejected",
                "reason": validation.rejection_reason,
                "original_smiles": validation.original_smiles,
            }

        canonical_smiles = validation.canonical_smiles

        rules = evaluate_rules(canonical_smiles)
        admet_predictions = self.predict_admet(canonical_smiles)
        interpretation = self.interpret_admet(admet_predictions)

        final_decision = self.make_final_decision(
            rules=rules,
            interpretation=interpretation,
        )

        return {
            "status": "ok",
            "original_smiles": validation.original_smiles,
            "canonical_smiles": canonical_smiles,
            "rules": rules,
            "admet_predictions": admet_predictions,
            "overall_assessment": interpretation,
            "final_decision": final_decision,
        }

    @staticmethod
    def make_final_decision(
        rules: Dict[str, Any],
        interpretation: Dict[str, Any],
    ) -> Dict[str, Any]:
        if not rules.get("valid", False):
            return {
                "decision": "reject",
                "reason": "invalid_smiles",
            }

        if rules.get("decision") == "reject":
            return {
                "decision": "needs_optimization",
                "reason": "rule_filter_or_qed_issue",
            }

        risks = interpretation.get("main_risks", [])
        status = interpretation.get("status")

        high_priority_risks = {
            "ames_mutagenicity_risk",
            "herg_cardiotoxicity_risk",
        }

        non_domain_risks = [
            risk for risk in risks
            if risk != "out_of_applicability_domain"
        ]

        has_high_priority_risk = any(
            risk in high_priority_risks
            for risk in risks
        )

        only_out_of_domain = (
            risks == ["out_of_applicability_domain"]
        )

        if has_high_priority_risk:
            return {
                "decision": "needs_optimization",
                "reason": "high_priority_admet_risk",
            }

        if only_out_of_domain:
            return {
                "decision": "uncertain",
                "reason": "out_of_applicability_domain",
            }

        if status == "acceptable" or not risks:
            return {
                "decision": "pass",
                "reason": "rules_and_admet_acceptable",
            }

        if non_domain_risks:
            return {
                "decision": "needs_optimization",
                "reason": "admet_risk",
            }

        return {
            "decision": "uncertain",
            "reason": "uncertain_admet_assessment",
        }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate a molecule using CentaurDrug rules and ADMET models."
    )

    parser.add_argument(
        "--smiles",
        required=True,
    )

    parser.add_argument(
        "--model-root",
        default=str(DEFAULT_MODEL_ROOT),
    )

    args = parser.parse_args()

    evaluator = ADMETPanelEvaluator(model_root=args.model_root)

    result = evaluator.evaluate_molecule(args.smiles)

    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
