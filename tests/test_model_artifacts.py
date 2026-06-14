import pytest

from src.mlops.smoke_artifacts import create_smoke_model_artifacts
from src.tools.evaluator import (
    MODEL_PATHS,
    ADMETPanelEvaluator,
    assert_model_artifacts_ready,
    validate_model_artifacts,
)


def test_validate_model_artifacts_reports_missing_files(tmp_path):
    result = validate_model_artifacts(tmp_path)

    assert result["status"] == "error"
    assert set(result["missing"]) == set(MODEL_PATHS)


def test_smoke_artifacts_are_loadable(tmp_path):
    root = create_smoke_model_artifacts(tmp_path / "models")

    assert_model_artifacts_ready(root)

    evaluator = ADMETPanelEvaluator(model_root=root)
    result = evaluator.evaluate_molecule("CCO")

    assert result["status"] == "ok"
    assert set(result["admet_predictions"]) == set(MODEL_PATHS)
    assert result["admet_predictions"]["solubility"]["prediction"] == -2.0
    assert result["admet_predictions"]["ames"]["probability_positive"] == pytest.approx(
        0.1
    )
