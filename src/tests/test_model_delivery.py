from pathlib import Path

from src.mlops.model_delivery import (
    RUNTIME_MANIFEST,
    create_bundle,
    verify_bundle,
)
from src.mlops.smoke_artifacts import create_smoke_model_artifacts
from src.tools.evaluator import validate_model_artifacts


def test_create_and_verify_model_bundle(tmp_path):
    model_root = create_smoke_model_artifacts(tmp_path / "models")
    config = {
        "model_panel": {
            "name": "centaurdrug-test-panel",
            "model_root": str(model_root),
        },
        "bundle": {
            "output_root": str(tmp_path / "bundles"),
            "runtime_model_subdir": "admet_xgboost",
        },
    }

    result = create_bundle(
        config=config,
        version="v-test",
    )

    bundle_dir = Path(result["bundle_dir"])
    runtime_model_root = bundle_dir / "admet_xgboost"

    assert result["status"] == "ok"
    assert (bundle_dir / "manifest.json").exists()
    assert (runtime_model_root / RUNTIME_MANIFEST).exists()

    verification = verify_bundle(bundle_dir)

    assert verification["status"] == "ok"
    assert verification["bundle_version"] == "v-test"

    readiness = validate_model_artifacts(runtime_model_root)

    assert readiness["status"] == "ok"
    assert readiness["bundle"]["bundle_version"] == "v-test"


def test_verify_bundle_reports_checksum_mismatch(tmp_path):
    model_root = create_smoke_model_artifacts(tmp_path / "models")
    config = {
        "model_panel": {
            "name": "centaurdrug-test-panel",
            "model_root": str(model_root),
        },
        "bundle": {
            "output_root": str(tmp_path / "bundles"),
            "runtime_model_subdir": "admet_xgboost",
        },
    }

    result = create_bundle(config=config, version="v-test")
    bundle_dir = Path(result["bundle_dir"])
    metadata_path = bundle_dir / "admet_xgboost" / "AMES" / "metadata.json"
    metadata_path.write_text("{}", encoding="utf-8")

    verification = verify_bundle(bundle_dir)

    assert verification["status"] == "error"
    assert any("mismatch" in error for error in verification["errors"])
