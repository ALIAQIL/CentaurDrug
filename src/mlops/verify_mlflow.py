from __future__ import annotations

import argparse
import json
import tempfile
from pathlib import Path
from typing import Any, Dict

import matplotlib
import yaml

matplotlib.use("Agg")

import matplotlib.pyplot as plt

try:
    import mlflow
    from mlflow.tracking import MlflowClient
except Exception as exc:  # pragma: no cover
    mlflow = None
    MlflowClient = None
    MLFLOW_IMPORT_ERROR = exc
else:
    MLFLOW_IMPORT_ERROR = None


def load_experiment_name(
    config_path: str | Path,
) -> str:
    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    return config.get("training", {}).get(
        "mlflow_experiment",
        "centaurdrug-admet-xgboost",
    )


def _write_smoke_plot(
    path: Path,
) -> None:
    fig, ax = plt.subplots(figsize=(4, 3))
    ax.plot([0, 1, 2], [0.0, 0.8, 1.0], marker="o")
    ax.set_title("MLflow smoke plot")
    ax.set_xlabel("step")
    ax.set_ylabel("metric")
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def verify_mlflow(
    experiment_name: str,
) -> Dict[str, Any]:
    if mlflow is None or MlflowClient is None:
        raise RuntimeError(
            "MLflow could not be imported. "
            f"Original error: {MLFLOW_IMPORT_ERROR}"
        )

    tracking_uri = mlflow.get_tracking_uri()
    mlflow.set_experiment(experiment_name)

    with tempfile.TemporaryDirectory() as tmpdir:
        tmp_path = Path(tmpdir)
        text_artifact = tmp_path / "mlflow_smoke_artifact.txt"
        plot_artifact = tmp_path / "mlflow_smoke_plot.png"

        text_artifact.write_text(
            "centaurdrug mlflow smoke check\n",
            encoding="utf-8",
        )
        _write_smoke_plot(plot_artifact)

        with mlflow.start_run(run_name="mlflow-smoke-check") as run:
            run_id = run.info.run_id
            experiment_id = run.info.experiment_id

            mlflow.log_param("check", "mlflow_smoke")
            mlflow.log_metric("smoke_metric", 1.0)
            mlflow.log_artifact(str(text_artifact))
            mlflow.log_artifact(str(plot_artifact))

    client = MlflowClient()
    run = client.get_run(run_id)
    artifacts = client.list_artifacts(run_id)
    artifact_names = {artifact.path for artifact in artifacts}

    if run.data.params.get("check") != "mlflow_smoke":
        raise RuntimeError("MLflow smoke check parameter was not recorded.")

    if run.data.metrics.get("smoke_metric") != 1.0:
        raise RuntimeError("MLflow smoke check metric was not recorded.")

    expected_artifacts = {
        "mlflow_smoke_artifact.txt",
        "mlflow_smoke_plot.png",
    }

    missing_artifacts = sorted(expected_artifacts.difference(artifact_names))

    if missing_artifacts:
        raise RuntimeError(
            "MLflow smoke check artifacts were not recorded: "
            f"{missing_artifacts}"
        )

    with tempfile.TemporaryDirectory() as download_dir:
        downloaded_path = Path(
            client.download_artifacts(
                run_id,
                "mlflow_smoke_artifact.txt",
                download_dir,
            )
        )
        if downloaded_path.read_text(encoding="utf-8") != (
            "centaurdrug mlflow smoke check\n"
        ):
            raise RuntimeError(
                "MLflow smoke check artifact could not be downloaded correctly."
            )

    registry_name = f"centaurdrug-smoke-{run_id}"
    registry_alias = "smoke"
    registry_created = False
    alias_set = False
    model_version = None

    try:
        client.create_registered_model(registry_name)
        registry_created = True
        model_version = client.create_model_version(
            name=registry_name,
            source=run.info.artifact_uri,
            run_id=run_id,
        )
        client.set_registered_model_alias(
            registry_name,
            registry_alias,
            model_version.version,
        )
        alias_set = True
        resolved_version = client.get_model_version_by_alias(
            registry_name,
            registry_alias,
        )

        if resolved_version.run_id != run_id:
            raise RuntimeError(
                "MLflow registry alias did not resolve to the smoke run."
            )
    finally:
        if alias_set:
            client.delete_registered_model_alias(
                registry_name,
                registry_alias,
            )
        if model_version is not None:
            client.delete_model_version(
                registry_name,
                model_version.version,
            )
        if registry_created:
            client.delete_registered_model(registry_name)

    return {
        "status": "ok",
        "tracking_uri": tracking_uri,
        "experiment_name": experiment_name,
        "experiment_id": experiment_id,
        "run_id": run_id,
        "logged_artifacts": sorted(artifact_names),
        "logged_metrics": dict(run.data.metrics),
        "registry_check": {
            "status": "ok",
            "alias": registry_alias,
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Verify that CentaurDrug can write and read MLflow runs."
    )
    parser.add_argument(
        "--config",
        default="configs/training.yaml",
    )
    parser.add_argument(
        "--experiment",
        default=None,
    )

    args = parser.parse_args()
    experiment_name = args.experiment or load_experiment_name(args.config)
    result = verify_mlflow(experiment_name)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
