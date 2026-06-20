from __future__ import annotations

import os
from datetime import datetime, timedelta, timezone
from pathlib import Path

from airflow import DAG
from airflow.providers.standard.operators.bash import BashOperator


PROJECT_ROOT = Path(
    os.environ.get(
        "CENTAURDRUG_PROJECT_ROOT",
        Path(__file__).resolve().parents[1],
    )
).resolve()

COMMON_ENV = {
    "PYTHONPATH": str(PROJECT_ROOT),
    "MLFLOW_TRACKING_URI": os.environ.get(
        "CENTAURDRUG_AIRFLOW_MLFLOW_TRACKING_URI",
        "sqlite:///mlflow.db",
    ),
    "CENTAURDRUG_MLFLOW_MODEL_ALIAS": os.environ.get(
        "CENTAURDRUG_MLFLOW_MODEL_ALIAS",
        "staging",
    ),
}

DVC_STAGES = (
    ("train_solubility", "train_solubility_aqsoldb"),
    ("train_lipophilicity", "train_lipophilicity_astrazeneca"),
    ("train_ames", "train_ames"),
    ("train_herg", "train_herg"),
    ("train_cyp3a4", "train_cyp3a4_veith"),
)


def project_bash_task(
    task_id: str,
    bash_command: str,
    *,
    execution_timeout: timedelta = timedelta(hours=6),
) -> BashOperator:
    return BashOperator(
        task_id=task_id,
        bash_command=f"set -euo pipefail\n{bash_command}",
        cwd=str(PROJECT_ROOT),
        env=COMMON_ENV,
        append_env=True,
        do_xcom_push=False,
        execution_timeout=execution_timeout,
    )


with DAG(
    dag_id="centaurdrug_training_pipeline",
    description="Train, verify, bundle, and register the full ADMET model panel.",
    start_date=datetime(2025, 1, 1, tzinfo=timezone.utc),
    schedule="@weekly",
    catchup=False,
    max_active_runs=1,
    default_args={
        "owner": "centaurdrug",
        "retries": 2,
        "retry_delay": timedelta(minutes=10),
    },
    tags=["centaurdrug", "training", "dvc", "mlflow"],
) as dag:
    check_pipeline = project_bash_task(
        task_id="check_pipeline",
        bash_command=(
            "test -f pyproject.toml\n"
            "test -f uv.lock\n"
            "test -f dvc.yaml\n"
            "uv lock --check\n"
            "uv run --frozen --group mlops dvc status"
        ),
        execution_timeout=timedelta(minutes=10),
    )

    previous_task = check_pipeline
    for task_id, stage_name in DVC_STAGES:
        train_task = project_bash_task(
            task_id=task_id,
            bash_command=(
                "uv run --frozen --group mlops --group training "
                f"dvc repro {stage_name}"
            ),
        )
        previous_task >> train_task
        previous_task = train_task

    verify_model_panel = project_bash_task(
        task_id="verify_model_panel",
        bash_command=(
            "uv run --frozen python - <<'PY'\n"
            "from src.tools.evaluator import assert_model_artifacts_ready\n"
            "assert_model_artifacts_ready('models/admet_xgboost')\n"
            "print('Verified the complete five-model ADMET panel.')\n"
            "PY"
        ),
        execution_timeout=timedelta(minutes=10),
    )

    publish_model_bundle = project_bash_task(
        task_id="publish_model_bundle",
        bash_command=(
            "bundle_json=$(mktemp)\n"
            "trap 'rm -f \"$bundle_json\"' EXIT\n"
            "uv run --frozen --group mlops python -m src.mlops.model_delivery "
            "--config configs/model_delivery.yaml bundle > \"$bundle_json\"\n"
            "bundle_dir=$(uv run --frozen python -c "
            "'import json,sys; print(json.load(open(sys.argv[1]))[\"bundle_dir\"])' "
            "\"$bundle_json\")\n"
            "cat \"$bundle_json\"\n"
            "uv run --frozen --group mlops python -m src.mlops.model_delivery "
            "--config configs/model_delivery.yaml register "
            "--bundle-dir \"$bundle_dir\" "
            "--alias \"$CENTAURDRUG_MLFLOW_MODEL_ALIAS\""
        ),
        execution_timeout=timedelta(hours=1),
    )

    previous_task >> verify_model_panel >> publish_model_bundle
