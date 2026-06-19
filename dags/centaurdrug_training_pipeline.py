from __future__ import annotations

import os
from datetime import datetime
from pathlib import Path

from airflow import DAG
from airflow.providers.standard.operators.bash import BashOperator


PROJECT_ROOT = Path(__file__).resolve().parents[1]
COMMON_ENV = {
    "PYTHONPATH": str(PROJECT_ROOT),
    "MLFLOW_TRACKING_URI": os.environ.get(
        "CENTAURDRUG_AIRFLOW_MLFLOW_TRACKING_URI",
        "sqlite:///mlflow.db",
    ),
}


def project_bash_task(task_id: str, bash_command: str) -> BashOperator:
    return BashOperator(
        task_id=task_id,
        bash_command=f"set -euo pipefail\n{bash_command}",
        cwd=str(PROJECT_ROOT),
        env=COMMON_ENV,
        append_env=True,
    )


with DAG(
    dag_id="centaurdrug_training_pipeline",
    description="Train and verify the CentaurDrug hERG ADMET model stage.",
    start_date=datetime(2025, 1, 1),
    schedule="@weekly",
    catchup=False,
    max_active_runs=1,
    tags=["centaurdrug", "training", "dvc", "mlflow"],
) as dag:
    check_pipeline = project_bash_task(
        task_id="check_pipeline",
        bash_command=(
            "uv lock --check\n"
            "uv run --group mlops dvc status --quiet || true"
        ),
    )

    train_herg_model = project_bash_task(
        task_id="train_herg_model",
        bash_command=(
            "uv run --group mlops --group training "
            "dvc repro train_herg"
        ),
    )

    verify_herg_artifacts = project_bash_task(
        task_id="verify_herg_artifacts",
        bash_command=(
            "test -f models/admet_xgboost/hERG/model.joblib\n"
            "test -f models/admet_xgboost/hERG/featurizer.joblib\n"
            "test -f models/admet_xgboost/hERG/metadata.json\n"
            "uv run python - <<'PY'\n"
            "from pathlib import Path\n"
            "import json\n"
            "artifact_dir = Path('models/admet_xgboost/hERG')\n"
            "metadata = json.loads((artifact_dir / 'metadata.json').read_text())\n"
            "metrics = metadata.get('metrics', {})\n"
            "print('Verified hERG artifact directory:', artifact_dir)\n"
            "print('Primary metric:', metadata.get('primary_metric'))\n"
            "print('Available metric splits:', sorted(metrics))\n"
            "PY"
        ),
    )

    check_pipeline >> train_herg_model >> verify_herg_artifacts
