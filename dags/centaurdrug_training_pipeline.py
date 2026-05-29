from airflow import DAG
from airflow.operators.bash import BashOperator
from datetime import datetime


with DAG(
    dag_id="centaurdrug_training_pipeline",
    start_date=datetime(2025, 1, 1),
    schedule="@weekly",
    catchup=False,
) as dag:

    validate_data = BashOperator(
        task_id="validate_data",
        bash_command="uv run python -c 'import pandas as pd; from src.mlops.validation import validate_dataset; df=pd.read_csv(\"data/processed/herg.csv\"); validate_dataset(df); print(\"valid\")'",
    )

    train_model = BashOperator(
        task_id="train_herg_model",
        bash_command="uv run python src/models/train_baseline.py",
    )

    dvc_repro = BashOperator(
        task_id="dvc_repro",
        bash_command="uv run dvc repro",
    )

    validate_data >> train_model >> dvc_repro