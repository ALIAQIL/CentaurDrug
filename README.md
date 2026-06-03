# CentaurDrug

CentaurDrug trains XGBoost ADMET models on TDC datasets and logs model
artifacts, metrics, dataset exploration summaries, and diagnostic plots.

## MLflow

Verify the local MLflow tracking setup:

```bash
make verify-mlflow
```

The verifier creates a smoke-check run, logs a metric plus text/PNG artifacts,
then reads the run back through the MLflow client.

Open the local MLflow UI:

```bash
uv run mlflow ui --host 0.0.0.0 --port 5000
```

Then visit `http://localhost:5000`.

## Training Artifacts

Each training run writes artifacts under `models/admet_xgboost/<dataset>/`.
Alongside the model bundle, the trainer now saves:

- `dataset_summary.json`: raw/valid/rejected counts, target summaries, split
  sizes, scaffold counts, and class balance for classification tasks.
- `predictions.csv`: split-level predictions for regression or classification.
- `plots/dataset_split_overview.png`: molecule and scaffold counts by split.
- `plots/target_distribution_by_split.png`: target or class balance by split.
- `plots/metrics_by_split.png`: model metrics across train, early-stop,
  validation, and test splits.
- `plots/regression_diagnostics.png`: predicted-vs-actual and residual plots.
- `plots/classification_curves.png`: ROC and precision-recall curves.
- `plots/classification_confusion_matrices.png`: confusion matrices by split.

When `training.use_mlflow` is true, these files are also logged to MLflow.
