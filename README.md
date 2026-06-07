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

## API And Web App

Run the FastAPI backend and built-in HTML/CSS/JS interface:

```bash
make api
```

Then visit `http://localhost:8000`.

The plain browser frontend lives in `src/ui/static/` and is served by FastAPI.
The older Streamlit prototype remains in `src/ui/app.py`; install the
`prototype` extra only if you want to run that legacy prototype locally.

Main endpoints:

- `GET /health`: service status.
- `POST /rules`: lightweight rule filters.
- `POST /evaluate`: full ADMET panel evaluation.
- `POST /optimize`: LangGraph agent optimization with candidate explanations.
- `POST /chat`: context-aware copilot chat for the current molecule/candidate.

## CI/CD

The GitHub Actions pipeline lives in `.github/workflows/ci.yml`.

What it does and why:

- `Python checks`: runs on every push, pull request, and manual workflow run.
  It checks out the repo, installs `uv`, syncs runtime plus dev dependencies,
  runs `ruff`, runs `pytest`, then starts FastAPI and calls `/health`. This
  catches code quality, test, and application startup problems before images
  are published.
- `Container image`: builds both `Dockerfile.api` and `Dockerfile.ui`. Pull
  requests build images without publishing, while pushes to `main`, version
  tags like `v1.0.0`, and manual runs publish images to GitHub Container
  Registry:
  `ghcr.io/<owner>/<repo>-api` and `ghcr.io/<owner>/<repo>-ui`.
- `Deploy to Kubernetes`: runs after image publishing on `main`, or manually
  when the workflow input `deploy` is enabled. It applies the Kubernetes
  manifests, updates the API and UI deployments to the exact `sha-<commit>`
  image tag that passed CI, then waits for both rollouts.

To enable Kubernetes deployment, add a repository or environment secret named
`KUBE_CONFIG` containing the kubeconfig for the target cluster. If that secret
is not configured, the workflow still publishes images and clearly skips the
cluster deployment step.

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
