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

The browser frontend lives in `src/ui/static/` and is served directly by
FastAPI. There is no Streamlit app and no separate UI container in the current
architecture.

Main endpoints:

- `GET /health`: service status.
- `GET /ready`: readiness check that verifies all five ADMET model artifact
  folders are present and complete.
- `POST /rules`: lightweight rule filters.
- `POST /evaluate`: full ADMET panel evaluation.
- `POST /optimize`: LangGraph agent optimization with candidate explanations.
- `POST /chat`: context-aware copilot chat for the current molecule/candidate.

## Model Artifacts And Containers

Container images do not bake trained ADMET models into the image. The Docker
and Kubernetes runtime contract is to mount model artifacts at:

```bash
/app/models/admet_xgboost
```

The API expects the five dataset folders under that root:

- `Solubility_AqSolDB`
- `Lipophilicity_AstraZeneca`
- `AMES`
- `hERG`
- `CYP3A4_Veith`

Each folder must contain at least `model.joblib`, `featurizer.joblib`, and
`metadata.json`. When `CENTAURDRUG_REQUIRE_MODELS_ON_STARTUP=1`, startup fails
if the mounted model panel is incomplete. `docker-compose.yml` mounts the local
`./models/admet_xgboost` folder read-only into the container.

## UI Architecture Decision Report

### Decision

CentaurDrug uses a single application service:

```text
FastAPI backend + static HTML/CSS/JavaScript frontend
```

The static frontend is served by the FastAPI application at `/`, with assets
under `/static`. The frontend calls the same FastAPI service for `/rules`,
`/evaluate`, `/optimize`, `/chat`, `/health`, `/ready`, and `/metrics`.

### Why This Architecture

This is the best fit for the current project because:

- the real UI is already implemented in plain HTML, CSS, and JavaScript;
- FastAPI already serves the static UI and backend endpoints in one process;
- deployment is simpler: one API image, one Kubernetes deployment, one service;
- CI/CD is easier to validate because one container contains the complete app;
- there is no need to maintain a second Streamlit prototype path.

### What Was Removed

The old Streamlit prototype was removed from the active codebase. The project no
longer has:

- `src/ui/app.py`;
- `Dockerfile.ui`;
- `k8s/ui-deployment.yaml`;
- Streamlit optional dependencies;
- a separate `ui` service in `docker-compose.yml`;
- a separate `centaurdrug-ui` image in CI/CD.

### Current Runtime Shape

```text
User browser
    -> FastAPI /
        -> serves src/ui/static/index.html
        -> serves src/ui/static/styles.css
        -> serves src/ui/static/app.js
        -> exposes API endpoints
        -> loads mounted ADMET model artifacts
```

### Future Direction

If the frontend becomes larger later, the next step should be a dedicated
frontend build system such as Vite, not Streamlit. For now, the current
HTML/CSS/JavaScript UI is simple, fast, and aligned with the PFA demo.

## CI/CD

The GitHub Actions pipeline lives in `.github/workflows/ci.yml`.

What it does and why:

- `Python checks`: runs on every push, pull request, and manual workflow run.
  It checks out the repo, installs `uv`, syncs runtime plus dev dependencies,
  runs `ruff`, runs `pytest`, then starts FastAPI and calls `/health`. This
  catches code quality, test, and application startup problems before images
  are published.
- `Container image`: builds `Dockerfile.api`. Pull requests build the image
  without publishing, while pushes to `main`, version tags like `v1.0.0`, and
  manual runs publish the API image to GitHub Container Registry:
  `ghcr.io/<owner>/<repo>-api`.
- `API container smoke test`: builds the API image locally, generates tiny
  smoke model artifacts, mounts them into the container, then calls `/ready`,
  `/health`, `/rules`, and `/evaluate`.
- `Deploy to Kubernetes`: runs after image publishing on `main`, or manually
  when the workflow input `deploy` is enabled. It applies the Kubernetes
  manifest, updates the API deployment to the exact `sha-<commit>` image tag
  that passed CI, then waits for the rollout.

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
