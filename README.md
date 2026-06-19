# CentaurDrug

CentaurDrug is an AI medicinal chemistry copilot for human-in-the-loop lead
optimization. The project combines ADMET prediction, rule-based medicinal
chemistry filters, RDKit candidate generation, LangGraph agent orchestration,
Gemini strategy guidance, a FastAPI backend, a plain HTML/CSS/JavaScript UI,
and an MLOps/DevOps pipeline.

The current version is a strong prototype for a PFA/MLOps demonstration. It is
not a clinical or production medicinal chemistry system. All ADMET predictions
are computational estimates and must be interpreted by a human expert.

## Project Report

### Problem

Early drug discovery needs more than a single prediction model. A useful
medicinal chemistry assistant should help answer questions such as:

- Is this molecule valid and drug-like?
- Does it respect classical oral drug-likeness rules?
- What are the predicted ADMET risks?
- Can I generate analogues that improve weak properties?
- Which generated candidates are better, and why?
- Can the system be packaged, tested, deployed, and monitored like an MLOps
  application?

CentaurDrug addresses this by building an end-to-end loop:

```text
SMILES input
    -> chemical validation
    -> rule filters
    -> ADMET model panel
    -> score vector and interpretation
    -> LLM-guided optimization strategy
    -> RDKit candidate generation
    -> candidate evaluation and ranking
    -> search tree and explanation
    -> API/UI output
```

### Main Objective

The objective is to build a demonstrable AI drug optimization backend where a
user can submit a molecule, evaluate its ADMET profile, generate candidate
analogues, inspect the optimization tree, and discuss the result with a
context-aware chat assistant.

### Current Status

| Part | Status | Notes |
| --- | --- | --- |
| ADMET model pipeline | Good prototype | XGBoost models trained per dataset with RDKit-based features. |
| Rule evaluator | Good prototype | Lipinski, Veber, PAINS, Brenk, QED, and SMILES validation. |
| Optimization agent | Good prototype | LangGraph loop with LLM strategy selection, RDKit transformations, scoring, and search tree. |
| Candidate explanations | Implemented | Score vector, delta vs parent, improvements, tradeoffs, scaffold preservation, synthetic accessibility proxy. |
| Backend API | Implemented | FastAPI endpoints for health, readiness, rules, evaluation, optimization, chat, and metrics. |
| Frontend UI | Implemented | Static HTML/CSS/JavaScript served directly by FastAPI. |
| MLOps/DevOps | Implemented prototype | DVC training stages, MLflow verification, Docker, Compose, GitHub Actions, GHCR, Kubernetes API manifest. |
| Scientific reliability | Limited | Predictions need more validation, more datasets, uncertainty, and expert review. |
| Production readiness | Not yet | Needs model registry/promotion, monitoring, security hardening, resource limits, and model governance. |

## Features

- Train five ADMET XGBoost models:
  - Solubility AqSolDB
  - Lipophilicity AstraZeneca
  - AMES mutagenicity
  - hERG cardiotoxicity risk
  - CYP3A4 inhibition
- Validate molecules with RDKit.
- Apply medicinal chemistry filters:
  - Lipinski
  - Veber
  - PAINS
  - Brenk
  - QED threshold
- Evaluate a full ADMET panel from mounted model artifacts.
- Compute interpretable score vectors for ranking and explanation.
- Generate analogues with RDKit transformations.
- Use Gemini through LangChain to select optimization strategies when an API
  key is available.
- Fall back to deterministic strategy/chat behavior when LLM credentials are
  not configured.
- Run a LangGraph optimization loop with a search tree.
- Return candidate explanations:
  - scalar score
  - score vector
  - delta vs parent
  - improvements
  - tradeoffs
  - scaffold preservation
  - synthetic accessibility proxy
  - constraints result
- Serve a static browser UI with FastAPI.
- Expose health, readiness, version, and metrics endpoints.
- Validate model artifacts at startup and through `/ready`.
- Package the app in a Docker API image.
- Run local integration with Docker Compose.
- Validate CI with tests, linting, API smoke test, and container smoke test.
- Deploy the API to Kubernetes with liveness and readiness probes.

## Architecture

```text
User browser
    -> FastAPI /
        -> serves src/ui/static/index.html
        -> serves src/ui/static/styles.css
        -> serves src/ui/static/app.js
        -> exposes REST endpoints
        -> loads ADMET models from mounted volume
        -> calls evaluator and LangGraph agent
        -> optionally calls Gemini for strategy/chat

Training and MLOps layer
    -> TDC datasets
    -> RDKit features
    -> XGBoost training
    -> MLflow tracking
    -> DVC stages
    -> model artifact folders

DevOps layer
    -> pytest and ruff
    -> Docker API image
    -> container smoke tests
    -> GHCR publishing
    -> Kubernetes deployment
```

## Repository Structure

```text
configs/
    training.yaml              Training, feature, split, and dataset config
dags/
    centaurdrug_training_pipeline.py
                               Airflow prototype DAG
k8s/
    api-deployment.yaml        Kubernetes API deployment, service, PVC, probes
src/api/
    main.py                    FastAPI app and REST endpoints
src/agent/
    graph.py                   LangGraph optimization workflow
    llm_agent.py               Gemini strategy agent and fallback behavior
    transformations.py         RDKit candidate generation
    scoring.py                 Score vectors, explanations, diversity
    search_tree.py             Optimization tree data structure
    constraints.py             Candidate constraints
src/models/
    train_admet_xgboost.py     ADMET model training pipeline
    predict.py                 Inference from model artifacts
    featurizers.py             Molecular features
    splitting.py               Scaffold/data splitting
    plots.py                   Training diagnostic plots
src/mlops/
    verify_mlflow.py           MLflow smoke check
    smoke_artifacts.py         Tiny model artifacts for CI container tests
src/tools/
    evaluator.py               Rule evaluation and ADMET panel evaluator
    filters.py                 Medicinal chemistry filters
    descriptors.py             Molecular descriptors
src/ui/static/
    index.html                 Static frontend
    styles.css                 UI styling
    app.js                     Frontend interactions and API calls
tests/
    test_*.py                  Unit and integration tests
Dockerfile.api                 API image definition
docker-compose.yml             Local API + MLflow services
dvc.yaml                       DVC training stages
Makefile                       Local commands
```

## Technology Stack

| Tool | Role in the project |
| --- | --- |
| Python | Main programming language. |
| FastAPI | Backend API and static UI serving. |
| HTML/CSS/JavaScript | Browser UI. |
| RDKit | SMILES parsing, descriptors, filters, fingerprints, scaffolds, transformations. |
| XGBoost | ADMET prediction models. |
| scikit-learn | Metrics, model utilities, and ML support. |
| TDC / PyTDC | Source of ADMET benchmark datasets. |
| LangGraph | Agent workflow orchestration. |
| LangChain Google GenAI | Gemini integration for strategy and chat. |
| MLflow | Experiment tracking and artifact logging. |
| DVC | Reproducible training stages and model output tracking. |
| Docker | Runtime packaging. |
| Docker Compose | Local multi-service execution. |
| Kubernetes | API deployment, service, PVC, liveness and readiness probes. |
| GitHub Actions | CI/CD automation. |
| GHCR | Container image registry. |
| pytest | Automated tests. |
| Ruff | Linting and code quality. |
| uv | Python dependency installation and command runner. |

## Data And Model Pipeline

The training configuration is in `configs/training.yaml`.

The five current datasets are:

| Dataset | Task | Primary metric | Output meaning |
| --- | --- | --- | --- |
| `Solubility_AqSolDB` | Regression | RMSE | Solubility-like continuous value. |
| `Lipophilicity_AstraZeneca` | Regression | RMSE | Lipophilicity-like continuous value. |
| `AMES` | Classification | ROC AUC | Mutagenicity/toxicity risk. |
| `hERG` | Classification | ROC AUC | Cardiotoxicity risk. |
| `CYP3A4_Veith` | Classification | ROC AUC | CYP3A4 inhibition risk. |

Training uses:

- RDKit molecular validation.
- Morgan fingerprints.
- MACCS keys.
- RDKit descriptors.
- Scaffold splitting:
  - train: 70%
  - early stop: 10%
  - validation: 10%
  - test: 10%
- Optional hyperparameter search with Optuna.
- Applicability-domain support using Tanimoto similarity to training
  fingerprints.

Each training run writes artifacts under:

```bash
models/admet_xgboost/<dataset>/
```

Important files produced per dataset:

- `model.joblib`
- `featurizer.joblib`
- `metadata.json`
- `metrics.json`
- `dataset_summary.json`
- `predictions.csv`
- `split_report.csv`
- `rejected_molecules.csv`
- `training_fps.joblib`
- `training_smiles.joblib`
- `plots/`

When `training.use_mlflow` is true, the training pipeline also logs metrics,
parameters, and artifacts to MLflow.

## Evaluation Layer

The evaluator lives in `src/tools/evaluator.py`.

It performs:

1. SMILES validation and canonicalization.
2. Rule evaluation:
   - Lipinski
   - Veber
   - PAINS
   - Brenk
   - QED
3. ADMET prediction with the five model artifacts.
4. ADMET interpretation:
   - poor solubility
   - high lipophilicity
   - AMES mutagenicity risk
   - hERG cardiotoxicity risk
   - CYP3A4 inhibition risk
   - out-of-applicability-domain risk
5. Final decision:
   - pass
   - review
   - reject or needs optimization depending on risks and rules

The evaluator is intentionally conservative. It does not claim that a molecule
is clinically safe. It only gives computational estimates and flags.

## Agent And Optimization Layer

The agent is implemented in `src/agent/graph.py` with LangGraph.

The workflow is:

```text
initialize
    -> evaluate parent molecule
    -> create root node in optimization tree

choose_strategy
    -> read current best frontier node
    -> ask Gemini for transformation strategy if configured
    -> use deterministic fallback if Gemini is unavailable

expand_frontier
    -> generate candidates with RDKit transformations
    -> apply constraints
    -> evaluate each candidate
    -> compute score vector
    -> add valid candidates to the tree

select_next_frontier
    -> rank candidates
    -> keep top beam-width frontier nodes

finalize
    -> select best and diverse candidates
    -> build explanations
    -> return tree summary and candidate summaries
```

The score vector is built in `src/agent/scoring.py`.

Current normalized score dimensions:

- solubility
- lipophilicity
- AMES safety
- hERG safety
- CYP3A4 safety
- QED
- Lipinski
- Veber
- PAINS
- Brenk
- applicability domain
- scaffold preservation
- synthetic accessibility proxy

The scalar score is a weighted combination of the vector. The vector is still
returned because the scalar alone is not scientifically explainable.

Candidate explanations include:

- `delta_vs_parent`
- `parent_comparison`
- `improvements`
- `tradeoffs`
- `scaffold`
- `synthetic_accessibility`
- `constraints`
- optional `delta_vs_root` when the candidate is deeper than one step

This makes the optimizer easier to defend than a black-box "candidate score".

## Backend API

The backend is implemented in `src/api/main.py`.

Run it locally:

```bash
make api
```

Then open:

```text
http://localhost:8000
```

Main endpoints:

| Method | Endpoint | Purpose |
| --- | --- | --- |
| GET | `/` | Serves the static web UI. |
| GET | `/health` | Checks that the API process responds. |
| GET | `/ready` | Checks that all five ADMET model folders are present and complete. |
| GET | `/version` | Returns API version and default model root. |
| GET | `/metrics` | Returns simple Prometheus-style request counters. |
| POST | `/rules` | Runs rule filters only. |
| POST | `/evaluate` | Runs full ADMET panel evaluation. |
| POST | `/optimize` | Runs the LangGraph optimization agent. |
| POST | `/chat` | Chat assistant for the current molecule or candidate. |

Example rule call:

```bash
curl --fail --silent \
  --header "Content-Type: application/json" \
  --data '{"smiles":"CCO"}' \
  http://127.0.0.1:8000/rules
```

Example evaluation call:

```bash
curl --fail --silent \
  --header "Content-Type: application/json" \
  --data '{"smiles":"CCO"}' \
  http://127.0.0.1:8000/evaluate
```

Example optimization call:

```bash
curl --fail --silent \
  --header "Content-Type: application/json" \
  --data '{"smiles":"CC(=O)Oc1ccccc1C(=O)O","max_depth":2,"beam_width":3}' \
  http://127.0.0.1:8000/optimize
```

## Frontend UI

The current UI is plain HTML/CSS/JavaScript in:

```text
src/ui/static/
```

FastAPI serves the frontend directly. There is no active Streamlit application,
no `Dockerfile.ui`, and no separate UI Kubernetes deployment.

This decision keeps the prototype simple:

- one app container;
- one API deployment;
- one port;
- same-origin API calls from the browser;
- easier CI/CD smoke testing.

If the frontend grows later, the recommended next step is a real frontend build
system such as Vite, not Streamlit.

## Chat Assistant

The `/chat` endpoint is context-aware. The UI can send the current molecule,
selected candidate, evaluation result, and conversation history.

If `GEMINI_API_KEY` is configured, the chat uses Gemini through LangChain. If no
key is configured, the endpoint returns deterministic fallback guidance so the
demo still works without external LLM access.

The chat assistant is designed to explain:

- why a candidate is better or worse;
- which ADMET properties improved;
- which tradeoffs remain;
- whether a molecule is outside the applicability domain;
- what to inspect next.

## Model Artifacts And Runtime Contract

The Docker image does not contain trained models. Model artifacts are delivered
at runtime by mounting a folder or volume.

The API expects:

```bash
/app/models/admet_xgboost
```

Inside this root, the required folders are:

- `Solubility_AqSolDB`
- `Lipophilicity_AstraZeneca`
- `AMES`
- `hERG`
- `CYP3A4_Veith`

Each folder must contain:

- `model.joblib`
- `featurizer.joblib`
- `metadata.json`

The readiness checker validates these files. When
`CENTAURDRUG_REQUIRE_MODELS_ON_STARTUP=1`, the API fails startup if the model
panel is incomplete.

Important environment variables:

| Variable | Meaning |
| --- | --- |
| `CENTAURDRUG_MODEL_ROOT` | Model root used by the API. |
| `CENTAURDRUG_ALLOWED_MODEL_ROOTS` | Allowed roots for model loading. Prevents arbitrary path access. |
| `CENTAURDRUG_REQUIRE_MODELS_ON_STARTUP` | Enables startup failure when artifacts are incomplete. |
| `CENTAURDRUG_CORS_ORIGINS` | Allowed browser origins. |
| `CENTAURDRUG_HOST_MODEL_ROOT` | Host folder mounted into Compose as `/app/models/admet_xgboost`. |
| `GEMINI_API_KEY` | Optional key for Gemini chat and strategy guidance. |
| `GEMINI_REQUEST_TIMEOUT` | Optional Gemini request timeout. |

## Model Delivery

CentaurDrug now supports three model-delivery paths:

1. DVC remote storage for trained model artifacts.
2. MLflow Model Registry entries for versioned bundle handoff.
3. Versioned mounted artifact bundles for Docker, Compose, and Kubernetes.

The local default DVC remote is:

```text
../centaurdrug-dvc-remote
```

From the repository root, this is a sibling folder next to `centaurdrug`.
DVC stores it as `../../centaurdrug-dvc-remote` in `.dvc/config` because DVC
remote paths are written relative to `.dvc/config`.

Create or refresh it, then push DVC-tracked model outputs:

```bash
make dvc-remote
make dvc-push
```

For cloud storage, replace `DVC_REMOTE_URL` with an S3, GCS, Azure, SSH, or
other DVC-supported remote:

```bash
make dvc-remote DVC_REMOTE_NAME=s3-model-store DVC_REMOTE_URL=s3://my-bucket/centaurdrug
```

Create a versioned mounted bundle from the current ADMET panel:

```bash
make bundle-models
```

This writes:

```text
model_bundles/centaurdrug-admet-panel/<version>/
    manifest.json
    admet_xgboost/
        _bundle_manifest.json
        Solubility_AqSolDB/
        Lipophilicity_AstraZeneca/
        AMES/
        hERG/
        CYP3A4_Veith/
```

The manifest contains bundle version, Git metadata, model metadata, metrics,
file sizes, and SHA-256 checksums. `/ready` reports the bundle metadata when
`_bundle_manifest.json` is present in the mounted model root.

Verify a bundle:

```bash
make verify-model-bundle MODEL_BUNDLE_DIR=model_bundles/centaurdrug-admet-panel/<version>
```

Run Compose with a versioned bundle:

```bash
CENTAURDRUG_HOST_MODEL_ROOT=./model_bundles/centaurdrug-admet-panel/<version>/admet_xgboost \
docker compose up --build
```

Register a verified bundle in MLflow:

```bash
export MLFLOW_TRACKING_URI=sqlite:///mlflow.db
make register-model-bundle \
  MODEL_BUNDLE_DIR=model_bundles/centaurdrug-admet-panel/<version> \
  MLFLOW_MODEL_ALIAS=staging
```

The registry entry points to the logged bundle artifact. This is a registry
handoff for the full ADMET panel, not a single MLflow pyfunc model.

## Local Development

Install dependencies:

```bash
uv sync --locked --group dev --group training
```

Install every optional workflow group:

```bash
uv sync --locked --all-groups
```

Run tests:

```bash
make test
```

Run the API and UI:

```bash
make api
```

Train all five phase-1 ADMET models:

```bash
make train-phase1
```

Run the evaluator from CLI:

```bash
make evaluate
```

Run a sample agent search:

```bash
make agent-search
```

Verify MLflow:

```bash
make verify-mlflow
```

Open MLflow locally:

```bash
uv run mlflow ui --host 0.0.0.0 --port 5000
```

Then visit:

```text
http://localhost:5000
```

## Docker And Docker Compose

Build the API image:

```bash
docker build -f Dockerfile.api -t centaurdrug-api .
```

Run with Docker Compose:

```bash
docker compose up --build
```

Compose starts:

- API and UI on `http://localhost:8000`
- MLflow on `http://localhost:5000`

The API service mounts:

```text
${CENTAURDRUG_HOST_MODEL_ROOT:-./models/admet_xgboost}:/app/models/admet_xgboost:ro
```

This mirrors the production idea: the image contains code and dependencies, and
the model panel is provided by an external volume.

## MLOps

### DVC

`dvc.yaml` defines one training stage per ADMET dataset. Each stage declares:

- training command;
- source dependencies;
- configuration dependency;
- dependency lockfile;
- model artifacts;
- reports;
- plots;
- metrics.

This makes model training reproducible and traceable.

The tracked default remote is `local-model-store`, pointing to
`../centaurdrug-dvc-remote`. Keep credentials and cloud-specific overrides out
of Git with DVC local config when moving beyond the local file remote.

### MLflow

MLflow is used for experiment tracking. The helper
`src/mlops/verify_mlflow.py` verifies that the project can:

- create a run;
- log parameters;
- log metrics;
- log a text artifact;
- log a plot artifact;
- read the run back.

`src/mlops/model_delivery.py` can also log a verified model bundle and create a
Model Registry version for `centaurdrug-admet-panel`.

### Smoke Model Artifacts

`src/mlops/smoke_artifacts.py` creates tiny fake ADMET artifacts used by CI.
These artifacts are not scientific models. They exist only to prove that the
API image can load a complete model panel and execute `/evaluate` inside a real
container.

This was important because a recent CI failure came from joblib pickling fake
models created through `python -m`. The smoke classes are now registered under
the importable module name so the container can load them reliably.

## CI/CD

The GitHub Actions workflow is in:

```text
.github/workflows/ci.yml
```

It runs on:

- push;
- pull request;
- version tags like `v1.0.0`;
- manual workflow dispatch.

Pipeline jobs:

1. `Python checks`
   - checkout;
   - install uv;
   - install project Python;
   - `uv sync --locked --group dev --group training`;
   - `ruff check src tests`;
   - `pytest`;
   - start FastAPI and call `/health`.

2. `API container smoke test`
   - generate smoke model artifacts;
   - build `Dockerfile.api`;
   - run the built image;
   - mount smoke models at `/app/models/admet_xgboost`;
   - call `/ready`;
   - call `/health`;
   - call `/rules`;
   - call `/evaluate`.

3. `Container image`
   - build the API image with Docker Buildx;
   - publish to GitHub Container Registry on main, tags, or manual runs;
   - image name pattern: `ghcr.io/<owner>/<repo>-api`.

4. `Deploy to Kubernetes`
   - applies `k8s/api-deployment.yaml`;
   - sets the deployment image to the exact `sha-<commit>` tag;
   - waits for rollout status;
   - skips clearly if `KUBE_CONFIG` is not configured.

This pipeline validates not only Python code, but also the real packaged API
runtime.

## Kubernetes

The core API Kubernetes manifest is:

```text
k8s/api-deployment.yaml
```

It defines:

- `centaurdrug-models-pvc` PersistentVolumeClaim;
- `centaurdrug-api` Deployment;
- API container on port 8000;
- model volume mounted read-only;
- `/ready` readiness probe;
- `/health` liveness probe;
- `centaurdrug-api-service` NodePort service.

The readiness probe is especially important in an ML application. A pod can be
alive but not useful if the model files are missing. `/ready` prevents traffic
from going to pods that do not have the full ADMET model panel.

The optional MLOps manifests are:

```text
k8s/mlflow-deployment.yaml
k8s/airflow-deployment.yaml
```

`k8s/mlflow-deployment.yaml` defines a PVC-backed MLflow tracking server on
port 5000 with SQLite metadata and local artifact storage under `/mlflow`.
Inside the cluster, clients can use:

```bash
MLFLOW_TRACKING_URI=http://centaurdrug-mlflow-service:5000
```

`k8s/airflow-deployment.yaml` defines a small Airflow 3 deployment with:

- Postgres metadata database;
- Airflow API server on port 8080;
- scheduler, dag processor, and triggerer containers;
- PVCs for DAGs and logs;
- startup database migration and default admin bootstrap;
- health probes for Airflow API and scheduler components.

The Airflow manifest is a working platform baseline, not yet the final training
orchestration image. Before exposing it outside a private cluster, replace the
sample Secret values. Before running the project DAGs, build an Airflow image or
worker path that includes the CentaurDrug source, `uv`, DVC, and the training
dependency groups.

## Testing

The test suite currently covers:

- API endpoints;
- frontend serving from FastAPI;
- rule filters;
- model artifact validation;
- smoke artifact loading;
- agent constraints;
- scoring;
- transformations;
- search tree behavior;
- training split and validation helpers;
- plotting helpers.

Run all tests:

```bash
uv run pytest -q
```

Run lint:

```bash
uv run ruff check src tests
```

Current local verification after the latest CI fix:

```text
50 passed
ruff: All checks passed
container smoke: /ready, /health, /rules, /evaluate passed
```

## Security And Reliability Notes

Implemented:

- `.env` is ignored by Git.
- Docker build context excludes caches, models, local data, docs, and secrets.
- API rejects unapproved model roots.
- CORS defaults to local FastAPI origins.
- Kubernetes deployment uses readiness and liveness probes.
- CI uses a real container smoke test.

Still needed:

- run the Docker image as a non-root user;
- add Kubernetes CPU and memory requests/limits;
- use Kubernetes Secrets for `GEMINI_API_KEY`;
- add vulnerability scanning;
- add structured logging;
- add Prometheus/Grafana monitoring;
- add model drift and prediction-quality monitoring.

## Scientific Limitations

CentaurDrug is a prototype. Important limitations:

- ADMET predictions are only as reliable as the datasets and features used.
- The synthetic accessibility score is a proxy, not a full retrosynthesis
  analysis.
- The scaffold preservation score is a similarity heuristic.
- The generated candidates are not guaranteed to be synthesizable.
- No docking, binding affinity, pharmacophore, or 3D conformer analysis is
  currently included.
- No uncertainty calibration or conformal prediction is currently enabled.
- The LLM guides strategy but must not be treated as a chemistry authority.
- Human medicinal chemistry review is required.

## Roadmap

Recommended next steps:

1. Add stricter model promotion rules: experimental, staging, production.
2. Add Kubernetes resource requests and limits.
3. Add Kubernetes Secrets for Gemini and other credentials.
4. Add Ingress and TLS for a real deployment URL.
5. Deploy MLflow properly in Kubernetes.
6. Replace the prototype Airflow DAG with a real five-model training
    orchestration.
7. Add Prometheus/Grafana dashboards.
8. Add uncertainty estimation and more ADMET datasets such as CYP2D6, CYP2C9,
    LD50, and DILI.
9. Add stronger synthetic feasibility checks.
10. Add optional docking or target-specific scoring after the ADMET and API
    layers are stable.

## Defense Summary

The project demonstrates an end-to-end AI and MLOps drug optimization prototype.
It starts from a SMILES string, evaluates chemical rules and ADMET predictions,
uses an LLM-guided LangGraph search loop to propose analogues, scores and
explains candidates, exposes the workflow through FastAPI and a static web UI,
and validates deployment through Docker, CI/CD, and Kubernetes readiness checks.

The strongest engineering idea is the separation between code and model
artifacts: the API image contains the application, while trained models are
mounted as runtime artifacts and validated through `/ready`. This makes the
project much closer to a real MLOps system than a simple notebook demo.

The strongest scientific idea is explainability: the system does not return
only a single score. It returns score vectors, parent deltas, improvements,
tradeoffs, scaffold preservation, synthetic accessibility proxy, and a search
tree, making the result easier to inspect and defend.

The honest conclusion is that CentaurDrug is demo-ready and architecturally
strong for a prototype, but it still needs more model validation, monitoring,
security hardening, deployment monitoring, and model governance before it can be
called production-ready.
