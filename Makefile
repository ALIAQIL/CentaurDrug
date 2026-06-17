APP_HOST ?= 0.0.0.0
APP_PORT ?= 8000
MODEL_DELIVERY_CONFIG ?= configs/model_delivery.yaml
DVC_REMOTE_NAME ?= local-model-store
DVC_REMOTE_URL ?= ../centaurdrug-dvc-remote
MLFLOW_MODEL_ALIAS ?= staging

.PHONY: test sync sync-all app run api verify-mlflow train-sol train-lipo train-ames train-herg train-cyp3a4 train-phase1 predict-sol dvc-remote dvc-push bundle-models verify-model-bundle register-model-bundle

test:
	uv run pytest -q

sync:
	uv sync --locked --group dev --group training

sync-all:
	uv sync --locked --all-groups

app:
	uv run uvicorn src.api.main:app --reload --host $(APP_HOST) --port $(APP_PORT)

run: app

api: app

verify-mlflow:
	uv run python -m src.mlops.verify_mlflow --config configs/training.yaml

dvc-remote:
	@case "$(DVC_REMOTE_URL)" in *://*) ;; *) mkdir -p "$(DVC_REMOTE_URL)" ;; esac
	uv run --group mlops dvc remote add --force -d $(DVC_REMOTE_NAME) $(DVC_REMOTE_URL)
	uv run --group mlops dvc remote list

dvc-push:
	uv run --group mlops dvc push

bundle-models:
	uv run --group mlops python -m src.mlops.model_delivery --config $(MODEL_DELIVERY_CONFIG) bundle

verify-model-bundle:
	@test -n "$(MODEL_BUNDLE_DIR)" || (echo "Set MODEL_BUNDLE_DIR=/path/to/bundle"; exit 1)
	uv run --group mlops python -m src.mlops.model_delivery --config $(MODEL_DELIVERY_CONFIG) verify --bundle-dir $(MODEL_BUNDLE_DIR)

register-model-bundle:
	@test -n "$(MODEL_BUNDLE_DIR)" || (echo "Set MODEL_BUNDLE_DIR=/path/to/bundle"; exit 1)
	uv run --group mlops python -m src.mlops.model_delivery --config $(MODEL_DELIVERY_CONFIG) register --bundle-dir $(MODEL_BUNDLE_DIR) --alias $(MLFLOW_MODEL_ALIAS)

train-sol:
	uv run python -m src.models.train_admet_xgboost --config configs/training.yaml --dataset Solubility_AqSolDB

train-lipo:
	uv run python -m src.models.train_admet_xgboost --config configs/training.yaml --dataset Lipophilicity_AstraZeneca

train-ames:
	uv run python -m src.models.train_admet_xgboost --config configs/training.yaml --dataset AMES

train-herg:
	uv run python -m src.models.train_admet_xgboost --config configs/training.yaml --dataset hERG

train-cyp3a4:
	uv run python -m src.models.train_admet_xgboost --config configs/training.yaml --dataset CYP3A4_Veith

train-phase1: train-sol train-lipo train-ames train-herg train-cyp3a4

predict-sol:
	uv run python -m src.models.predict --artifact-dir models/admet_xgboost/Solubility_AqSolDB --smiles "CCO"

evaluate:
	uv run python -m src.tools.evaluator --smiles "CCO"

agent-search:
	uv run python -m src.agent.graph --smiles "CC(=O)Oc1ccccc1C(=O)O" --max-depth 2 --beam-width 3 --max-candidates-per-node 10
