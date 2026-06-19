APP_HOST ?= 0.0.0.0
APP_PORT ?= 8000
MODEL_DELIVERY_CONFIG ?= configs/model_delivery.yaml
DVC_REMOTE_NAME ?= local-model-store
DVC_REMOTE_URL ?= ../centaurdrug-dvc-remote
MLFLOW_HOST ?= 0.0.0.0
MLFLOW_PORT ?= 5000
MLFLOW_BACKEND_STORE_URI ?= sqlite:///mlflow.db
MLFLOW_ARTIFACT_ROOT ?= ./mlruns
MLFLOW_ALLOWED_HOSTS ?= localhost,127.0.0.1,0.0.0.0
MLFLOW_TRACKING_URI ?= http://127.0.0.1:$(MLFLOW_PORT)
MLFLOW_MODEL_ALIAS ?= staging
AIRFLOW_HOME ?= .airflow
AIRFLOW_USERNAME ?= admin
AIRFLOW_PASSWORD ?= centaurdrug
AIRFLOW_ROLE ?= admin
AIRFLOW_MLFLOW_TRACKING_URI ?= sqlite:///mlflow.db
AIRFLOW_CONTAINER ?= api-server
KUBECTL ?= kubectl
K8S_NAMESPACE ?= default
K8S_MLFLOW_LOCAL_PORT ?= 5000
K8S_AIRFLOW_LOCAL_PORT ?= 8080
K8S = $(KUBECTL) -n $(K8S_NAMESPACE)

.PHONY: \
	test sync sync-all app run api \
	mlflow-local mlflow-local-verify airflow-local \
	k8s-namespace k8s-mlflow-up k8s-mlflow-port-forward k8s-mlflow-logs \
	k8s-airflow-up k8s-airflow-sync-dags k8s-airflow-port-forward k8s-airflow-logs \
	k8s-mlops-up k8s-mlops-down \
	verify-mlflow train-sol train-lipo train-ames train-herg train-cyp3a4 \
	train-phase1 predict-sol dvc-remote dvc-push bundle-models \
	verify-model-bundle register-model-bundle evaluate agent-search

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

mlflow-local:
	@mkdir -p "$(MLFLOW_ARTIFACT_ROOT)"
	@echo "MLflow local server: http://127.0.0.1:$(MLFLOW_PORT)"
	uv run --group mlops mlflow server \
		--host $(MLFLOW_HOST) \
		--port $(MLFLOW_PORT) \
		--backend-store-uri $(MLFLOW_BACKEND_STORE_URI) \
		--artifacts-destination file://$(abspath $(MLFLOW_ARTIFACT_ROOT)) \
		--allowed-hosts "$(MLFLOW_ALLOWED_HOSTS)"

mlflow-local-verify:
	MLFLOW_TRACKING_URI=$(MLFLOW_TRACKING_URI) \
		uv run --group mlops python -m src.mlops.verify_mlflow --config configs/training.yaml

airflow-local:
	@mkdir -p "$(AIRFLOW_HOME)"
	@echo "Airflow standalone: http://127.0.0.1:8080"
	@printf '{"$(AIRFLOW_USERNAME)":"$(AIRFLOW_PASSWORD)"}\n' > "$(AIRFLOW_HOME)/simple_auth_manager_passwords.json.generated"
	@chmod 600 "$(AIRFLOW_HOME)/simple_auth_manager_passwords.json.generated"
	@echo "Airflow local credentials: $(AIRFLOW_USERNAME) / $(AIRFLOW_PASSWORD)"
	AIRFLOW_HOME="$(abspath $(AIRFLOW_HOME))" \
	AIRFLOW__CORE__SIMPLE_AUTH_MANAGER_USERS="$(AIRFLOW_USERNAME):$(AIRFLOW_ROLE)" \
	AIRFLOW__CORE__SIMPLE_AUTH_MANAGER_PASSWORDS_FILE="$(abspath $(AIRFLOW_HOME))/simple_auth_manager_passwords.json.generated" \
	AIRFLOW__CORE__DAGS_FOLDER="$(CURDIR)/dags" \
	AIRFLOW__CORE__DAGS_ARE_PAUSED_AT_CREATION="True" \
	AIRFLOW__CORE__LOAD_EXAMPLES="False" \
	AIRFLOW__CORE__EXECUTION_API_SERVER_URL="http://127.0.0.1:8080/execution/" \
	AIRFLOW__SCHEDULER__ENABLE_HEALTH_CHECK="True" \
	CENTAURDRUG_AIRFLOW_MLFLOW_TRACKING_URI="$(AIRFLOW_MLFLOW_TRACKING_URI)" \
	PYTHONPATH="$(CURDIR)" \
	uv run --group orchestration --group mlops --group training airflow standalone

k8s-namespace:
	@if [ "$(K8S_NAMESPACE)" != "default" ]; then \
		$(KUBECTL) get namespace "$(K8S_NAMESPACE)" >/dev/null 2>&1 || \
		$(KUBECTL) create namespace "$(K8S_NAMESPACE)"; \
	fi

k8s-mlflow-up: k8s-namespace
	$(K8S) apply -f k8s/mlflow-deployment.yaml
	$(K8S) rollout status deployment/centaurdrug-mlflow --timeout=180s
	@echo "MLflow is inside the cluster at http://centaurdrug-mlflow-service:5000"
	@echo "Open locally with: make k8s-mlflow-port-forward"

k8s-mlflow-port-forward:
	@echo "MLflow local URL: http://127.0.0.1:$(K8S_MLFLOW_LOCAL_PORT)"
	$(K8S) port-forward service/centaurdrug-mlflow-service $(K8S_MLFLOW_LOCAL_PORT):5000

k8s-mlflow-logs:
	$(K8S) logs -f deployment/centaurdrug-mlflow -c mlflow

k8s-airflow-up: k8s-namespace
	$(K8S) apply -f k8s/airflow-deployment.yaml
	$(K8S) rollout status deployment/centaurdrug-airflow-postgres --timeout=180s
	$(K8S) rollout status deployment/centaurdrug-airflow --timeout=300s
	@echo "Airflow is inside the cluster at http://centaurdrug-airflow-service:8080"
	@echo "Open locally with: make k8s-airflow-port-forward"
	@echo "Sync repo DAGs with: make k8s-airflow-sync-dags"

k8s-airflow-sync-dags:
	@pod="$$( \
		$(K8S) get pod \
			-l app.kubernetes.io/name=centaurdrug,app.kubernetes.io/component=airflow \
			-o jsonpath='{.items[0].metadata.name}' \
	)"; \
	test -n "$$pod" || (echo "Airflow pod not found. Run make k8s-airflow-up first."; exit 1); \
	$(K8S) cp dags/. "$$pod":/opt/airflow/dags -c api-server

k8s-airflow-port-forward:
	@echo "Airflow local URL: http://127.0.0.1:$(K8S_AIRFLOW_LOCAL_PORT)"
	$(K8S) port-forward service/centaurdrug-airflow-service $(K8S_AIRFLOW_LOCAL_PORT):8080

k8s-airflow-logs:
	$(K8S) logs -f deployment/centaurdrug-airflow -c $(AIRFLOW_CONTAINER)

k8s-mlops-up: k8s-mlflow-up k8s-airflow-up
	@echo "MLOps services are applied. Use the port-forward targets to open them locally."

k8s-mlops-down:
	-$(K8S) delete -f k8s/airflow-deployment.yaml --ignore-not-found
	-$(K8S) delete -f k8s/mlflow-deployment.yaml --ignore-not-found

verify-mlflow:
	uv run --group mlops python -m src.mlops.verify_mlflow --config configs/training.yaml

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
