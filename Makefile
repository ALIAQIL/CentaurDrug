APP_HOST ?= 0.0.0.0
APP_PORT ?= 8000
MODEL_DELIVERY_CONFIG ?= configs/model_delivery.yaml
DVC_REMOTE_NAME ?= local-model-store
DVC_REMOTE_URL ?= ../centaurdrug-dvc-remote
MLFLOW_HOST ?= 0.0.0.0
MLFLOW_PORT ?= 5000
MLFLOW_BACKEND_STORE_URI ?= sqlite:///mlflow.db
MLFLOW_ARTIFACT_ROOT ?= ./mlruns
MLFLOW_ALLOWED_HOSTS ?= localhost,localhost:5000,127.0.0.1,127.0.0.1:5000,0.0.0.0,centaurdrug-mlflow-service,centaurdrug-mlflow-service:5000
MLFLOW_CORS_ALLOWED_ORIGINS ?= http://localhost:*,http://127.0.0.1:*
MLFLOW_TRACKING_URI ?= http://127.0.0.1:$(MLFLOW_PORT)
MLFLOW_TRACKING_USERNAME ?=
MLFLOW_TRACKING_PASSWORD ?=
MLFLOW_MODEL_ALIAS ?= staging
MLFLOW_IMAGE ?= ghcr.io/aliaqil/centaurdrug-mlflow:latest
MLFLOW_DATABASE_URI ?=
MLFLOW_ARTIFACTS_DESTINATION ?=
MLFLOW_ADMIN_USERNAME ?=
MLFLOW_ADMIN_PASSWORD ?=
MLFLOW_FLASK_SERVER_SECRET_KEY ?=
MLFLOW_POSTGRES_PASSWORD ?=
MLFLOW_LOCAL_DATABASE_URI = postgresql+psycopg2://mlflow:$(MLFLOW_POSTGRES_PASSWORD)@centaurdrug-mlflow-postgres:5432/mlflow
MLFLOW_LOCAL_ARTIFACTS_DESTINATION ?= file:///mlflow/artifacts
AIRFLOW_HOME ?= .airflow
AIRFLOW_USERNAME ?= admin
AIRFLOW_PASSWORD ?= centaurdrug
AIRFLOW_ROLE ?= admin
AIRFLOW_MLFLOW_TRACKING_URI ?= sqlite:///mlflow.db
AIRFLOW_IMAGE ?= ghcr.io/aliaqil/centaurdrug-airflow:latest
AIRFLOW_ADMIN_USERNAME ?= admin
AIRFLOW_DATABASE_URI ?=
AIRFLOW_ADMIN_PASSWORD ?=
AIRFLOW_FERNET_KEY ?=
AIRFLOW_JWT_SECRET ?=
AIRFLOW_POSTGRES_PASSWORD ?=
AIRFLOW_COMPONENT ?= scheduler
KUBECTL ?= kubectl
K8S_NAMESPACE ?= centaurdrug
K8S_MLFLOW_LOCAL_PORT ?= 5000
K8S_AIRFLOW_LOCAL_PORT ?= 8080
K8S = $(KUBECTL) -n $(K8S_NAMESPACE)

.PHONY: \
		test sync sync-all app run api \
			mlflow-local mlflow-local-verify mlflow-compose-up mlflow-compose-down \
			mlflow-compose-logs airflow-local mlflow-image \
			k8s-namespace k8s-mlflow-secret k8s-mlflow-local-secret \
			k8s-mlflow-db-up k8s-mlflow-up k8s-mlflow-local-up \
			k8s-mlflow-port-forward k8s-mlflow-logs \
		airflow-image k8s-airflow-secret k8s-airflow-db-up k8s-airflow-up \
		k8s-airflow-local-up \
		k8s-airflow-port-forward k8s-airflow-logs \
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
		MLFLOW_TRACKING_USERNAME="$(MLFLOW_TRACKING_USERNAME)" \
		MLFLOW_TRACKING_PASSWORD="$(MLFLOW_TRACKING_PASSWORD)" \
			uv run --group mlops python -m src.mlops.verify_mlflow --config configs/training.yaml

mlflow-compose-up:
	@test -n "$(MLFLOW_ADMIN_USERNAME)" || (echo "Set MLFLOW_ADMIN_USERNAME."; exit 1)
	@test -n "$(MLFLOW_ADMIN_PASSWORD)" || (echo "Set MLFLOW_ADMIN_PASSWORD."; exit 1)
	@test -n "$(MLFLOW_FLASK_SERVER_SECRET_KEY)" || (echo "Set MLFLOW_FLASK_SERVER_SECRET_KEY."; exit 1)
	@test -n "$(MLFLOW_POSTGRES_PASSWORD)" || (echo "Set MLFLOW_POSTGRES_PASSWORD to a URL-safe value."; exit 1)
	MLFLOW_ADMIN_USERNAME="$(MLFLOW_ADMIN_USERNAME)" \
		MLFLOW_ADMIN_PASSWORD="$(MLFLOW_ADMIN_PASSWORD)" \
		MLFLOW_FLASK_SERVER_SECRET_KEY="$(MLFLOW_FLASK_SERVER_SECRET_KEY)" \
		MLFLOW_POSTGRES_PASSWORD="$(MLFLOW_POSTGRES_PASSWORD)" \
		docker compose up --build -d --wait --wait-timeout 300 mlflow
	@echo "MLflow: http://127.0.0.1:$(MLFLOW_PORT)"

mlflow-compose-down:
	MLFLOW_ADMIN_USERNAME=unused \
		MLFLOW_ADMIN_PASSWORD=unused \
		MLFLOW_FLASK_SERVER_SECRET_KEY=unused \
		MLFLOW_POSTGRES_PASSWORD=unused \
		docker compose down

mlflow-compose-logs:
	MLFLOW_ADMIN_USERNAME=unused \
		MLFLOW_ADMIN_PASSWORD=unused \
		MLFLOW_FLASK_SERVER_SECRET_KEY=unused \
		MLFLOW_POSTGRES_PASSWORD=unused \
		docker compose logs -f mlflow mlflow-migrate mlflow-postgres

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
	$(KUBECTL) apply -f k8s/namespace.yaml

mlflow-image:
	docker build --file Dockerfile.mlflow --tag $(MLFLOW_IMAGE) .

k8s-mlflow-secret: k8s-namespace
	@test -n "$(MLFLOW_DATABASE_URI)" || (echo "Set MLFLOW_DATABASE_URI."; exit 1)
	@test -n "$(MLFLOW_ARTIFACTS_DESTINATION)" || (echo "Set MLFLOW_ARTIFACTS_DESTINATION."; exit 1)
	@test -n "$(MLFLOW_ADMIN_USERNAME)" || (echo "Set MLFLOW_ADMIN_USERNAME."; exit 1)
	@test -n "$(MLFLOW_ADMIN_PASSWORD)" || (echo "Set MLFLOW_ADMIN_PASSWORD."; exit 1)
	@test -n "$(MLFLOW_FLASK_SERVER_SECRET_KEY)" || (echo "Set MLFLOW_FLASK_SERVER_SECRET_KEY."; exit 1)
	@$(K8S) create secret generic centaurdrug-mlflow-server-secret \
		--from-literal=MLFLOW_BACKEND_STORE_URI="$(MLFLOW_DATABASE_URI)" \
		--from-literal=MLFLOW_AUTH_DATABASE_URI="$(MLFLOW_DATABASE_URI)" \
		--from-literal=MLFLOW_ARTIFACTS_DESTINATION="$(MLFLOW_ARTIFACTS_DESTINATION)" \
		--from-literal=MLFLOW_ADMIN_USERNAME="$(MLFLOW_ADMIN_USERNAME)" \
		--from-literal=MLFLOW_ADMIN_PASSWORD="$(MLFLOW_ADMIN_PASSWORD)" \
		--from-literal=MLFLOW_FLASK_SERVER_SECRET_KEY="$(MLFLOW_FLASK_SERVER_SECRET_KEY)" \
		--from-literal=MLFLOW_ALLOWED_HOSTS="$(MLFLOW_ALLOWED_HOSTS)" \
		--from-literal=MLFLOW_CORS_ALLOWED_ORIGINS="$(MLFLOW_CORS_ALLOWED_ORIGINS)" \
		--dry-run=client -o yaml | $(K8S) apply -f -
	@$(K8S) create secret generic centaurdrug-mlflow-client-secret \
		--from-literal=MLFLOW_TRACKING_USERNAME="$(MLFLOW_ADMIN_USERNAME)" \
		--from-literal=MLFLOW_TRACKING_PASSWORD="$(MLFLOW_ADMIN_PASSWORD)" \
		--dry-run=client -o yaml | $(K8S) apply -f -

k8s-mlflow-local-secret: k8s-namespace
	@test -n "$(MLFLOW_ADMIN_USERNAME)" || (echo "Set MLFLOW_ADMIN_USERNAME."; exit 1)
	@test -n "$(MLFLOW_ADMIN_PASSWORD)" || (echo "Set MLFLOW_ADMIN_PASSWORD."; exit 1)
	@test -n "$(MLFLOW_FLASK_SERVER_SECRET_KEY)" || (echo "Set MLFLOW_FLASK_SERVER_SECRET_KEY."; exit 1)
	@test -n "$(MLFLOW_POSTGRES_PASSWORD)" || (echo "Set MLFLOW_POSTGRES_PASSWORD to a URL-safe value."; exit 1)
	@$(K8S) create secret generic centaurdrug-mlflow-server-secret \
		--from-literal=MLFLOW_BACKEND_STORE_URI="$(MLFLOW_LOCAL_DATABASE_URI)" \
		--from-literal=MLFLOW_AUTH_DATABASE_URI="$(MLFLOW_LOCAL_DATABASE_URI)" \
		--from-literal=MLFLOW_ARTIFACTS_DESTINATION="$(MLFLOW_LOCAL_ARTIFACTS_DESTINATION)" \
		--from-literal=MLFLOW_ADMIN_USERNAME="$(MLFLOW_ADMIN_USERNAME)" \
		--from-literal=MLFLOW_ADMIN_PASSWORD="$(MLFLOW_ADMIN_PASSWORD)" \
		--from-literal=MLFLOW_FLASK_SERVER_SECRET_KEY="$(MLFLOW_FLASK_SERVER_SECRET_KEY)" \
		--from-literal=MLFLOW_ALLOWED_HOSTS="$(MLFLOW_ALLOWED_HOSTS)" \
		--from-literal=MLFLOW_CORS_ALLOWED_ORIGINS="$(MLFLOW_CORS_ALLOWED_ORIGINS)" \
		--from-literal=MLFLOW_POSTGRES_PASSWORD="$(MLFLOW_POSTGRES_PASSWORD)" \
		--dry-run=client -o yaml | $(K8S) apply -f -
	@$(K8S) create secret generic centaurdrug-mlflow-client-secret \
		--from-literal=MLFLOW_TRACKING_USERNAME="$(MLFLOW_ADMIN_USERNAME)" \
		--from-literal=MLFLOW_TRACKING_PASSWORD="$(MLFLOW_ADMIN_PASSWORD)" \
		--dry-run=client -o yaml | $(K8S) apply -f -

k8s-mlflow-db-up: k8s-mlflow-local-secret
	$(K8S) apply -f k8s/mlflow-postgres.yaml
	$(K8S) rollout status statefulset/centaurdrug-mlflow-postgres --timeout=180s

k8s-mlflow-up: k8s-mlflow-secret
	-$(K8S) delete job centaurdrug-mlflow-migrate --ignore-not-found --wait=true
	@sed 's|ghcr.io/aliaqil/centaurdrug-mlflow:latest|$(MLFLOW_IMAGE)|g' \
		k8s/mlflow-migrate.yaml | $(K8S) apply -f -
	$(K8S) wait --for=condition=complete job/centaurdrug-mlflow-migrate --timeout=300s
	@sed 's|ghcr.io/aliaqil/centaurdrug-mlflow:latest|$(MLFLOW_IMAGE)|g' \
		k8s/mlflow-deployment.yaml | $(K8S) apply -f -
	$(K8S) rollout status deployment/centaurdrug-mlflow --timeout=300s
	@echo "MLflow is inside the cluster at http://centaurdrug-mlflow-service:5000"
	@echo "Open locally with: make k8s-mlflow-port-forward"

k8s-mlflow-local-up: k8s-mlflow-db-up
	-$(K8S) delete job centaurdrug-mlflow-migrate --ignore-not-found --wait=true
	@sed 's|ghcr.io/aliaqil/centaurdrug-mlflow:latest|$(MLFLOW_IMAGE)|g' \
		k8s/mlflow-migrate.yaml | $(K8S) apply -f -
	$(K8S) wait --for=condition=complete job/centaurdrug-mlflow-migrate --timeout=300s
	@sed 's|ghcr.io/aliaqil/centaurdrug-mlflow:latest|$(MLFLOW_IMAGE)|g' \
		k8s/mlflow-local-deployment.yaml | $(K8S) apply -f -
	$(K8S) rollout status deployment/centaurdrug-mlflow --timeout=300s
	@echo "MLflow local stack is ready. Run make k8s-mlflow-port-forward."

k8s-mlflow-port-forward:
	@echo "MLflow local URL: http://127.0.0.1:$(K8S_MLFLOW_LOCAL_PORT)"
	$(K8S) port-forward service/centaurdrug-mlflow-service $(K8S_MLFLOW_LOCAL_PORT):5000

k8s-mlflow-logs:
	$(K8S) logs -f deployment/centaurdrug-mlflow -c mlflow

airflow-image:
	docker build --file Dockerfile.airflow --tag $(AIRFLOW_IMAGE) .

k8s-airflow-secret: k8s-namespace
	@test -n "$(AIRFLOW_DATABASE_URI)" || (echo "Set AIRFLOW_DATABASE_URI."; exit 1)
	@test -n "$(AIRFLOW_ADMIN_PASSWORD)" || (echo "Set AIRFLOW_ADMIN_PASSWORD."; exit 1)
	@test -n "$(AIRFLOW_FERNET_KEY)" || (echo "Set AIRFLOW_FERNET_KEY."; exit 1)
	@test -n "$(AIRFLOW_JWT_SECRET)" || (echo "Set AIRFLOW_JWT_SECRET."; exit 1)
	@$(K8S) create secret generic centaurdrug-airflow-secret \
		--from-literal=AIRFLOW__DATABASE__SQL_ALCHEMY_CONN="$(AIRFLOW_DATABASE_URI)" \
		--from-literal=AIRFLOW__CORE__FERNET_KEY="$(AIRFLOW_FERNET_KEY)" \
		--from-literal=AIRFLOW__API_AUTH__JWT_SECRET="$(AIRFLOW_JWT_SECRET)" \
		--from-literal=AIRFLOW_ADMIN_USERNAME="$(AIRFLOW_ADMIN_USERNAME)" \
		--from-literal=AIRFLOW_ADMIN_PASSWORD="$(AIRFLOW_ADMIN_PASSWORD)" \
		--from-literal=POSTGRES_PASSWORD="$(AIRFLOW_POSTGRES_PASSWORD)" \
		--dry-run=client -o yaml | $(K8S) apply -f -

k8s-airflow-db-up: k8s-airflow-secret
	@test -n "$(AIRFLOW_POSTGRES_PASSWORD)" || (echo "Set AIRFLOW_POSTGRES_PASSWORD."; exit 1)
	$(K8S) apply -f k8s/airflow-postgres.yaml
	$(K8S) rollout status statefulset/centaurdrug-airflow-postgres --timeout=180s

k8s-airflow-up: k8s-airflow-secret
	-$(K8S) delete job centaurdrug-airflow-migrate --ignore-not-found --wait=true
	@sed 's|ghcr.io/aliaqil/centaurdrug-airflow:latest|$(AIRFLOW_IMAGE)|g' \
		k8s/airflow-deployment.yaml | $(K8S) apply -f -
	$(K8S) wait --for=condition=complete job/centaurdrug-airflow-migrate --timeout=300s
	$(K8S) rollout status deployment/centaurdrug-airflow-api-server --timeout=300s
	$(K8S) rollout status deployment/centaurdrug-airflow-scheduler --timeout=300s
	$(K8S) rollout status deployment/centaurdrug-airflow-dag-processor --timeout=300s
	$(K8S) rollout status deployment/centaurdrug-airflow-triggerer --timeout=300s
	@echo "Airflow is inside the cluster at http://centaurdrug-airflow-service:8080"
	@echo "Open locally with: make k8s-airflow-port-forward"

k8s-airflow-local-up: k8s-airflow-db-up k8s-airflow-up

k8s-airflow-port-forward:
	@echo "Airflow local URL: http://127.0.0.1:$(K8S_AIRFLOW_LOCAL_PORT)"
	$(K8S) port-forward service/centaurdrug-airflow-service $(K8S_AIRFLOW_LOCAL_PORT):8080

k8s-airflow-logs:
	$(K8S) logs -f deployment/centaurdrug-airflow-$(AIRFLOW_COMPONENT) -c $(AIRFLOW_COMPONENT)

k8s-mlops-up: k8s-mlflow-up k8s-airflow-up
	@echo "MLOps services are applied. Use the port-forward targets to open them locally."

k8s-mlops-down:
	-$(K8S) delete -f k8s/airflow-deployment.yaml --ignore-not-found
	-$(K8S) delete -f k8s/airflow-postgres.yaml --ignore-not-found
	-$(K8S) delete -f k8s/mlflow-deployment.yaml --ignore-not-found
	-$(K8S) delete -f k8s/mlflow-local-deployment.yaml --ignore-not-found
	-$(K8S) delete -f k8s/mlflow-postgres.yaml --ignore-not-found
	-$(K8S) delete -f k8s/mlflow-migrate.yaml --ignore-not-found

verify-mlflow:
	MLFLOW_TRACKING_URI="$(MLFLOW_TRACKING_URI)" \
		MLFLOW_TRACKING_USERNAME="$(MLFLOW_TRACKING_USERNAME)" \
		MLFLOW_TRACKING_PASSWORD="$(MLFLOW_TRACKING_PASSWORD)" \
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
	MLFLOW_TRACKING_URI="$(MLFLOW_TRACKING_URI)" \
		MLFLOW_TRACKING_USERNAME="$(MLFLOW_TRACKING_USERNAME)" \
		MLFLOW_TRACKING_PASSWORD="$(MLFLOW_TRACKING_PASSWORD)" \
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
