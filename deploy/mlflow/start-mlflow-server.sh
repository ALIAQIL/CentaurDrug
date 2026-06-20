#!/bin/sh
set -eu

: "${MLFLOW_BACKEND_STORE_URI:?MLFLOW_BACKEND_STORE_URI is required}"
: "${MLFLOW_AUTH_DATABASE_URI:?MLFLOW_AUTH_DATABASE_URI is required}"
: "${MLFLOW_ARTIFACTS_DESTINATION:?MLFLOW_ARTIFACTS_DESTINATION is required}"
: "${MLFLOW_ADMIN_USERNAME:?MLFLOW_ADMIN_USERNAME is required}"
: "${MLFLOW_ADMIN_PASSWORD:?MLFLOW_ADMIN_PASSWORD is required}"
: "${MLFLOW_FLASK_SERVER_SECRET_KEY:?MLFLOW_FLASK_SERVER_SECRET_KEY is required}"

MLFLOW_HOST="${MLFLOW_HOST:-0.0.0.0}"
MLFLOW_PORT="${MLFLOW_PORT:-5000}"
MLFLOW_WORKERS="${MLFLOW_WORKERS:-2}"
MLFLOW_ALLOWED_HOSTS="${MLFLOW_ALLOWED_HOSTS:-localhost,127.0.0.1}"
MLFLOW_CORS_ALLOWED_ORIGINS="${MLFLOW_CORS_ALLOWED_ORIGINS:-http://localhost:*}"
MLFLOW_AUTH_CONFIG_PATH="${MLFLOW_AUTH_CONFIG_PATH:-/tmp/mlflow/basic_auth.ini}"

escape_ini_value() {
    printf '%s' "$1" | sed 's/%/%%/g'
}

mkdir -p "$(dirname "$MLFLOW_AUTH_CONFIG_PATH")"
umask 077
cat > "$MLFLOW_AUTH_CONFIG_PATH" <<EOF
[mlflow]
default_permission = READ
database_uri = $(escape_ini_value "$MLFLOW_AUTH_DATABASE_URI")
admin_username = $(escape_ini_value "$MLFLOW_ADMIN_USERNAME")
admin_password = $(escape_ini_value "$MLFLOW_ADMIN_PASSWORD")
authorization_function = mlflow.server.auth:authenticate_request_basic_auth
grant_default_workspace_access = false
auth_cache_max_size = 10000
auth_cache_ttl_seconds = 30
EOF

export MLFLOW_AUTH_CONFIG_PATH

exec mlflow server \
    --app-name basic-auth \
    --host "$MLFLOW_HOST" \
    --port "$MLFLOW_PORT" \
    --workers "$MLFLOW_WORKERS" \
    --backend-store-uri "$MLFLOW_BACKEND_STORE_URI" \
    --artifacts-destination "$MLFLOW_ARTIFACTS_DESTINATION" \
    --serve-artifacts \
    --allowed-hosts "$MLFLOW_ALLOWED_HOSTS" \
    --cors-allowed-origins "$MLFLOW_CORS_ALLOWED_ORIGINS"
