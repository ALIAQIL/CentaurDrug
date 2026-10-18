import json
from pathlib import Path

import yaml


K8S_DIR = Path(__file__).parents[1] / "k8s"


def load_documents(path: Path) -> list[dict]:
    return [doc for doc in yaml.safe_load_all(path.read_text()) if doc]


def find_document(documents: list[dict], kind: str, name: str) -> dict:
    return next(
        document
        for document in documents
        if document.get("kind") == kind
        and document.get("metadata", {}).get("name") == name
    )


def test_kubernetes_manifests_parse_and_dashboard_is_valid_json():
    for path in K8S_DIR.glob("*.yaml"):
        assert load_documents(path), path

    grafana_documents = load_documents(K8S_DIR / "grafana.yaml")
    dashboard = find_document(
        grafana_documents,
        "ConfigMap",
        "centaurdrug-grafana-dashboard",
    )
    dashboard_json = json.loads(dashboard["data"]["centaurdrug-api.json"])
    titles = {panel["title"] for panel in dashboard_json["panels"]}

    assert {
        "Request rate by route",
        "5xx error rate",
        "p95 latency by route",
        "API readiness",
        "Chat fallback rate",
        "Optimization rate",
        "Evaluation rate",
    } <= titles


def test_prometheus_selectors_match_api_service_and_monitoring_resources():
    api_documents = load_documents(K8S_DIR / "api-deployment.yaml")
    monitoring_documents = load_documents(K8S_DIR / "monitoring.yaml")

    api_service = find_document(
        api_documents,
        "Service",
        "centaurdrug-api-service",
    )
    service_monitor = find_document(monitoring_documents, "ServiceMonitor", "centaurdrug-api")
    prometheus = find_document(
        monitoring_documents,
        "Prometheus",
        "centaurdrug-prometheus",
    )
    prometheus_rule = find_document(
        monitoring_documents,
        "PrometheusRule",
        "centaurdrug-api",
    )

    service_labels = api_service["metadata"]["labels"]
    service_selector = service_monitor["spec"]["selector"]["matchLabels"]
    assert all(service_labels[key] == value for key, value in service_selector.items())

    monitor_labels = service_monitor["metadata"]["labels"]
    rule_labels = prometheus_rule["metadata"]["labels"]
    monitor_selector = prometheus["spec"]["serviceMonitorSelector"]["matchLabels"]
    rule_selector = prometheus["spec"]["ruleSelector"]["matchLabels"]
    assert all(monitor_labels[key] == value for key, value in monitor_selector.items())
    assert all(rule_labels[key] == value for key, value in rule_selector.items())


def test_monitoring_manifests_do_not_contain_secret_values():
    grafana_text = (K8S_DIR / "grafana.yaml").read_text()

    assert "kind: Secret" not in grafana_text
    assert "GF_SECURITY_ADMIN_PASSWORD: " not in grafana_text
