from fastapi.testclient import TestClient

from src.api import main


client = TestClient(main.app)


def test_index_serves_frontend():
    response = client.get("/")

    assert response.status_code == 200
    assert "text/html" in response.headers["content-type"]
    assert "CentaurDrug" in response.text


def test_health_endpoint():
    response = client.get("/health")

    assert response.status_code == 200
    assert response.json()["status"] == "ok"


def test_version_endpoint():
    response = client.get("/version")

    assert response.status_code == 200
    assert response.json()["api_version"] == main.app.version


def test_metrics_endpoint():
    response = client.get("/metrics")

    assert response.status_code == 200
    assert "centaurdrug_requests_total" in response.text


def test_rules_endpoint():
    response = client.post("/rules", json={"smiles": "CCO"})

    assert response.status_code == 200
    assert response.json()["valid"] is True
    assert "veber" in response.json()
    assert "brenk" in response.json()


def test_rejects_unapproved_model_root():
    response = client.post(
        "/evaluate",
        json={"smiles": "CCO", "model_root": "../outside"},
    )

    assert response.status_code == 400
    assert response.json()["detail"] == "model_root is not allowed for this API."


def test_evaluate_endpoint_uses_panel_evaluator(monkeypatch):
    class DummyEvaluator:
        def evaluate_molecule(self, smiles):
            return {
                "status": "ok",
                "canonical_smiles": smiles,
                "final_decision": {"decision": "pass"},
            }

    monkeypatch.setattr(main, "get_evaluator", lambda model_root: DummyEvaluator())

    response = client.post("/evaluate", json={"smiles": "CCO"})

    assert response.status_code == 200
    assert response.json() == {
        "status": "ok",
        "canonical_smiles": "CCO",
        "final_decision": {"decision": "pass"},
    }


def test_evaluate_endpoint_hides_internal_errors(monkeypatch):
    class BrokenEvaluator:
        def evaluate_molecule(self, smiles):
            raise RuntimeError("secret filesystem detail")

    monkeypatch.setattr(main, "get_evaluator", lambda model_root: BrokenEvaluator())

    response = client.post("/evaluate", json={"smiles": "CCO"})

    assert response.status_code == 500
    assert "secret filesystem detail" not in response.json()["detail"]


def test_optimize_endpoint_runs_agent_graph(monkeypatch):
    def fake_run_graph(**kwargs):
        return {
            "status": "ok",
            "input_smiles": kwargs["smiles"],
            "max_depth": kwargs["max_depth"],
            "best_candidate_nodes": [],
            "tree_summary": {"root_id": "root", "nodes": {}},
        }

    monkeypatch.setattr(main, "run_graph", fake_run_graph)

    response = client.post(
        "/optimize",
        json={
            "smiles": "CCO",
            "max_depth": 1,
            "beam_width": 2,
            "max_candidates_per_node": 3,
        },
    )

    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "ok"
    assert data["input_smiles"] == "CCO"
    assert data["max_depth"] == 1


def test_optimize_endpoint_passes_constraints(monkeypatch):
    def fake_run_graph(**kwargs):
        return {
            "status": "ok",
            "constraints": kwargs["constraints"],
            "best_candidate_nodes": [],
        }

    monkeypatch.setattr(main, "run_graph", fake_run_graph)

    response = client.post(
        "/optimize",
        json={
            "smiles": "CCO",
            "constraints": {
                "avoid_substructures": ["[N+](=O)[O-]"],
                "max_mw": 500,
            },
        },
    )

    assert response.status_code == 200
    assert response.json()["constraints"] == {
        "avoid_substructures": ["[N+](=O)[O-]"],
        "max_mw": 500.0,
    }


def test_chat_endpoint_fallback(monkeypatch):
    monkeypatch.setattr(main, "get_chat_llm", lambda: None)

    response = client.post(
        "/chat",
        json={
            "smiles": "CCO",
            "prompt": "Explain the current drug.",
            "context": {
                "type": "candidate",
                "candidate": {
                    "smiles": "CCO",
                    "transformation": "aromatic_H_to_OH",
                    "scalar_score": 7.1,
                    "delta_vs_parent": 0.2,
                    "improvements": ["solubility improved"],
                    "tradeoffs": ["hERG remains out of domain"],
                },
                "main_risks": ["out_of_applicability_domain"],
            },
        },
    )

    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "ok"
    assert data["mode"] == "fallback"
    assert "CCO" in data["reply"]
    assert data["suggested_prompts"]


def test_chat_endpoint_uses_llm_when_available(monkeypatch):
    class FakeResult:
        content = "LLM explanation"

    class FakeLlm:
        def invoke(self, prompt):
            assert "CentaurDrug medicinal chemistry copilot" in prompt
            return FakeResult()

    monkeypatch.setattr(main, "get_chat_llm", lambda: FakeLlm())

    response = client.post(
        "/chat",
        json={
            "smiles": "CCO",
            "prompt": "What changed?",
            "messages": [
                {
                    "role": "user",
                    "content": "Explain this candidate.",
                }
            ],
        },
    )

    assert response.status_code == 200
    data = response.json()
    assert data["mode"] == "gemini"
    assert data["reply"] == "LLM explanation"
