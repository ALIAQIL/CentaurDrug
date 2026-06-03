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


def test_rules_endpoint():
    response = client.post("/rules", json={"smiles": "CCO"})

    assert response.status_code == 200
    assert response.json()["valid"] is True


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
