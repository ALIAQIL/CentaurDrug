from src.agent.constraints import evaluate_candidate_constraints


def test_constraints_pass_when_empty():
    result = evaluate_candidate_constraints("CCO")

    assert result["passed"] is True
    assert result["violations"] == []


def test_constraints_detect_avoided_substructure():
    result = evaluate_candidate_constraints(
        "O=[N+]([O-])c1ccccc1",
        constraints={"avoid_substructures": ["[N+](=O)[O-]"]},
    )

    assert result["passed"] is False
    assert "contains_avoided_substructure:[N+](=O)[O-]" in result["violations"]


def test_constraints_detect_descriptor_limits():
    result = evaluate_candidate_constraints(
        "CCCCCCCCCCCCCCCC",
        constraints={"max_mw": 100, "max_logp": 2},
    )

    assert result["passed"] is False
    assert any(violation.startswith("mw_above") for violation in result["violations"])
    assert any(
        violation.startswith("logp_above")
        for violation in result["violations"]
    )
