from evals.langsmith_evaluators import (
    chat_safety_evaluator,
    optimization_constraints_evaluator,
    strategy_structure_evaluator,
)


def test_strategy_structure_evaluator_rejects_unknown_transformations():
    result = strategy_structure_evaluator(
        {"available_transformations": ["aromatic_H_to_F"]},
        {
            "selected_transformations": ["not_allowed"],
            "avoid_transformations": [],
        },
    )

    assert result["score"] == 0.0


def test_optimization_constraints_evaluator_accepts_passed_candidates():
    result = optimization_constraints_evaluator(
        {"constraints": {"max_mw": 500}},
        {"best_candidate_nodes": [{"constraints": {"passed": True}}]},
    )

    assert result["score"] == 1.0


def test_chat_safety_evaluator_requires_caution_and_risk_context():
    result = chat_safety_evaluator(
        {"context": {"main_risks": ["out_of_applicability_domain"]}},
        "This is a computational prediction; applicability-domain risk remains.",
    )

    assert result["score"] == 1.0
