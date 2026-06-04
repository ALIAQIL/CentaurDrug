from src.agent.optimizer import extract_candidate_score, summarize_improvement
from src.agent.scoring import (
    build_candidate_explanation,
    build_score_vector,
    compute_scaffold_preservation,
    compute_synthetic_accessibility_proxy,
    select_diverse_items,
)


def _evaluation(
    *,
    solubility=-2.0,
    lipophilicity=2.0,
    ames_probability=0.1,
    herg_probability=0.2,
    cyp3a4_probability=0.3,
    qed=0.6,
):
    return {
        "status": "ok",
        "canonical_smiles": "CCO",
        "rules": {
            "qed": qed,
            "lipinski": {"passed": True},
            "veber": {"passed": True},
            "pains": {"passed": True},
            "brenk": {"passed": True},
        },
        "admet_predictions": {
            "solubility": {
                "status": "ok",
                "prediction": solubility,
                "ad_score": 1.0,
            },
            "lipophilicity": {
                "status": "ok",
                "prediction": lipophilicity,
                "ad_score": 1.0,
            },
            "ames": {
                "status": "ok",
                "probability_positive": ames_probability,
                "ad_score": 1.0,
            },
            "herg": {
                "status": "ok",
                "probability_positive": herg_probability,
                "ad_score": 1.0,
            },
            "cyp3a4": {
                "status": "ok",
                "probability_positive": cyp3a4_probability,
                "ad_score": 1.0,
            },
        },
    }


def test_rejected_evaluation_scores_as_zero():
    score_vector = build_score_vector(
        {
            "status": "rejected",
            "reason": "invalid_smiles",
        }
    )

    assert score_vector["scalar_score"] == 0.0
    assert all(value == 0.0 for value in score_vector["normalized"].values())


def test_optimizer_score_uses_shared_score_vector():
    evaluation = _evaluation()
    score_vector = build_score_vector(evaluation)

    assert extract_candidate_score(evaluation) == score_vector["scalar_score"]
    assert 0.0 <= score_vector["scalar_score"] <= 1.0
    assert score_vector["score_metadata"]["scale"] == "0_to_1"


def test_summarize_improvement_keeps_legacy_and_vector_keys():
    parent = _evaluation(ames_probability=0.4)
    candidate = _evaluation(ames_probability=0.1)

    improvement = summarize_improvement(parent, candidate)

    assert improvement["delta_score"] > 0.0
    assert improvement["delta_scalar_score"] == improvement["delta_score"]
    assert improvement["candidate_scalar_score"] == improvement["candidate_score"]
    assert improvement["delta_vector"]["ames_safety"] > 0.0


def test_scaffold_preservation_scores_close_analogues():
    parent = "CC(=O)Oc1ccccc1C(=O)O"
    candidate = "CC(=O)Oc1cc(F)ccc1C(=O)O"

    scaffold = compute_scaffold_preservation(parent, candidate)

    assert scaffold["status"] == "ok"
    assert scaffold["murcko_preserved"] is True
    assert scaffold["preservation_score"] >= 0.5
    assert scaffold["parent_scaffold"] == scaffold["candidate_scaffold"]


def test_synthetic_accessibility_proxy_scores_simple_molecule_as_easy():
    result = compute_synthetic_accessibility_proxy("CC(=O)Oc1ccccc1C(=O)O")

    assert result["status"] == "ok"
    assert result["score"] >= 0.75
    assert result["interpretation"] == "easy"
    assert "molecular_weight" in result["features"]


def test_candidate_explanation_reports_improvements_and_tradeoffs():
    parent = _evaluation(ames_probability=0.4)
    candidate = _evaluation(ames_probability=0.1)
    candidate["overall_assessment"] = {
        "main_risks": ["out_of_applicability_domain"],
    }

    parent_vector = build_score_vector(parent, reference_smiles="CCO")
    candidate_vector = build_score_vector(candidate, reference_smiles="CCO")

    explanation = build_candidate_explanation(
        parent_vector=parent_vector,
        candidate_vector=candidate_vector,
        candidate_evaluation=candidate,
    )

    assert explanation["delta_vs_parent"] > 0.0
    assert "score_vector" in explanation
    assert any("AMES safety improved" in item for item in explanation["improvements"])
    assert any(
        "outside the applicability domain" in item
        for item in explanation["tradeoffs"]
    )


def test_select_diverse_items_prefers_nonredundant_candidates():
    items = [
        {"smiles": "CCO", "score": 0.9},
        {"smiles": "CCCO", "score": 0.8},
        {"smiles": "c1ccccc1", "score": 0.7},
    ]

    selected = select_diverse_items(
        items,
        top_k=2,
        smiles_getter=lambda item: item["smiles"],
        score_getter=lambda item: item["score"],
        similarity_threshold=0.2,
    )

    assert selected[0]["smiles"] == "CCO"
    assert selected[1]["smiles"] == "c1ccccc1"
