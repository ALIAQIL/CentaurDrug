from src.agent.optimizer import extract_candidate_score, summarize_improvement
from src.agent.scoring import build_score_vector


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
        "rules": {
            "qed": qed,
            "lipinski": {"passed": True},
            "pains": {"passed": True},
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

    assert extract_candidate_score(evaluation) == build_score_vector(evaluation)[
        "scalar_score"
    ]


def test_summarize_improvement_keeps_legacy_and_vector_keys():
    parent = _evaluation(ames_probability=0.4)
    candidate = _evaluation(ames_probability=0.1)

    improvement = summarize_improvement(parent, candidate)

    assert improvement["delta_score"] > 0.0
    assert improvement["delta_scalar_score"] == improvement["delta_score"]
    assert improvement["candidate_scalar_score"] == improvement["candidate_score"]
    assert improvement["delta_vector"]["ames_safety"] > 0.0
