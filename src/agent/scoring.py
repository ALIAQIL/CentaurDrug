from __future__ import annotations

from typing import Any, Dict


def clamp(value: float, low: float = 0.0, high: float = 1.0) -> float:
    return max(low, min(high, value))


def normalize_solubility(log_solubility: float | None) -> float:
    """
    Rough normalization for AqSolDB-like log solubility.

    Very low solubility around -6 -> near 0.
    Good solubility around 0 or higher -> near 1.
    """

    if log_solubility is None:
        return 0.5

    return clamp((float(log_solubility) + 6.0) / 6.0)


def normalize_lipophilicity(value: float | None, target: float = 2.0) -> float:
    """
    Prefer moderate lipophilicity around target=2.
    Penalize values far away.
    """

    if value is None:
        return 0.5

    return clamp(1.0 - abs(float(value) - target) / 4.0)


def probability_to_safety(probability_positive: float | None) -> float:
    """
    For AMES/hERG/CYP3A4:
    probability_positive = risk probability.
    safety score = 1 - risk probability.
    """

    if probability_positive is None:
        return 0.5

    return clamp(1.0 - float(probability_positive))


def extract_ad_score(admet_predictions: Dict[str, Any]) -> float:
    scores = []

    for result in admet_predictions.values():
        if result.get("status") == "ok":
            value = result.get("ad_score")
            if value is not None:
                scores.append(float(value))

    if not scores:
        return 0.0

    return clamp(sum(scores) / len(scores))


def build_score_vector(evaluation: Dict[str, Any]) -> Dict[str, Any]:
    """
    Build a vector of interpretable scores.

    This is designed for:
    - optimization ranking
    - LLM reasoning
    - human review
    """

    if evaluation.get("status") != "ok":
        return {
            "raw": {
                "status": evaluation.get("status"),
                "reason": evaluation.get("reason"),
            },
            "normalized": {
                "solubility": 0.0,
                "lipophilicity": 0.0,
                "ames_safety": 0.0,
                "herg_safety": 0.0,
                "cyp3a4_safety": 0.0,
                "qed": 0.0,
                "lipinski": 0.0,
                "pains": 0.0,
                "applicability_domain": 0.0,
            },
            "scalar_score": 0.0,
        }

    rules = evaluation.get("rules", {})
    admet = evaluation.get("admet_predictions", {})

    sol = admet.get("solubility", {})
    lipo = admet.get("lipophilicity", {})
    ames = admet.get("ames", {})
    herg = admet.get("herg", {})
    cyp3a4 = admet.get("cyp3a4", {})

    raw = {
        "solubility": sol.get("prediction"),
        "lipophilicity": lipo.get("prediction"),
        "ames_probability": ames.get("probability_positive"),
        "herg_probability": herg.get("probability_positive"),
        "cyp3a4_probability": cyp3a4.get("probability_positive"),
        "qed": rules.get("qed"),
        "lipinski_passed": rules.get("lipinski", {}).get("passed"),
        "pains_passed": rules.get("pains", {}).get("passed"),
    }

    normalized = {
        "solubility": normalize_solubility(raw["solubility"]),
        "lipophilicity": normalize_lipophilicity(raw["lipophilicity"]),
        "ames_safety": probability_to_safety(raw["ames_probability"]),
        "herg_safety": probability_to_safety(raw["herg_probability"]),
        "cyp3a4_safety": probability_to_safety(raw["cyp3a4_probability"]),
        "qed": clamp(float(raw["qed"])) if raw["qed"] is not None else 0.5,
        "lipinski": 1.0 if raw["lipinski_passed"] else 0.0,
        "pains": 1.0 if raw["pains_passed"] else 0.0,
        "applicability_domain": extract_ad_score(admet),
    }

    scalar_score = compute_scalar_score(normalized)

    return {
        "raw": raw,
        "normalized": normalized,
        "scalar_score": scalar_score,
    }


def compute_scalar_score(normalized: Dict[str, float]) -> float:
    """
    Weighted score for ranking.

    The LLM should receive the vector.
    The optimizer can use the scalar for sorting.
    """

    weights = {
        "solubility": 1.2,
        "lipophilicity": 1.0,
        "ames_safety": 1.5,
        "herg_safety": 1.5,
        "cyp3a4_safety": 1.0,
        "qed": 0.8,
        "lipinski": 0.8,
        "pains": 1.0,
        "applicability_domain": 0.8,
    }

    total = 0.0

    for key, weight in weights.items():
        total += weight * normalized.get(key, 0.0)

    return float(total)


def compare_score_vectors(
    parent_vector: Dict[str, Any],
    candidate_vector: Dict[str, Any],
) -> Dict[str, Any]:
    parent_norm = parent_vector["normalized"]
    candidate_norm = candidate_vector["normalized"]

    deltas = {}

    for key, candidate_value in candidate_norm.items():
        parent_value = parent_norm.get(key, 0.0)
        deltas[key] = float(candidate_value - parent_value)

    return {
        "parent_scalar_score": parent_vector["scalar_score"],
        "candidate_scalar_score": candidate_vector["scalar_score"],
        "delta_scalar_score": candidate_vector["scalar_score"]
        - parent_vector["scalar_score"],
        "delta_vector": deltas,
    }
