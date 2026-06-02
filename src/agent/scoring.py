from __future__ import annotations

from typing import Any, Dict

from rdkit import Chem, DataStructs
from rdkit.Chem import Descriptors, Lipinski, rdFingerprintGenerator
from rdkit.Chem.Scaffolds import MurckoScaffold


MORGAN_GENERATOR = rdFingerprintGenerator.GetMorganGenerator(
    radius=2,
    fpSize=2048,
)


def clamp(value: float, low: float = 0.0, high: float = 1.0) -> float:
    return max(low, min(high, value))


def _get_evaluation_smiles(evaluation: Dict[str, Any]) -> str | None:
    return evaluation.get("canonical_smiles") or evaluation.get("original_smiles")


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


def compute_scaffold_preservation(
    parent_smiles: str | None,
    candidate_smiles: str | None,
) -> Dict[str, Any]:
    """
    Estimate whether a candidate still resembles the parent molecule.

    The main score uses Morgan fingerprint Tanimoto similarity. A Murcko scaffold
    match adds an explainable bonus when both molecules have ring scaffolds.
    """

    if not parent_smiles or not candidate_smiles:
        return {
            "status": "missing_smiles",
            "parent_scaffold": None,
            "candidate_scaffold": None,
            "murcko_preserved": False,
            "fingerprint_similarity": 0.5,
            "preservation_score": 0.5,
            "interpretation": "unknown",
        }

    parent_mol = Chem.MolFromSmiles(parent_smiles)
    candidate_mol = Chem.MolFromSmiles(candidate_smiles)

    if parent_mol is None or candidate_mol is None:
        return {
            "status": "invalid_smiles",
            "parent_scaffold": None,
            "candidate_scaffold": None,
            "murcko_preserved": False,
            "fingerprint_similarity": 0.0,
            "preservation_score": 0.0,
            "interpretation": "invalid",
        }

    parent_scaffold = MurckoScaffold.MurckoScaffoldSmiles(mol=parent_mol)
    candidate_scaffold = MurckoScaffold.MurckoScaffoldSmiles(mol=candidate_mol)
    has_scaffold = bool(parent_scaffold or candidate_scaffold)
    murcko_preserved = (
        has_scaffold
        and parent_scaffold == candidate_scaffold
    )

    parent_fp = MORGAN_GENERATOR.GetFingerprint(parent_mol)
    candidate_fp = MORGAN_GENERATOR.GetFingerprint(candidate_mol)
    similarity = float(DataStructs.TanimotoSimilarity(parent_fp, candidate_fp))

    if has_scaffold:
        preservation_score = clamp(
            0.75 * similarity + 0.25 * float(murcko_preserved)
        )
    else:
        preservation_score = clamp(similarity)

    if preservation_score >= 0.7:
        interpretation = "strong"
    elif preservation_score >= 0.5:
        interpretation = "moderate"
    else:
        interpretation = "weak"

    return {
        "status": "ok",
        "parent_scaffold": parent_scaffold or None,
        "candidate_scaffold": candidate_scaffold or None,
        "murcko_preserved": murcko_preserved,
        "fingerprint_similarity": similarity,
        "preservation_score": preservation_score,
        "interpretation": interpretation,
    }


def compute_synthetic_accessibility_proxy(smiles: str | None) -> Dict[str, Any]:
    """
    Lightweight synthetic accessibility proxy.

    This is not the full RDKit contrib SA score. It is an interpretable heuristic
    that penalizes high molecular weight, excessive rotatable bonds, many rings,
    many stereocenters, extreme heteroatom ratios, and high graph complexity.
    """

    if not smiles:
        return {
            "status": "missing_smiles",
            "score": 0.5,
            "interpretation": "unknown",
        }

    mol = Chem.MolFromSmiles(smiles)

    if mol is None:
        return {
            "status": "invalid_smiles",
            "score": 0.0,
            "interpretation": "invalid",
        }

    heavy_atoms = max(mol.GetNumHeavyAtoms(), 1)
    molecular_weight = float(Descriptors.MolWt(mol))
    rotatable_bonds = int(Lipinski.NumRotatableBonds(mol))
    ring_count = int(Lipinski.RingCount(mol))
    stereocenters = len(Chem.FindMolChiralCenters(mol, includeUnassigned=True))
    heteroatoms = sum(
        1
        for atom in mol.GetAtoms()
        if atom.GetAtomicNum() not in {1, 6}
    )
    heteroatom_ratio = heteroatoms / heavy_atoms
    complexity = float(Descriptors.BertzCT(mol))

    penalties = {
        "molecular_weight": clamp((molecular_weight - 500.0) / 300.0),
        "rotatable_bonds": clamp((rotatable_bonds - 10.0) / 10.0),
        "ring_count": clamp((ring_count - 4.0) / 4.0),
        "stereocenters": clamp((stereocenters - 3.0) / 5.0),
        "heteroatom_ratio": clamp((heteroatom_ratio - 0.45) / 0.35),
        "complexity": clamp((complexity - 900.0) / 700.0),
    }
    weights = {
        "molecular_weight": 0.22,
        "rotatable_bonds": 0.18,
        "ring_count": 0.15,
        "stereocenters": 0.15,
        "heteroatom_ratio": 0.10,
        "complexity": 0.20,
    }

    total_penalty = sum(
        weights[key] * penalties[key]
        for key in weights
    )
    score = clamp(1.0 - total_penalty)

    if score >= 0.75:
        interpretation = "easy"
    elif score >= 0.5:
        interpretation = "moderate"
    else:
        interpretation = "difficult"

    return {
        "status": "ok",
        "score": score,
        "interpretation": interpretation,
        "features": {
            "molecular_weight": molecular_weight,
            "rotatable_bonds": rotatable_bonds,
            "ring_count": ring_count,
            "stereocenters": stereocenters,
            "heteroatom_count": heteroatoms,
            "heteroatom_ratio": heteroatom_ratio,
            "bertz_complexity": complexity,
        },
        "penalties": penalties,
    }


def build_score_vector(
    evaluation: Dict[str, Any],
    reference_smiles: str | None = None,
) -> Dict[str, Any]:
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
                "scaffold_preservation": 0.0,
                "synthetic_accessibility": 0.0,
            },
            "scalar_score": 0.0,
        }

    rules = evaluation.get("rules", {})
    admet = evaluation.get("admet_predictions", {})
    candidate_smiles = _get_evaluation_smiles(evaluation)
    reference_smiles = reference_smiles or candidate_smiles

    scaffold = compute_scaffold_preservation(
        parent_smiles=reference_smiles,
        candidate_smiles=candidate_smiles,
    )
    synthetic_accessibility = compute_synthetic_accessibility_proxy(
        candidate_smiles
    )

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
        "scaffold": scaffold,
        "synthetic_accessibility": synthetic_accessibility,
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
        "scaffold_preservation": float(scaffold["preservation_score"]),
        "synthetic_accessibility": float(synthetic_accessibility["score"]),
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
        "scaffold_preservation": 1.2,
        "synthetic_accessibility": 0.8,
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


def build_candidate_explanation(
    parent_vector: Dict[str, Any],
    candidate_vector: Dict[str, Any],
    candidate_evaluation: Dict[str, Any] | None = None,
    delta_threshold: float = 0.03,
) -> Dict[str, Any]:
    comparison = compare_score_vectors(parent_vector, candidate_vector)
    normalized = candidate_vector["normalized"]
    raw = candidate_vector.get("raw", {})
    delta_vector = comparison["delta_vector"]

    labels = {
        "solubility": "solubility",
        "lipophilicity": "lipophilicity balance",
        "ames_safety": "AMES safety",
        "herg_safety": "hERG safety",
        "cyp3a4_safety": "CYP3A4 safety",
        "qed": "QED drug-likeness",
        "lipinski": "Lipinski compliance",
        "pains": "PAINS filter status",
        "applicability_domain": "prediction applicability domain",
        "scaffold_preservation": "scaffold preservation",
        "synthetic_accessibility": "synthetic accessibility",
    }

    improvements = []
    tradeoffs = []

    for key, delta in delta_vector.items():
        if key in {"scaffold_preservation", "synthetic_accessibility"}:
            continue

        label = labels.get(key, key)

        if delta >= delta_threshold:
            improvements.append(f"{label} improved")
        elif delta <= -delta_threshold:
            tradeoffs.append(f"{label} decreased")

    candidate_evaluation = candidate_evaluation or {}
    risks = candidate_evaluation.get("overall_assessment", {}).get(
        "main_risks",
        [],
    )

    if "out_of_applicability_domain" in risks:
        tradeoffs.append(
            "one or more ADMET predictions remain outside the applicability domain"
        )

    scaffold = raw.get("scaffold", {})
    if scaffold.get("interpretation") == "weak":
        tradeoffs.append("candidate weakly preserves the parent scaffold")
    elif scaffold.get("interpretation") == "strong":
        improvements.append("candidate strongly preserves the parent scaffold")

    sa = raw.get("synthetic_accessibility", {})
    if sa.get("interpretation") == "difficult":
        tradeoffs.append("candidate may be difficult to synthesize")
    elif sa.get("interpretation") == "easy":
        improvements.append("candidate has a favorable synthetic accessibility proxy")

    return {
        "scalar_score": candidate_vector["scalar_score"],
        "delta_vs_parent": comparison["delta_scalar_score"],
        "score_vector": normalized,
        "parent_comparison": comparison,
        "improvements": improvements,
        "tradeoffs": tradeoffs,
        "scaffold": scaffold,
        "synthetic_accessibility": sa,
    }
