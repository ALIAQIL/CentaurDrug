from src.tools.filters import (
    is_valid_smiles,
    lipinski_filter,
    pains_filter,
    qed_score,
)


def evaluate_rules(smiles: str) -> dict:
    if not is_valid_smiles(smiles):
        return {
            "valid": False,
            "decision": "reject",
            "reason": "Invalid SMILES",
        }

    lipinski = lipinski_filter(smiles)
    pains = pains_filter(smiles)
    qed = qed_score(smiles)

    passed = lipinski["passed"] and pains["passed"] and qed >= 0.35

    return {
        "valid": True,
        "lipinski": lipinski,
        "pains": pains,
        "qed": qed,
        "decision": "pass" if passed else "reject",
    }