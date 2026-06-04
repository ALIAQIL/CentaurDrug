from __future__ import annotations

from typing import Any, Dict

from rdkit import Chem
from rdkit.Chem import Descriptors, rdMolDescriptors

from src.agent.scoring import compute_scaffold_preservation


def evaluate_candidate_constraints(
    smiles: str,
    constraints: Dict[str, Any] | None = None,
    reference_smiles: str | None = None,
) -> Dict[str, Any]:
    constraints = constraints or {}

    if not constraints:
        return {
            "passed": True,
            "violations": [],
            "descriptors": {},
        }

    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return {
            "passed": False,
            "violations": ["invalid_smiles"],
            "descriptors": {},
        }

    descriptors = {
        "mw": float(Descriptors.MolWt(mol)),
        "logp": float(Descriptors.MolLogP(mol)),
        "tpsa": float(rdMolDescriptors.CalcTPSA(mol)),
    }
    violations = []

    max_mw = constraints.get("max_mw")
    if max_mw is not None and descriptors["mw"] > float(max_mw):
        violations.append(f"mw_above_{float(max_mw):g}")

    max_logp = constraints.get("max_logp")
    if max_logp is not None and descriptors["logp"] > float(max_logp):
        violations.append(f"logp_above_{float(max_logp):g}")

    max_tpsa = constraints.get("max_tpsa")
    if max_tpsa is not None and descriptors["tpsa"] > float(max_tpsa):
        violations.append(f"tpsa_above_{float(max_tpsa):g}")

    for smarts in constraints.get("avoid_substructures", []):
        pattern = Chem.MolFromSmarts(smarts)
        if pattern is None:
            violations.append(f"invalid_avoid_smarts:{smarts}")
        elif mol.HasSubstructMatch(pattern):
            violations.append(f"contains_avoided_substructure:{smarts}")

    min_scaffold = constraints.get("min_scaffold_preservation")
    if min_scaffold is not None:
        scaffold = compute_scaffold_preservation(reference_smiles, smiles)
        descriptors["scaffold_preservation"] = scaffold.get(
            "preservation_score"
        )

        if scaffold.get("preservation_score", 0.0) < float(min_scaffold):
            violations.append(f"scaffold_below_{float(min_scaffold):g}")

    return {
        "passed": not violations,
        "violations": violations,
        "descriptors": descriptors,
    }
