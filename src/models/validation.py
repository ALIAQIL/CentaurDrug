from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional

from rdkit import Chem


@dataclass(frozen=True)
class MoleculeValidationResult:
    original_smiles: str
    is_valid: bool
    canonical_smiles: Optional[str] = None
    rejection_reason: Optional[str] = None


def validate_smiles(smiles: Any) -> MoleculeValidationResult:

    if smiles is None:
        return MoleculeValidationResult(
            original_smiles="",
            is_valid=False,
            rejection_reason="missing_smiles",
        )

    smiles_str = str(smiles).strip()

    if smiles_str == "":
        return MoleculeValidationResult(
            original_smiles=str(smiles),
            is_valid=False,
            rejection_reason="empty_smiles",
        )

    mol = Chem.MolFromSmiles(smiles_str)

    if mol is None:
        return MoleculeValidationResult(
            original_smiles=smiles_str,
            is_valid=False,
            rejection_reason="invalid_smiles",
        )

    canonical = Chem.MolToSmiles(mol, canonical=True)

    return MoleculeValidationResult(
        original_smiles=smiles_str,
        is_valid=True,
        canonical_smiles=canonical,
    )