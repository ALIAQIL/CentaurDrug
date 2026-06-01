from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, List, Sequence, Set

from rdkit import rdBase
from rdkit import Chem
from rdkit.Chem import AllChem


@dataclass(frozen=True)
class MoleculeCandidate:
    smiles: str
    parent_smiles: str
    transformation: str


REACTION_SMARTS: dict[str, Sequence[str]] = {
    # Very simple first-pass medicinal chemistry transformations.
    # These are conservative and mostly useful for prototype optimization.
    "aromatic_H_to_F": ("[cH:1]>>[c:1]F",),
    "aromatic_H_to_Cl": ("[cH:1]>>[c:1]Cl",),
    "aromatic_H_to_OH": ("[cH:1]>>[c:1]O",),
    "aromatic_H_to_OMe": ("[cH:1]>>[c:1]OC",),
    "aromatic_H_to_Me": ("[cH:1]>>[c:1]C",),

    # Reduce ester liability with hydrolysis-like products. Both products are
    # considered so aryl esters can retain the larger scaffold fragment.
    "ester_to_acid": (
        "[C:1](=[O:2])[O:3][#6:4]>>[C:1](=[O:2])O",
        "[C:1](=[O:2])[O:3][#6:4]>>[O:3][#6:4]",
    ),

    # Amide N-methylation/de-methylation examples are intentionally omitted for now
    # because they can easily create invalid or biologically misleading candidates.
}


def _quiet_rdkit_warnings():
    return rdBase.BlockLogs()


def canonicalize_smiles(smiles: str) -> str | None:
    mol = Chem.MolFromSmiles(smiles)

    if mol is None:
        return None

    return Chem.MolToSmiles(mol, canonical=True)


def is_valid_molecule(smiles: str) -> bool:
    return canonicalize_smiles(smiles) is not None


def apply_reaction(
    smiles: str,
    reaction_name: str,
    max_products: int = 20,
    min_heavy_atom_retention: float = 0.35,
) -> List[MoleculeCandidate]:
    if reaction_name not in REACTION_SMARTS:
        raise ValueError(f"Unknown reaction: {reaction_name}")

    parent = Chem.MolFromSmiles(smiles)

    if parent is None:
        return []

    parent_heavy_atoms = parent.GetNumHeavyAtoms()
    candidates: List[MoleculeCandidate] = []
    seen: Set[str] = set()

    for reaction_smarts in REACTION_SMARTS[reaction_name]:
        with _quiet_rdkit_warnings():
            reaction = AllChem.ReactionFromSmarts(reaction_smarts)
            products = reaction.RunReactants((parent,))

        for product_tuple in products:
            for product in product_tuple:
                try:
                    Chem.SanitizeMol(product)
                    product_smiles = Chem.MolToSmiles(product, canonical=True)
                except Exception:
                    continue

                if product_smiles in seen:
                    continue

                if parent_heavy_atoms:
                    retention = product.GetNumHeavyAtoms() / parent_heavy_atoms
                    if retention < min_heavy_atom_retention:
                        continue

                seen.add(product_smiles)

                candidates.append(
                    MoleculeCandidate(
                        smiles=product_smiles,
                        parent_smiles=smiles,
                        transformation=reaction_name,
                    )
                )

                if len(candidates) >= max_products:
                    return candidates

    return candidates


def generate_candidates(
    smiles: str,
    transformations: list[str] | None = None,
    max_candidates_per_transformation: int = 20,
    max_total_candidates: int = 100,
    excluded_smiles: Iterable[str] | None = None,
) -> List[MoleculeCandidate]:
    canonical_parent = canonicalize_smiles(smiles)

    if canonical_parent is None:
        return []

    if transformations is None:
        transformations = list(REACTION_SMARTS.keys())

    seen: Set[str] = {canonical_parent}

    if excluded_smiles is not None:
        seen.update(
            canonical
            for candidate_smiles in excluded_smiles
            if (canonical := canonicalize_smiles(candidate_smiles)) is not None
        )

    candidates_by_transformation: List[List[MoleculeCandidate]] = []
    for transformation in transformations:
        candidates = apply_reaction(
            canonical_parent,
            reaction_name=transformation,
            max_products=max_candidates_per_transformation,
        )

        filtered_candidates: List[MoleculeCandidate] = []
        local_seen: Set[str] = set()

        for candidate in candidates:
            canonical = canonicalize_smiles(candidate.smiles)

            if canonical is None:
                continue

            if canonical in seen or canonical in local_seen:
                continue

            local_seen.add(canonical)
            filtered_candidates.append(
                MoleculeCandidate(
                    smiles=canonical,
                    parent_smiles=canonical_parent,
                    transformation=candidate.transformation,
                )
            )

        candidates_by_transformation.append(filtered_candidates)

    all_candidates: List[MoleculeCandidate] = []

    while len(all_candidates) < max_total_candidates:
        added_candidate = False

        for candidates in candidates_by_transformation:
            while candidates:
                candidate = candidates.pop(0)

                if candidate.smiles in seen:
                    continue

                seen.add(candidate.smiles)
                all_candidates.append(candidate)
                added_candidate = True
                break

            if len(all_candidates) >= max_total_candidates:
                break

        if not added_candidate:
            break

    return all_candidates
