from src.agent.transformations import canonicalize_smiles, generate_candidates


ASPIRIN = "CC(=O)Oc1ccccc1C(=O)O"


def test_generate_candidates_round_robins_transformations():
    candidates = generate_candidates(
        ASPIRIN,
        transformations=[
            "aromatic_H_to_F",
            "aromatic_H_to_Cl",
            "aromatic_H_to_OH",
        ],
        max_total_candidates=3,
    )

    assert [candidate.transformation for candidate in candidates] == [
        "aromatic_H_to_F",
        "aromatic_H_to_Cl",
        "aromatic_H_to_OH",
    ]
    assert len({candidate.smiles for candidate in candidates}) == 3


def test_ester_hydrolysis_keeps_scaffold_sized_product():
    candidates = generate_candidates(
        ASPIRIN,
        transformations=["ester_to_acid"],
        max_total_candidates=10,
    )

    smiles = {candidate.smiles for candidate in candidates}

    assert canonicalize_smiles("O=C(O)c1ccccc1O") in smiles
    assert canonicalize_smiles("CC(=O)O") not in smiles
