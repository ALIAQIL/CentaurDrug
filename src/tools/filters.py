from rdkit import Chem
from rdkit.Chem import Descriptors, QED
from rdkit.Chem.FilterCatalog import FilterCatalog, FilterCatalogParams


def smiles_to_mol(smiles: str):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    try:
        Chem.SanitizeMol(mol)
        return mol
    except Exception:
        return None


def is_valid_smiles(smiles: str) -> bool:
    return smiles_to_mol(smiles) is not None


def lipinski_filter(smiles: str) -> dict:
    mol = smiles_to_mol(smiles)
    if mol is None:
        return {"valid": False}

    mw = Descriptors.MolWt(mol)
    logp = Descriptors.MolLogP(mol)
    hbd = Descriptors.NumHDonors(mol)
    hba = Descriptors.NumHAcceptors(mol)

    violations = 0
    violations += mw > 500
    violations += logp > 5
    violations += hbd > 5
    violations += hba > 10

    return {
        "valid": True,
        "mw": mw,
        "logp": logp,
        "hbd": hbd,
        "hba": hba,
        "violations": int(violations),
        "passed": violations <= 1,
    }


def qed_score(smiles: str) -> float | None:
    mol = smiles_to_mol(smiles)
    if mol is None:
        return None
    return float(QED.qed(mol))


def pains_filter(smiles: str) -> dict:
    mol = smiles_to_mol(smiles)
    if mol is None:
        return {"valid": False}

    params = FilterCatalogParams()
    params.AddCatalog(FilterCatalogParams.FilterCatalogs.PAINS)
    catalog = FilterCatalog(params)

    entry = catalog.GetFirstMatch(mol)

    return {
        "valid": True,
        "pains": entry is not None,
        "pains_description": entry.GetDescription() if entry else None,
        "passed": entry is None,
    }