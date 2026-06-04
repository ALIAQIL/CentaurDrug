from functools import lru_cache

from rdkit import Chem
from rdkit.Chem import Descriptors, Lipinski, QED, rdMolDescriptors
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


@lru_cache(maxsize=None)
def _filter_catalog(name: str) -> FilterCatalog:
    params = FilterCatalogParams()
    catalog = getattr(FilterCatalogParams.FilterCatalogs, name)
    params.AddCatalog(catalog)
    return FilterCatalog(params)


def _catalog_filter(smiles: str, catalog_name: str, result_key: str) -> dict:
    mol = smiles_to_mol(smiles)
    if mol is None:
        return {"valid": False}

    entry = _filter_catalog(catalog_name).GetFirstMatch(mol)

    return {
        "valid": True,
        result_key: entry is not None,
        f"{result_key}_description": entry.GetDescription() if entry else None,
        "passed": entry is None,
    }


def pains_filter(smiles: str) -> dict:
    return _catalog_filter(smiles, "PAINS", "pains")


def brenk_filter(smiles: str) -> dict:
    return _catalog_filter(smiles, "BRENK", "brenk")


def veber_filter(smiles: str) -> dict:
    mol = smiles_to_mol(smiles)
    if mol is None:
        return {"valid": False}

    rotatable_bonds = Lipinski.NumRotatableBonds(mol)
    tpsa = rdMolDescriptors.CalcTPSA(mol)
    violations = 0
    violations += rotatable_bonds > 10
    violations += tpsa > 140

    return {
        "valid": True,
        "rotatable_bonds": int(rotatable_bonds),
        "tpsa": float(tpsa),
        "violations": int(violations),
        "passed": violations == 0,
    }
