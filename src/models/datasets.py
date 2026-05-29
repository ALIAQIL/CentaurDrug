from __future__ import annotations

from dataclasses import asdict
from typing import Tuple

import pandas as pd
from rdkit import RDLogger
from tdc.single_pred import ADME, Tox

from src.models.validation import validate_smiles


RDLogger.DisableLog("rdApp.*")

SMILES_COL = "Drug"
TARGET_COL = "Y"


ADME_DATASETS = {
    "Solubility_AqSolDB",
    "Lipophilicity_AstraZeneca",
    "Caco2_Wang",
    "HIA_Hou",
    "Pgp_Broccatelli",
    "Bioavailability_Ma",
    "BBB_Martins",
    "PPBR_AZ",
    "VDss_Lombardo",
    "CYP2C9_Veith",
    "CYP2D6_Veith",
    "CYP3A4_Veith",
    "CYP1A2_Veith",
    "CYP2C19_Veith",
    "CYP2C9_Substrate_CarbonMangels",
    "CYP2D6_Substrate_CarbonMangels",
    "CYP3A4_Substrate_CarbonMangels",
    "Half_Life_Obach",
    "Clearance_Hepatocyte_AZ",
    "Clearance_Microsome_AZ",
}


TOX_DATASETS = {
    "AMES",
    "hERG",
    "DILI",
    "LD50_Zhu",
}


def load_tdc_dataset(name: str) -> pd.DataFrame:
    """
    Load a TDC single-prediction molecular dataset.

    ADME datasets are loaded with tdc.single_pred.ADME.
    Toxicity datasets are loaded with tdc.single_pred.Tox.
    """

    if name in ADME_DATASETS:
        dataset = ADME(name=name)

    elif name in TOX_DATASETS:
        dataset = Tox(name=name)

    else:
        raise ValueError(
            f"Unknown or unsupported TDC dataset: {name}. "
            f"Known ADME datasets: {sorted(ADME_DATASETS)}. "
            f"Known Tox datasets: {sorted(TOX_DATASETS)}."
        )

    df = dataset.get_data()

    missing = {SMILES_COL, TARGET_COL}.difference(df.columns)

    if missing:
        raise ValueError(
            f"TDC dataset {name} is missing required columns: {missing}. "
            f"Available columns: {list(df.columns)}"
        )

    return df[[SMILES_COL, TARGET_COL]].copy()


def load_tdc_adme_dataset(name: str) -> pd.DataFrame:
    """
    Backward-compatible alias.

    The old code used this function name, but now it can load both ADME and Tox.
    """

    return load_tdc_dataset(name)


def validate_dataset(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Validate SMILES and split into valid and rejected rows.

    Returns:
    - valid_df: canonicalized SMILES, usable for training
    - rejected_df: invalid rows with rejection reasons
    """

    validation_series = df[SMILES_COL].apply(validate_smiles)

    validation_df = pd.DataFrame(
        [asdict(result) for result in validation_series],
        index=df.index,
    )

    merged = pd.concat([df.copy(), validation_df], axis=1)

    valid_df = merged[merged["is_valid"]].copy()
    rejected_df = merged[~merged["is_valid"]].copy()

    valid_df[SMILES_COL] = valid_df["canonical_smiles"]

    return valid_df.reset_index(drop=True), rejected_df.reset_index(drop=True)