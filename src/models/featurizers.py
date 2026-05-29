from __future__ import annotations

from typing import List, Optional

import numpy as np
from rdkit import Chem, DataStructs
from rdkit.Chem import Descriptors, MACCSkeys, rdMolDescriptors, rdFingerprintGenerator
from sklearn.base import BaseEstimator, TransformerMixin


RDKIT_DESCRIPTOR_FUNCS = [
    ("MolWt", Descriptors.MolWt),
    ("MolLogP", Descriptors.MolLogP),
    ("TPSA", Descriptors.TPSA),
    ("NumHDonors", Descriptors.NumHDonors),
    ("NumHAcceptors", Descriptors.NumHAcceptors),
    ("NumRotatableBonds", Descriptors.NumRotatableBonds),
    ("RingCount", Descriptors.RingCount),
    ("HeavyAtomCount", Descriptors.HeavyAtomCount),
    ("FractionCSP3", rdMolDescriptors.CalcFractionCSP3),
    ("NHOHCount", Descriptors.NHOHCount),
    ("NOCount", Descriptors.NOCount),
    ("NumAliphaticRings", rdMolDescriptors.CalcNumAliphaticRings),
    ("NumAromaticRings", rdMolDescriptors.CalcNumAromaticRings),
    ("NumSaturatedRings", rdMolDescriptors.CalcNumSaturatedRings),
]


class MolecularFeatureExtractor(BaseEstimator, TransformerMixin):
    """
    Convert SMILES to molecular features.

    Feature vector:
    [
        Morgan fingerprint,
        optional MACCS keys,
        optional RDKit descriptors
    ]

    Uses RDKit's modern MorganGenerator API.
    """

    def __init__(
        self,
        radius: int = 2,
        n_bits: int = 2048,
        use_maccs: bool = True,
        use_rdkit_descriptors: bool = True,
    ):
        self.radius = radius
        self.n_bits = n_bits
        self.use_maccs = use_maccs
        self.use_rdkit_descriptors = use_rdkit_descriptors

        self.descriptor_names_ = [
            name for name, _ in RDKIT_DESCRIPTOR_FUNCS
        ]

    def _get_morgan_generator(self):
        return rdFingerprintGenerator.GetMorganGenerator(
            radius=self.radius,
            fpSize=self.n_bits,
        )

    def fit(
        self,
        smiles: List[str],
        y: Optional[np.ndarray] = None,
    ):
        return self

    def transform(
        self,
        smiles: List[str],
    ) -> np.ndarray:
        features = []

        morgan_generator = self._get_morgan_generator()

        for s in smiles:
            mol = Chem.MolFromSmiles(str(s))

            if mol is None:
                raise ValueError(
                    f"Invalid SMILES passed to featurizer: {s}"
                )

            parts = []

            # Morgan fingerprint with new RDKit API
            morgan_fp = morgan_generator.GetFingerprint(mol)

            morgan_arr = np.zeros((self.n_bits,), dtype=np.float32)
            DataStructs.ConvertToNumpyArray(morgan_fp, morgan_arr)
            parts.append(morgan_arr)

            # MACCS keys
            if self.use_maccs:
                maccs_fp = MACCSkeys.GenMACCSKeys(mol)

                maccs_arr = np.zeros(
                    (maccs_fp.GetNumBits(),),
                    dtype=np.float32,
                )

                DataStructs.ConvertToNumpyArray(maccs_fp, maccs_arr)
                parts.append(maccs_arr)

            # RDKit descriptors
            if self.use_rdkit_descriptors:
                desc_values = []

                for _, func in RDKIT_DESCRIPTOR_FUNCS:
                    try:
                        value = float(func(mol))
                    except Exception:
                        value = np.nan

                    desc_values.append(value)

                desc_arr = np.asarray(desc_values, dtype=np.float32)
                parts.append(desc_arr)

            full_vector = np.concatenate(parts)
            features.append(full_vector)

        X = np.vstack(features).astype(np.float32)

        return np.nan_to_num(
            X,
            nan=0.0,
            posinf=0.0,
            neginf=0.0,
        )


def morgan_fingerprint_for_ad(
    smiles: str,
    radius: int = 2,
    n_bits: int = 2048,
):
    """
    RDKit ExplicitBitVect for applicability-domain Tanimoto checks.

    Uses MorganGenerator instead of deprecated GetMorganFingerprintAsBitVect.
    """

    mol = Chem.MolFromSmiles(str(smiles))

    if mol is None:
        raise ValueError(f"Invalid SMILES for AD fingerprint: {smiles}")

    generator = rdFingerprintGenerator.GetMorganGenerator(
        radius=radius,
        fpSize=n_bits,
    )

    return generator.GetFingerprint(mol)