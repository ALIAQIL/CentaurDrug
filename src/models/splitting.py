from __future__ import annotations

import random
from typing import Dict, Tuple

import pandas as pd
from rdkit import Chem
from rdkit.Chem.Scaffolds import MurckoScaffold


import logging

logger = logging.getLogger("centaurdrug.splitting")


def compute_scaffold(smiles: str) -> str:
    """
    Compute Bemis-Murcko scaffold with robust fallback.

    Some molecules are parseable but fail Murcko scaffold generation
    because of problematic stereochemistry.
    """

    mol = Chem.MolFromSmiles(smiles)

    if mol is None:
        return "invalid"

    try:
        scaffold = MurckoScaffold.MurckoScaffoldSmiles(
            mol=mol,
            includeChirality=False,
        )

        if scaffold:
            return scaffold

    except Exception as exc:
        logger.warning(
            "Murcko scaffold failed for SMILES=%s | error=%s",
            smiles,
            exc,
        )

    try:
        canonical = Chem.MolToSmiles(
            mol,
            canonical=True,
            isomericSmiles=False,
        )
        return f"fallback::{canonical}"

    except Exception as exc:
        logger.warning(
            "Canonical fallback failed for SMILES=%s | error=%s",
            smiles,
            exc,
        )
        return f"fallback::{smiles}"


def add_scaffolds(
    df: pd.DataFrame,
    smiles_col: str = "Drug",
) -> pd.DataFrame:
    df = df.copy()
    df["scaffold"] = df[smiles_col].apply(compute_scaffold)
    return df


def scaffold_ordered_split(
    df: pd.DataFrame,
    train_frac: float,
    early_stop_frac: float,
    valid_frac: float,
    test_frac: float,
    seed: int,
) -> Dict[str, pd.DataFrame]:
    """
    Scaffold-grouped deterministic split.

    Guarantees:
    - all molecules sharing a scaffold stay in the same split
    - no empty split if there are at least 4 scaffold groups
    - approximately respects requested fractions
    """

    total = train_frac + early_stop_frac + valid_frac + test_frac

    if abs(total - 1.0) > 1e-8:
        raise ValueError("Split fractions must sum to 1.0")

    if "scaffold" not in df.columns:
        raise ValueError("Dataframe must contain a scaffold column.")

    split_names = ["train_core", "early_stop", "validation", "test"]

    groups = [
        group.reset_index(drop=True)
        for _, group in df.groupby("scaffold", sort=False)
    ]

    if len(groups) < len(split_names):
        raise RuntimeError(
            f"Cannot create {len(split_names)} non-empty scaffold splits "
            f"from only {len(groups)} scaffold groups."
        )

    rng = random.Random(seed)
    rng.shuffle(groups)

    # Sort large scaffold groups first, while keeping seed-based randomness
    # for groups with similar sizes.
    groups = sorted(
        groups,
        key=lambda g: len(g),
        reverse=True,
    )

    n_total = len(df)

    target_counts = {
        "train_core": train_frac * n_total,
        "early_stop": early_stop_frac * n_total,
        "validation": valid_frac * n_total,
        "test": test_frac * n_total,
    }

    buckets = {name: [] for name in split_names}
    counts = {name: 0 for name in split_names}

    # First pass: guarantee every split receives at least one scaffold group.
    # Assign the first scaffold groups to the currently most underfilled splits.
    for group, split_name in zip(groups[: len(split_names)], split_names):
        buckets[split_name].append(group)
        counts[split_name] += len(group)

    # Second pass: assign remaining scaffold groups to the split
    # with largest remaining capacity.
    for group in groups[len(split_names) :]:
        remaining_capacity = {
            name: target_counts[name] - counts[name]
            for name in split_names
        }

        best_split = max(remaining_capacity, key=remaining_capacity.get)

        buckets[best_split].append(group)
        counts[best_split] += len(group)

    splits = {}

    for name in split_names:
        parts = buckets[name]

        if not parts:
            raise RuntimeError(f"Empty split generated: {name}")

        split_df = pd.concat(parts)
        split_df = split_df.sample(frac=1.0, random_state=seed)
        splits[name] = split_df.reset_index(drop=True)

    return splits

def assert_no_scaffold_leakage(
    splits: Dict[str, pd.DataFrame],
) -> None:
    """
    Hard safety check.

    If the same scaffold appears in two different splits,
    the evaluation is contaminated.
    """

    names = list(splits.keys())

    for i, left in enumerate(names):
        for right in names[i + 1:]:
            left_scaffolds = set(splits[left]["scaffold"])
            right_scaffolds = set(splits[right]["scaffold"])

            overlap = left_scaffolds.intersection(right_scaffolds)

            if overlap:
                examples = list(overlap)[:5]

                raise AssertionError(
                    f"Scaffold leakage between {left} and {right}: "
                    f"{len(overlap)} overlapping scaffolds. "
                    f"Examples: {examples}"
                )


def split_train_and_early_stop_by_scaffold(
    df: pd.DataFrame,
    early_stop_fraction: float,
    seed: int,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Split a training fold into inner-train and early-stop subsets by scaffold.

    This is used inside Optuna CV.
    """

    if not 0.0 < early_stop_fraction < 0.5:
        raise ValueError("early_stop_fraction must be in (0, 0.5).")

    groups = list(df.groupby("scaffold", sort=False))

    rng = random.Random(seed)
    rng.shuffle(groups)

    n_total = len(df)
    target_early = early_stop_fraction * n_total

    early_parts = []
    train_parts = []
    early_count = 0

    for _, group in groups:
        if early_count < target_early:
            early_parts.append(group)
            early_count += len(group)
        else:
            train_parts.append(group)

    if not early_parts or not train_parts:
        raise RuntimeError(
            "Could not create non-empty inner train / early-stop split."
        )

    train_df = pd.concat(train_parts)
    train_df = train_df.sample(frac=1.0, random_state=seed)
    train_df = train_df.reset_index(drop=True)

    early_df = pd.concat(early_parts)
    early_df = early_df.sample(frac=1.0, random_state=seed)
    early_df = early_df.reset_index(drop=True)

    return train_df, early_df