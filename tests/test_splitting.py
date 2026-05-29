import pandas as pd

from src.models.splitting import (
    add_scaffolds,
    assert_no_scaffold_leakage,
    scaffold_ordered_split,
)


def test_scaffold_split_no_leakage():
    df = pd.DataFrame(
        {
            "Drug": [
                "c1ccccc1",
                "Cc1ccccc1",
                "Oc1ccccc1",
                "CCO",
                "CCCO",
                "CCN",
                "c1ccncc1",
                "Cc1ccncc1",
                "C1CCCCC1",
                "CC1CCCCC1",
            ],
            "Y": list(range(10)),
        }
    )

    df = add_scaffolds(df)

    splits = scaffold_ordered_split(
        df,
        train_frac=0.5,
        early_stop_frac=0.2,
        valid_frac=0.1,
        test_frac=0.2,
        seed=42,
    )

    assert_no_scaffold_leakage(splits)