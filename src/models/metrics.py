from __future__ import annotations

from typing import Dict

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    precision_score,
    r2_score,
    recall_score,
    roc_auc_score,
)


def regression_metrics(
    y_true,
    y_pred,
) -> Dict[str, float]:
    """
    Regression metrics.

    RMSE is computed manually to avoid sklearn version issues with squared=False.
    """

    mse = mean_squared_error(y_true, y_pred)

    return {
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "mse": float(mse),
        "rmse": float(np.sqrt(mse)),
        "r2": float(r2_score(y_true, y_pred)),
    }


def classification_metrics(
    y_true,
    y_pred,
    y_proba=None,
) -> Dict[str, float]:
    """
    Binary classification metrics.

    For ADMET toxicity models, AUROC and AUPRC are important.
    """

    metrics = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(
            precision_score(y_true, y_pred, zero_division=0)
        ),
        "recall": float(
            recall_score(y_true, y_pred, zero_division=0)
        ),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
    }

    if y_proba is not None and len(np.unique(y_true)) == 2:
        metrics["roc_auc"] = float(roc_auc_score(y_true, y_proba))
        metrics["auprc"] = float(
            average_precision_score(y_true, y_proba)
        )

    return metrics