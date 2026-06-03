from __future__ import annotations

import argparse
import json
import logging
import os
import random
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import joblib
import numpy as np
import pandas as pd
import yaml
from sklearn.model_selection import GroupKFold
from xgboost import XGBClassifier, XGBRegressor
from rdkit import RDLogger
try:
    import mlflow
except Exception:
    mlflow = None

try:
    import optuna
except Exception:
    optuna = None

from src.models.datasets import (
    TARGET_COL,
    SMILES_COL,
    load_tdc_adme_dataset,
    validate_dataset,
)
from src.models.featurizers import (
    MolecularFeatureExtractor,
    morgan_fingerprint_for_ad,
)
from src.models.metrics import (
    classification_metrics,
    regression_metrics,
)
from src.models.plots import (
    build_dataset_summary,
    collect_split_predictions,
    save_training_plots,
    write_dataset_summary,
)
from src.models.splitting import (
    add_scaffolds,
    assert_no_scaffold_leakage,
    scaffold_ordered_split,
    split_train_and_early_stop_by_scaffold,
)

RDLogger.DisableLog("rdApp.warning")
logger = logging.getLogger("centaurdrug.training")


def configure_logging(
    level: int = logging.INFO,
) -> None:
    """
    Configure logging only in CLI entrypoint.

    We do not call logging.basicConfig at import time.
    """

    logging.basicConfig(
        level=level,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )


def set_seed(
    seed: int,
) -> None:
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)


def load_config(
    path: str | Path,
) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def safe_fit_with_early_stopping(
    model,
    X_train,
    y_train,
    X_early,
    y_early,
    rounds: int = 50,
):
    """
    Fit XGBoost with early stopping.

    Modern XGBoost versions require early_stopping_rounds in the constructor,
    not in fit(). So we silently rebuild the model with early_stopping_rounds.
    """

    params = model.get_params()
    params["early_stopping_rounds"] = rounds

    model_with_es = type(model)(**params)

    model_with_es.fit(
        X_train,
        y_train,
        eval_set=[(X_early, y_early)],
        verbose=False,
    )

    return model_with_es


def classification_scale_pos_weight(
    y: np.ndarray,
) -> float:
    """
    Compute scale_pos_weight = negative_count / positive_count.

    This matters for imbalanced safety datasets like AMES and hERG.
    """

    negative_count = int(np.sum(y == 0))
    positive_count = int(np.sum(y == 1))

    if positive_count == 0:
        return 1.0

    return float(negative_count / positive_count)


def default_params() -> Dict[str, Any]:
    """
    Safe default XGBoost parameters.

    Used only if Optuna search is disabled.
    """

    return {
        "n_estimators": 800,
        "max_depth": 5,
        "learning_rate": 0.03,
        "subsample": 0.9,
        "colsample_bytree": 0.8,
        "min_child_weight": 3,
        "reg_alpha": 0.01,
        "reg_lambda": 2.0,
    }


def build_xgb_model(
    task_type: str,
    seed: int,
    params: Dict[str, Any],
    n_jobs: int,
    scale_pos_weight: Optional[float] = None,
):
    """
    Build either XGBRegressor or XGBClassifier.
    """

    common_params = {
        "random_state": seed,
        "n_jobs": n_jobs,
        "tree_method": "hist",
        **params,
    }

    if task_type == "regression":
        return XGBRegressor(
            objective="reg:squarederror",
            eval_metric="rmse",
            **common_params,
        )

    if task_type == "classification":
        classifier_params = {}

        if scale_pos_weight is not None:
            classifier_params["scale_pos_weight"] = scale_pos_weight

        return XGBClassifier(
            objective="binary:logistic",
            eval_metric="auc",
            **classifier_params,
            **common_params,
        )

    raise ValueError(f"Unsupported task_type: {task_type}")


def suggest_xgb_params(
    trial,
) -> Dict[str, Any]:
    """
    Optuna search space.
    """

    return {
        "n_estimators": trial.suggest_int(
            "n_estimators",
            300,
            1200,
            step=100,
        ),
        "max_depth": trial.suggest_int(
            "max_depth",
            3,
            8,
        ),
        "learning_rate": trial.suggest_float(
            "learning_rate",
            0.01,
            0.12,
            log=True,
        ),
        "subsample": trial.suggest_float(
            "subsample",
            0.65,
            1.0,
        ),
        "colsample_bytree": trial.suggest_float(
            "colsample_bytree",
            0.60,
            1.0,
        ),
        "min_child_weight": trial.suggest_int(
            "min_child_weight",
            1,
            10,
        ),
        "reg_alpha": trial.suggest_float(
            "reg_alpha",
            1e-8,
            1.0,
            log=True,
        ),
        "reg_lambda": trial.suggest_float(
            "reg_lambda",
            0.5,
            10.0,
            log=True,
        ),
    }


def minimum_valid_classification_cv_folds(
    cv_folds: int,
    hp_cfg: Dict[str, Any],
) -> int:
    """
    Minimum number of folds that must support AUROC/AUPRC scoring.

    Scaffold grouping can occasionally produce validation folds containing
    only one class. Those folds are skipped for threshold-independent
    classification metrics, but too many skipped folds makes tuning unreliable.
    """

    configured_value = hp_cfg.get("min_valid_classification_folds")

    if configured_value is not None:
        min_valid_folds = int(configured_value)
    else:
        # Default: require at least two usable folds, and for larger CV runs
        # require at least half of the configured folds to be usable.
        min_valid_folds = max(2, (cv_folds + 1) // 2)

    if not 1 <= min_valid_folds <= cv_folds:
        raise ValueError(
            "min_valid_classification_folds must be between 1 and "
            f"cv_folds={cv_folds}. Got {min_valid_folds}."
        )

    return min_valid_folds


def assert_enough_classification_cv_scores(
    fold_scores: list[float],
    cv_folds: int,
    skipped_folds: list[int],
    min_valid_folds: int,
) -> None:
    """
    Fail clearly when too few folds can compute classification AUROC/AUPRC.
    """

    if len(fold_scores) >= min_valid_folds:
        return

    raise RuntimeError(
        "Too few classification CV folds had both target classes in the "
        "validation fold for AUROC/AUPRC scoring. "
        f"Usable folds: {len(fold_scores)}/{cv_folds}. "
        f"Required usable folds: {min_valid_folds}. "
        f"Skipped folds: {skipped_folds}. "
        "Reduce cv_folds, use a grouped-stratified splitter, or provide a "
        "larger/more balanced scaffold training set."
    )


def prepare_features(
    featurizer: MolecularFeatureExtractor,
    df: pd.DataFrame,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Featurize dataframe and extract target.
    """

    X = featurizer.transform(df[SMILES_COL].tolist())
    y = df[TARGET_COL].to_numpy()

    return X, y


def tune_hyperparameters(
    train_df: pd.DataFrame,
    task_type: str,
    featurizer: MolecularFeatureExtractor,
    seed: int,
    xgboost_n_jobs: int,
    hp_cfg: Dict[str, Any],
) -> Dict[str, Any]:
    """
    Optuna tuning with scaffold-group cross-validation.

    Important:
    - CV uses GroupKFold by scaffold.
    - Early-stopping subset inside each fold is also scaffold-split.
    """

    if optuna is None:
        raise RuntimeError(
            "Optuna is enabled in config but not installed. "
            "Run: uv add optuna"
        )

    n_trials = int(hp_cfg.get("n_trials", 25))
    cv_folds = int(hp_cfg.get("cv_folds", 5))
    early_fraction = float(
        hp_cfg.get("early_stop_fraction_inside_cv", 0.15)
    )
    timeout = hp_cfg.get("timeout_seconds", None)

    if cv_folds < 2:
        raise ValueError(f"cv_folds must be at least 2. Got {cv_folds}.")

    X_all, y_all = prepare_features(featurizer, train_df)
    groups = train_df["scaffold"].to_numpy()

    n_scaffold_groups = len(np.unique(groups))

    if n_scaffold_groups < cv_folds:
        raise ValueError(
            f"cv_folds={cv_folds} cannot exceed the number of scaffold "
            f"groups in train_core ({n_scaffold_groups})."
        )

    if task_type == "classification":
        y_all = y_all.astype(int)
        min_valid_classification_folds = (
            minimum_valid_classification_cv_folds(
                cv_folds=cv_folds,
                hp_cfg=hp_cfg,
            )
        )
    else:
        min_valid_classification_folds = 0

    group_kfold = GroupKFold(n_splits=cv_folds)

    direction = "minimize" if task_type == "regression" else "maximize"

    def objective(trial):
        params = suggest_xgb_params(trial)
        fold_scores = []
        skipped_classification_folds = []

        for fold_idx, (train_idx, valid_idx) in enumerate(
            group_kfold.split(X_all, y_all, groups=groups),
            start=1,
        ):
            if task_type == "classification":
                fold_valid_classes = np.unique(y_all[valid_idx])

                if len(fold_valid_classes) < 2:
                    skipped_classification_folds.append(fold_idx)
                    logger.warning(
                        "Skipping Optuna trial=%s fold=%s for AUROC/AUPRC: "
                        "validation fold has one target class: %s",
                        trial.number,
                        fold_idx,
                        fold_valid_classes.tolist(),
                    )
                    continue

            fold_train_df = train_df.iloc[train_idx].reset_index(drop=True)
            fold_valid_df = train_df.iloc[valid_idx].reset_index(drop=True)

            inner_train_df, inner_early_df = (
                split_train_and_early_stop_by_scaffold(
                    fold_train_df,
                    early_stop_fraction=early_fraction,
                    seed=seed + fold_idx,
                )
            )

            X_inner_train, y_inner_train = prepare_features(
                featurizer,
                inner_train_df,
            )
            X_inner_early, y_inner_early = prepare_features(
                featurizer,
                inner_early_df,
            )
            X_fold_valid, y_fold_valid = prepare_features(
                featurizer,
                fold_valid_df,
            )

            if task_type == "classification":
                y_inner_train = y_inner_train.astype(int)
                y_inner_early = y_inner_early.astype(int)
                y_fold_valid = y_fold_valid.astype(int)

                spw = classification_scale_pos_weight(y_inner_train)
            else:
                spw = None

            model = build_xgb_model(
                task_type=task_type,
                seed=seed + fold_idx,
                params=params,
                n_jobs=xgboost_n_jobs,
                scale_pos_weight=spw,
            )

            model = safe_fit_with_early_stopping(
                model,
                X_inner_train,
                y_inner_train,
                X_inner_early,
                y_inner_early,
                rounds=50,
            )

            if task_type == "regression":
                pred = model.predict(X_fold_valid)
                score = regression_metrics(y_fold_valid, pred)["rmse"]

            else:
                proba = model.predict_proba(X_fold_valid)[:, 1]
                pred = (proba >= 0.5).astype(int)

                score = classification_metrics(
                    y_fold_valid,
                    pred,
                    proba,
                )["roc_auc"]

            fold_scores.append(score)

        if task_type == "classification":
            assert_enough_classification_cv_scores(
                fold_scores=fold_scores,
                cv_folds=cv_folds,
                skipped_folds=skipped_classification_folds,
                min_valid_folds=min_valid_classification_folds,
            )

        return float(np.mean(fold_scores))

    sampler = optuna.samplers.TPESampler(seed=seed)

    study = optuna.create_study(
        direction=direction,
        sampler=sampler,
    )

    study.optimize(
        objective,
        n_trials=n_trials,
        timeout=timeout,
    )

    logger.info("Best Optuna value: %.5f", study.best_value)
    logger.info("Best Optuna params: %s", study.best_params)

    return dict(study.best_params)


def evaluate_model(
    model,
    task_type: str,
    splits: Dict[str, pd.DataFrame],
    featurizer: MolecularFeatureExtractor,
) -> Dict[str, Dict[str, float]]:
    """
    Evaluate model on all splits.

    early_stop is reported for transparency,
    but it must not be used as the final model-selection metric.
    """

    metrics = {}

    for split_name in [
        "train_core",
        "early_stop",
        "validation",
        "test",
    ]:
        split_df = splits[split_name]
        X, y = prepare_features(featurizer, split_df)

        if task_type == "regression":
            pred = model.predict(X)
            metrics[split_name] = regression_metrics(y, pred)

        else:
            y = y.astype(int)

            proba = model.predict_proba(X)[:, 1]
            pred = (proba >= 0.5).astype(int)

            metrics[split_name] = classification_metrics(
                y,
                pred,
                proba,
            )

    return metrics


def save_training_fingerprints(
    train_df: pd.DataFrame,
    artifact_dir: Path,
    radius: int,
    n_bits: int,
    max_fps: int,
) -> None:
    """
    Save training fingerprints for applicability-domain Tanimoto checks.
    """

    df = train_df[[SMILES_COL, "scaffold"]].copy()

    if len(df) > max_fps:
        df = df.sample(
            n=max_fps,
            random_state=42,
        ).reset_index(drop=True)

    smiles_list = df[SMILES_COL].tolist()

    fps = [
        morgan_fingerprint_for_ad(
            smiles,
            radius=radius,
            n_bits=n_bits,
        )
        for smiles in smiles_list
    ]

    joblib.dump(
        smiles_list,
        artifact_dir / "training_smiles.joblib",
    )
    joblib.dump(
        fps,
        artifact_dir / "training_fps.joblib",
    )


def train_dataset(
    dataset_name: str,
    config: Dict[str, Any],
) -> Path:
    """
    Main production training function.
    """

    seed = int(config["training"]["random_seed"])
    set_seed(seed)
    xgboost_n_jobs = int(config["training"].get("xgboost_n_jobs", 1))

    if xgboost_n_jobs == 0:
        raise ValueError("training.xgboost_n_jobs cannot be 0.")

    if dataset_name not in config["datasets"]:
        raise ValueError(f"Dataset {dataset_name} is not configured.")

    dataset_cfg = config["datasets"][dataset_name]
    task_type = dataset_cfg["task_type"]

    artifact_root = Path(config["training"]["artifact_dir"])
    artifact_dir = artifact_root / dataset_name
    artifact_dir.mkdir(parents=True, exist_ok=True)

    logger.info("Training dataset=%s task_type=%s", dataset_name, task_type)

    raw_df = load_tdc_adme_dataset(dataset_name)

    valid_df, rejected_df = validate_dataset(raw_df)

    rejected_df.to_csv(
        artifact_dir / "rejected_molecules.csv",
        index=False,
    )

    valid_df = add_scaffolds(
        valid_df,
        smiles_col=SMILES_COL,
    )

    split_cfg = config["splitting"]

    splits = scaffold_ordered_split(
        valid_df,
        train_frac=float(split_cfg["train_frac"]),
        early_stop_frac=float(split_cfg["early_stop_frac"]),
        valid_frac=float(split_cfg["valid_frac"]),
        test_frac=float(split_cfg["test_frac"]),
        seed=seed,
    )

    assert_no_scaffold_leakage(splits)

    split_report = pd.DataFrame(
        {
            "split": list(splits.keys()),
            "n_molecules": [len(v) for v in splits.values()],
            "n_scaffolds": [
                v["scaffold"].nunique()
                for v in splits.values()
            ],
            "target_mean": [
                float(v[TARGET_COL].mean())
                for v in splits.values()
            ],
            "target_std": [
                float(v[TARGET_COL].std())
                for v in splits.values()
            ],
            "target_min": [
                float(v[TARGET_COL].min())
                for v in splits.values()
            ],
            "target_max": [
                float(v[TARGET_COL].max())
                for v in splits.values()
            ],
        }
    )

    split_report.to_csv(
        artifact_dir / "split_report.csv",
        index=False,
    )

    features_cfg = config["features"]

    featurizer = MolecularFeatureExtractor(
        radius=int(features_cfg["morgan_radius"]),
        n_bits=int(features_cfg["morgan_n_bits"]),
        use_maccs=bool(features_cfg["use_maccs"]),
        use_rdkit_descriptors=bool(
            features_cfg["use_rdkit_descriptors"]
        ),
    )

    hp_cfg = config.get("hyperparameter_search", {})

    if hp_cfg.get("enabled", False):
        params = tune_hyperparameters(
            train_df=splits["train_core"],
            task_type=task_type,
            featurizer=featurizer,
            seed=seed,
            xgboost_n_jobs=xgboost_n_jobs,
            hp_cfg=hp_cfg,
        )
    else:
        params = default_params()

    X_train, y_train = prepare_features(
        featurizer,
        splits["train_core"],
    )
    X_early, y_early = prepare_features(
        featurizer,
        splits["early_stop"],
    )

    if task_type == "classification":
        y_train = y_train.astype(int)
        y_early = y_early.astype(int)
        scale_pos_weight = classification_scale_pos_weight(y_train)
    else:
        scale_pos_weight = None

    model = build_xgb_model(
        task_type=task_type,
        seed=seed,
        params=params,
        n_jobs=xgboost_n_jobs,
        scale_pos_weight=scale_pos_weight,
    )

    model = safe_fit_with_early_stopping(
        model,
        X_train,
        y_train,
        X_early,
        y_early,
        rounds=50,
    )

    metrics = evaluate_model(
        model,
        task_type,
        splits,
        featurizer,
    )
    predictions = collect_split_predictions(
        model=model,
        task_type=task_type,
        splits=splits,
        featurizer=featurizer,
    )
    dataset_summary = build_dataset_summary(
        dataset_name=dataset_name,
        task_type=task_type,
        raw_df=raw_df,
        valid_df=valid_df,
        rejected_df=rejected_df,
        splits=splits,
    )

    logger.info("Metrics:\n%s", json.dumps(metrics, indent=2))

    model_path = artifact_dir / "model.joblib"
    featurizer_path = artifact_dir / "featurizer.joblib"
    metrics_path = artifact_dir / "metrics.json"
    metadata_path = artifact_dir / "metadata.json"
    predictions_path = artifact_dir / "predictions.csv"
    dataset_summary_path = artifact_dir / "dataset_summary.json"

    joblib.dump(model, model_path)
    joblib.dump(featurizer, featurizer_path)

    metrics_path.write_text(
        json.dumps(metrics, indent=2),
        encoding="utf-8",
    )
    predictions.to_csv(
        predictions_path,
        index=False,
    )
    write_dataset_summary(
        dataset_summary,
        dataset_summary_path,
    )
    plot_paths = save_training_plots(
        splits=splits,
        task_type=task_type,
        metrics=metrics,
        predictions=predictions,
        artifact_dir=artifact_dir,
    )

    if config.get("applicability_domain", {}).get("enabled", True):
        ad_cfg = config["applicability_domain"]

        save_training_fingerprints(
            train_df=splits["train_core"],
            artifact_dir=artifact_dir,
            radius=int(features_cfg["morgan_radius"]),
            n_bits=int(features_cfg["morgan_n_bits"]),
            max_fps=int(
                ad_cfg.get("max_training_fps_for_ad", 50000)
            ),
        )

    metadata = {
        "project": config.get("project", {}).get(
            "name",
            "centaurdrug",
        ),
        "dataset": dataset_name,
        "task_type": task_type,
        "primary_metric": dataset_cfg["primary_metric"],
        "prediction_unit": dataset_cfg.get("prediction_unit"),
        "positive_class_name": dataset_cfg.get("positive_class_name"),
        "model_type": type(model).__name__,
        "smiles_col": SMILES_COL,
        "target_col": TARGET_COL,
        "random_seed": seed,
        "xgboost_n_jobs": xgboost_n_jobs,
        "features": features_cfg,
        "preprocessing": config.get("preprocessing", {}),
        "splitting": split_cfg,
        "best_params": params,
        "scale_pos_weight": scale_pos_weight,
        "metrics": metrics,
        "artifacts": {
            "model": "model.joblib",
            "featurizer": "featurizer.joblib",
            "training_fps": "training_fps.joblib",
            "training_smiles": "training_smiles.joblib",
            "metrics": "metrics.json",
            "predictions": "predictions.csv",
            "dataset_summary": "dataset_summary.json",
            "split_report": "split_report.csv",
            "rejections": "rejected_molecules.csv",
            "plots": [
                path.relative_to(artifact_dir).as_posix()
                for path in plot_paths
            ],
        },
        "inference_contract": {
            "valid_status": "ok",
            "invalid_status": "rejected",
            "rejection_reasons": [
                "missing_smiles",
                "empty_smiles",
                "invalid_smiles",
            ],
        },
    }

    metadata_path.write_text(
        json.dumps(metadata, indent=2),
        encoding="utf-8",
    )

    if config["training"].get("use_mlflow", False):
        if mlflow is None:
            logger.warning(
                "MLflow enabled but not installed. Skipping MLflow logging."
            )

        else:
            mlflow.set_experiment(
                config["training"]["mlflow_experiment"]
            )

            with mlflow.start_run(
                run_name=f"{dataset_name}-xgboost",
            ):
                mlflow.log_param("dataset", dataset_name)
                mlflow.log_param("task_type", task_type)
                mlflow.log_param("model_type", type(model).__name__)
                mlflow.log_param("xgboost_n_jobs", xgboost_n_jobs)

                mlflow.log_params(
                    {
                        f"feature_{k}": v
                        for k, v in features_cfg.items()
                    }
                )

                mlflow.log_params(
                    {
                        f"xgb_{k}": v
                        for k, v in params.items()
                    }
                )

                if scale_pos_weight is not None:
                    mlflow.log_param(
                        "scale_pos_weight",
                        scale_pos_weight,
                    )

                for split_name, split_metrics in metrics.items():
                    for metric_name, value in split_metrics.items():
                        mlflow.log_metric(
                            f"{split_name}_{metric_name}",
                            value,
                        )

                for filename in [
                    "model.joblib",
                    "featurizer.joblib",
                    "metrics.json",
                    "metadata.json",
                    "predictions.csv",
                    "dataset_summary.json",
                    "split_report.csv",
                    "rejected_molecules.csv",
                    "training_fps.joblib",
                    "training_smiles.joblib",
                ]:
                    path = artifact_dir / filename

                    if path.exists():
                        mlflow.log_artifact(str(path))

                plots_dir = artifact_dir / "plots"

                if plots_dir.exists():
                    mlflow.log_artifacts(
                        str(plots_dir),
                        artifact_path="plots",
                    )

    logger.info("Training completed. Artifacts saved to %s", artifact_dir)

    return artifact_dir


def main() -> None:
    configure_logging()

    parser = argparse.ArgumentParser(
        description="Train CentaurDrug TDC ADMET XGBoost model."
    )

    parser.add_argument(
        "--config",
        default="configs/training.yaml",
    )

    parser.add_argument(
        "--dataset",
        required=True,
    )

    args = parser.parse_args()

    config = load_config(args.config)

    train_dataset(
        dataset_name=args.dataset,
        config=config,
    )


if __name__ == "__main__":
    main()
