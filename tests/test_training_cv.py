import pytest

from src.models.train_admet_xgboost import (
    assert_enough_classification_cv_scores,
    build_xgb_model,
    minimum_valid_classification_cv_folds,
)


def test_minimum_valid_classification_cv_folds_defaults_to_half_or_two():
    assert minimum_valid_classification_cv_folds(3, {}) == 2
    assert minimum_valid_classification_cv_folds(5, {}) == 3


def test_minimum_valid_classification_cv_folds_can_be_configured():
    hp_cfg = {"min_valid_classification_folds": 4}

    assert minimum_valid_classification_cv_folds(5, hp_cfg) == 4


def test_minimum_valid_classification_cv_folds_rejects_invalid_config():
    hp_cfg = {"min_valid_classification_folds": 6}

    with pytest.raises(ValueError, match="between 1 and cv_folds=5"):
        minimum_valid_classification_cv_folds(5, hp_cfg)


def test_assert_enough_classification_cv_scores_fails_clearly():
    with pytest.raises(RuntimeError, match="Usable folds: 1/3"):
        assert_enough_classification_cv_scores(
            fold_scores=[0.75],
            cv_folds=3,
            skipped_folds=[2, 3],
            min_valid_folds=2,
        )


def test_assert_enough_classification_cv_scores_allows_enough_folds():
    assert_enough_classification_cv_scores(
        fold_scores=[0.75, 0.80],
        cv_folds=3,
        skipped_folds=[2],
        min_valid_folds=2,
    )


def test_build_xgb_model_uses_configured_n_jobs():
    model = build_xgb_model(
        task_type="classification",
        seed=42,
        params={"n_estimators": 10},
        n_jobs=2,
        scale_pos_weight=1.5,
    )

    assert model.get_params()["n_jobs"] == 2
