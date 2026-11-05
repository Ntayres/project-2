import numpy as np
import pandas as pd
import pytest

from src import config, models, preprocessing


def small_training_data():
    rng = np.random.RandomState(config.RANDOM_STATE)
    X = pd.DataFrame(
        rng.normal(size=(12, 4)),
        columns=["api_a", "api_b", "api_c", "api_d"],
    )
    X["api_a"] = np.arange(len(X))
    y = pd.Series([0, 1] * 6, name=config.LABEL_COLUMN)
    years = pd.Series(
        [2019] * 4 + [2020] * 4 + [2021] * 4,
        name=config.YEAR_COLUMN,
    )
    return X, y, years


def valid_folds():
    _, _, years = small_training_data()
    folds_plan = preprocessing.forward_chaining_folds([2019, 2020, 2021])
    cv = preprocessing.fold_indices(years, folds_plan)
    return years, folds_plan, cv


def test_model_specs_and_pipelines_use_expected_estimators_and_preprocessing():
    assert set(models.MODEL_SPECS) == {
        "DummyClassifier",
        "LogisticRegression",
        "SGDClassifier",
        "RandomForestClassifier",
    }

    for name in models.MODEL_SPECS:
        pipeline = models.make_model_pipeline(name)
        assert list(pipeline.named_steps)[:1] == ["drop_constant"]
        assert pipeline.named_steps["model"].__class__.__name__ == name
        assert pipeline.named_steps["model"].random_state == config.RANDOM_STATE

    assert "scale" in models.make_model_pipeline("LogisticRegression").named_steps
    assert "scale" in models.make_model_pipeline("SGDClassifier").named_steps
    assert "scale" not in models.make_model_pipeline("RandomForestClassifier").named_steps
    assert "scale" not in models.make_model_pipeline("DummyClassifier").named_steps


def test_check_folds_accepts_forward_chaining_plan():
    years, folds_plan, cv = valid_folds()

    assert models.check_folds(
        cv, years, [2019, 2020, 2021], folds=folds_plan,
        y_train=small_training_data()[1],
    )


@pytest.mark.parametrize(
    "bad_cv, message",
    [
        ([(np.array([-1]), np.array([4])), (np.array([0, 1]), np.array([8, 9]))],
         "outside"),
        ([(np.array([0, 0]), np.array([4, 5])), (np.array([0, 1]), np.array([8, 9]))],
         "repeats"),
        ([(np.array([0, 1]), np.array([1, 2])), (np.array([0, 1]), np.array([8, 9]))],
         "both"),
        ([(np.array([0, 1]), np.array([4, 5])), (np.array([0, 1]), np.array([6, 7]))],
         "not exactly"),
    ],
)
def test_check_folds_rejects_malformed_or_wrong_folds(bad_cv, message):
    years, folds_plan, _ = valid_folds()

    with pytest.raises(ValueError, match=message):
        models.check_folds(
            bad_cv,
            years,
            [2019, 2020, 2021],
            folds=folds_plan,
            y_train=small_training_data()[1],
        )


def test_check_folds_rejects_single_class_validation_year():
    years, folds_plan, cv = valid_folds()
    labels = small_training_data()[1].copy()
    labels.iloc[4:8] = 0

    with pytest.raises(ValueError, match="only class"):
        models.check_folds(
            cv, years, [2019, 2020, 2021],
            folds=folds_plan, y_train=labels,
        )


def test_reproducibility_check_compares_predictions_and_fitted_state():
    X, y, _ = small_training_data()
    result = models.reproducibility_check(
        models.make_model_pipeline("LogisticRegression"),
        X,
        y,
        n_fit_rows=8,
        n_eval_rows=4,
    )

    assert result["n_fit_rows"] == 8
    assert result["n_eval_rows"] == 4
    assert result["predictions_match"]
    assert result["state_match"]
    assert result["all_match"]

    with pytest.raises(ValueError, match="no rows left"):
        models.reproducibility_check(
            models.make_model_pipeline("LogisticRegression"),
            X,
            y,
            n_fit_rows=len(X),
            n_eval_rows=1,
        )


def test_small_dummy_search_produces_finite_scores_and_summary_row():
    X, y, years = small_training_data()
    _, folds_plan, cv = valid_folds()
    search = models.select_and_fit("DummyClassifier", X, y, cv)

    assert search.best_estimator_.predict(X).shape == (len(X),)
    unavailable = models.check_search_scores("DummyClassifier", search)
    assert unavailable == []

    reproducibility = models.reproducibility_check(
        search.best_estimator_, X, y, n_fit_rows=8, n_eval_rows=4
    )
    row = models.best_params_row(
        "DummyClassifier", search, 0.1, reproducibility, unavailable
    )
    assert row["model"] == "DummyClassifier"
    assert row["n_folds"] == len(folds_plan)
    assert np.isfinite(row["cv_accuracy_mean"])
    assert len(models.cv_results_table("DummyClassifier", search)) == 1


def test_model_and_cv_table_guards_reject_missing_or_inconsistent_rows():
    valid = pd.DataFrame([
        {
            "model": name,
            "cv_accuracy_mean": 0.5,
            "n_candidates": 1,
        }
        for name in models.MODEL_SPECS
    ])
    models.check_model_rows(valid)

    missing = valid.iloc[:-1]
    with pytest.raises(ValueError, match="expected exactly 1"):
        models.check_model_rows(missing)

    nonfinite = valid.copy()
    nonfinite.loc[0, "cv_accuracy_mean"] = np.nan
    with pytest.raises(ValueError, match="non-finite"):
        models.check_model_rows(nonfinite)


def test_class_balance_input_is_validated_before_writing_decision_note():
    _, _, years = small_training_data()
    split = preprocessing.Split(
        train=pd.DataFrame(),
        test=pd.DataFrame(),
        train_years=[2019, 2020],
        test_years=[2021],
    )
    summary = pd.DataFrame({
        "year": [2019, 2020, 2021],
        "split": ["train", "train", "test"],
        "malware_share": [0.5, 0.5, 0.5],
    })

    shares = models.check_class_balance_input(summary, split)
    assert shares.tolist() == [0.5, 0.5]
    assert "Not used" in models.class_balance_note(summary, split)

    incomplete = summary.iloc[[0, 2]]
    with pytest.raises(ValueError, match="training years"):
        models.check_class_balance_input(incomplete, split)

    invalid = summary.copy()
    invalid.loc[0, "malware_share"] = 1.5
    with pytest.raises(ValueError, match="within"):
        models.check_class_balance_input(invalid, split)
