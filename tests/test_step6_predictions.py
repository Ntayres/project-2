import joblib
import numpy as np
import pandas as pd
import pytest
from sklearn.dummy import DummyClassifier

from src import models, predictions, preprocessing


def fitted_model(feature_names):
    X = pd.DataFrame([[0, 0], [1, 0], [0, 1], [1, 1]], columns=feature_names)
    y = pd.Series([0, 0, 1, 1])
    model = DummyClassifier(strategy="most_frequent")
    model.fit(X, y)
    return model


def write_model_set(directory, feature_names=("api_a", "api_b")):
    directory.mkdir(parents=True, exist_ok=True)
    for name in models.MODEL_SPECS:
        joblib.dump(
            fitted_model(list(feature_names)),
            directory / f"{name}.joblib",
        )


def test_load_model_requires_fitted_predictor_and_matching_features(tmp_path):
    path = tmp_path / "model.joblib"
    model = fitted_model(["api_a", "api_b"])
    joblib.dump(model, path)

    loaded = predictions._load_model(path, ["api_a", "api_b"])
    assert loaded.predict(pd.DataFrame([[0, 1]], columns=["api_a", "api_b"])).shape == (1,)

    with pytest.raises(ValueError, match="feature names/order"):
        predictions._load_model(path, ["api_b", "api_a"])

    joblib.dump(object(), path)
    with pytest.raises(TypeError, match="does not provide predict"):
        predictions._load_model(path, ["api_a", "api_b"])


def test_predict_all_returns_one_shared_prediction_array_per_model(tmp_path):
    write_model_set(tmp_path)
    X = pd.DataFrame(
        [[0, 0], [1, 0], [0, 1], [1, 1]],
        columns=["api_a", "api_b"],
        index=[10, 11, 12, 13],
    )
    y = pd.Series([0, 0, 1, 1], index=X.index)
    years = pd.Series([2021, 2021, 2022, 2022], index=X.index)

    result, X_out, y_out, years_out = predictions._predict_all(tmp_path, X, y, years)

    assert set(result) == set(models.MODEL_SPECS)
    assert all(len(values) == 4 for values in result.values())
    assert X_out.index.tolist() == [0, 1, 2, 3]
    assert y_out.index.tolist() == [0, 1, 2, 3]
    assert years_out.tolist() == [2021, 2021, 2022, 2022]


@pytest.mark.parametrize(
    "X, y, years, message",
    [
        (
            pd.DataFrame([[0, 1]], columns=["api_a", "api_b"]),
            pd.Series([0, 1]),
            pd.Series([2021]),
            "equal length",
        ),
        (
            pd.DataFrame(columns=["api_a", "api_b"]),
            pd.Series(dtype=int),
            pd.Series(dtype=int),
            "cannot be empty",
        ),
        (
            pd.DataFrame([[0, 1]], columns=["api_a", "api_b"]),
            pd.Series([0]),
            pd.Series([np.nan]),
            "valid year",
        ),
    ],
)
def test_predict_all_rejects_invalid_test_inputs(tmp_path, X, y, years, message):
    write_model_set(tmp_path)
    with pytest.raises(ValueError, match=message):
        predictions._predict_all(tmp_path, X, y, years)


def test_metrics_from_predictions_has_overall_and_each_year_rows():
    y = pd.Series([0, 1, 0, 1])
    years = pd.Series([2021, 2021, 2022, 2022])
    predictions_by_model = {
        name: np.array([0, 1, 0, 0])
        for name in models.MODEL_SPECS
    }

    by_year, overall = predictions._metrics_from_predictions(
        predictions_by_model, y, years
    )

    assert len(by_year) == len(models.MODEL_SPECS) * 2
    assert len(overall) == len(models.MODEL_SPECS)
    assert set(by_year["year"]) == {2021, 2022}
    assert (by_year["n_test_rows"] == 2).all()
    assert (overall["n_test_rows"] == 4).all()
    assert by_year[evaluation_metrics()].apply(
        lambda column: column.between(0, 1).all()
    ).all()


def evaluation_metrics():
    return ["accuracy", "precision", "recall", "f1"]


def valid_step6_tables():
    rows = []
    overall = []
    for name in models.MODEL_SPECS:
        for year in (2021, 2022):
            rows.append({
                "classifier": name,
                "year": year,
                "n_test_rows": 2,
                "accuracy": 0.5 if year == 2021 else 0.25,
                "precision": 0.5,
                "recall": 0.5,
                "f1": 0.5,
            })
        overall.append({
            "classifier": name,
            "year": "all_test_years",
            "n_test_rows": 4,
            "accuracy": 0.375,
            "precision": 0.5,
            "recall": 0.5,
            "f1": 0.5,
        })
    return pd.DataFrame(rows), pd.DataFrame(overall)


def test_write_report_creates_drift_comparison_and_rejects_incomplete_tables(tmp_path):
    by_year, overall = valid_step6_tables()
    split = preprocessing.Split(
        train=pd.DataFrame(),
        test=pd.DataFrame(),
        train_years=[2019, 2020],
        test_years=[2021, 2022],
    )
    report_path = tmp_path / "step6_test_results.md"

    drift = predictions.write_report(by_year, overall, split, report_path)

    assert set(drift["classifier"]) == set(models.MODEL_SPECS)
    assert (drift["earliest_year"] == 2021).all()
    assert (drift["latest_year"] == 2022).all()
    assert "Step 6 Report: Test Data Prediction" in report_path.read_text(
        encoding="utf-8"
    )

    with pytest.raises(ValueError, match="every test year"):
        predictions.write_report(
            by_year[by_year["year"] != 2022],
            overall,
            split,
            tmp_path / "bad.md",
        )


def test_report_rejects_duplicate_classifier_year_and_nonfinite_metrics(tmp_path):
    by_year, overall = valid_step6_tables()
    split = preprocessing.Split(
        train=pd.DataFrame(),
        test=pd.DataFrame(),
        train_years=[2019, 2020],
        test_years=[2021, 2022],
    )

    duplicate = pd.concat([by_year, by_year.iloc[[0]]], ignore_index=True)
    with pytest.raises(ValueError, match="duplicate classifier/year"):
        predictions.write_report(duplicate, overall, split, tmp_path / "bad.md")

    invalid = by_year.copy()
    invalid.loc[0, "accuracy"] = np.nan
    with pytest.raises(ValueError, match="must be finite"):
        predictions.write_report(invalid, overall, split, tmp_path / "bad.md")
