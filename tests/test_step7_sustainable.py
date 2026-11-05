import json

import joblib
import numpy as np
import pandas as pd
import pytest
from sklearn.linear_model import SGDClassifier

from src import config, models, preprocessing, sustainable


def split_fixture():
    train = pd.DataFrame({
        config.YEAR_COLUMN: [2019, 2019, 2020, 2020],
        config.LABEL_COLUMN: [0, 1, 0, 1],
        "api_a": [0, 1, 0, 1],
        "api_b": [1, 0, 1, 0],
    })
    test = pd.DataFrame({
        config.YEAR_COLUMN: [2021, 2021, 2022, 2022],
        config.LABEL_COLUMN: [0, 1, 0, 1],
        "api_a": [0, 1, 1, 0],
        "api_b": [1, 0, 0, 1],
    })
    return preprocessing.Split(
        train=train,
        test=test,
        train_years=[2019, 2020],
        test_years=[2021, 2022],
    )


def valid_results():
    rows = []
    for strategy in (
        "frozen_baseline",
        "periodic_retraining",
        "incremental_partial_fit",
    ):
        for year in (2021, 2022):
            rows.append({
                "strategy": strategy,
                "classifier": sustainable.MODEL_NAME,
                "year": year,
                "n_test_rows": 2,
                "accuracy": 0.5,
                "precision": 0.5,
                "recall": 0.5,
                "f1": 0.5,
            })
    return pd.DataFrame(rows)


def test_selected_sgd_params_requires_one_valid_alpha_row(tmp_path):
    path = tmp_path / "step4_best_params.csv"
    pd.DataFrame([
        {"model": "SGDClassifier", "best_params": json.dumps({"alpha": 0.001})}
    ]).to_csv(path, index=False)
    assert sustainable._selected_sgd_params(tmp_path) == pytest.approx(0.001)

    pd.DataFrame([
        {"model": "LogisticRegression", "best_params": "{}"}
    ]).to_csv(path, index=False)
    with pytest.raises(ValueError, match="expected exactly one"):
        sustainable._selected_sgd_params(tmp_path)

    with pytest.raises(FileNotFoundError):
        sustainable._selected_sgd_params(tmp_path / "missing")


def test_score_predictions_uses_malware_positive_metrics():
    row = sustainable._score_predictions(
        "test_strategy",
        2021,
        np.array([0, 0, 1, 1]),
        np.array([0, 1, 1, 1]),
    )
    assert row["strategy"] == "test_strategy"
    assert row["classifier"] == "SGDClassifier"
    assert row["year"] == 2021
    assert row["n_test_rows"] == 4
    assert row["accuracy"] == pytest.approx(0.75)
    assert row["recall"] == pytest.approx(1.0)


def test_validate_results_requires_all_strategy_year_combinations():
    results = valid_results()
    sustainable._validate_results(results, [2021, 2022])

    with pytest.raises(ValueError, match="all sustainability strategies"):
        sustainable._validate_results(
            results[results["strategy"] != "incremental_partial_fit"],
            [2021, 2022],
        )

    duplicate = pd.concat([results, results.iloc[[0]]], ignore_index=True)
    with pytest.raises(ValueError, match="expected 6 rows, got 7"):
        sustainable._validate_results(duplicate, [2021, 2022])

    invalid = results.copy()
    invalid.loc[0, "accuracy"] = np.nan
    with pytest.raises(ValueError, match="non-finite"):
        sustainable._validate_results(invalid, [2021, 2022])


def test_frozen_predictions_validates_artifact_and_returns_one_row_per_year(tmp_path):
    X = pd.DataFrame(
        [[0, 1], [1, 0], [1, 1], [0, 0]],
        columns=["api_a", "api_b"],
    )
    y = np.array([0, 1, 1, 0])
    model = SGDClassifier(loss="log_loss", alpha=0.001, random_state=42)
    model.fit(X, y)
    joblib.dump(model, tmp_path / "SGDClassifier.joblib")

    rows = sustainable._frozen_predictions(
        tmp_path,
        X,
        y,
        np.array([2021, 2021, 2022, 2022]),
        0.001,
    )
    assert len(rows) == 2
    assert {row["year"] for row in rows} == {2021, 2022}
    assert all(row["strategy"] == "frozen_baseline" for row in rows)

    with pytest.raises(ValueError, match="alpha"):
        sustainable._frozen_predictions(
            tmp_path, X, y, np.array([2021, 2021, 2022, 2022]), 0.01
        )


def test_periodic_retraining_uses_only_years_before_each_test_year(monkeypatch):
    df = pd.DataFrame({
        config.YEAR_COLUMN: [2019, 2019, 2020, 2020, 2021, 2021],
        config.LABEL_COLUMN: [0, 1, 0, 1, 0, 1],
        "api_a": [0, 1, 0, 1, 1, 0],
        "api_b": [1, 0, 1, 0, 0, 1],
    })
    calls = []

    def fake_feature_frame(frame):
        calls.append(sorted(frame[config.YEAR_COLUMN].unique()))
        return (
            frame[["api_a", "api_b"]].reset_index(drop=True),
            frame[config.LABEL_COLUMN].reset_index(drop=True),
            frame[config.YEAR_COLUMN].reset_index(drop=True),
        )

    monkeypatch.setattr(sustainable, "_feature_frame", fake_feature_frame)
    monkeypatch.setattr(
        sustainable.preprocessing,
        "fit_preprocessor",
        lambda pipeline, X, train_years: pipeline.fit(X),
    )
    monkeypatch.setattr(
        sustainable.preprocessing,
        "preprocessor_for",
        lambda name: preprocessing.make_preprocessor(scale=True),
    )

    rows = sustainable._periodic_retraining(df, [2021], 0.001)
    assert len(rows) == 1
    assert calls == [[2019, 2020], [2021]]


def test_incremental_update_scores_before_updating_each_year(monkeypatch):
    split = split_fixture()
    calls = []

    def fake_feature_frame(frame):
        calls.append(sorted(frame[config.YEAR_COLUMN].unique()))
        return (
            frame[["api_a", "api_b"]].reset_index(drop=True),
            frame[config.LABEL_COLUMN].reset_index(drop=True),
            frame[config.YEAR_COLUMN].reset_index(drop=True),
        )

    monkeypatch.setattr(sustainable, "_feature_frame", fake_feature_frame)
    monkeypatch.setattr(
        sustainable.preprocessing,
        "fit_preprocessor",
        lambda pipeline, X, train_years: pipeline.fit(X),
    )
    monkeypatch.setattr(
        sustainable.preprocessing,
        "preprocessor_for",
        lambda name: preprocessing.make_preprocessor(scale=True),
    )

    rows = sustainable._incremental_update(split, 0.001)
    assert len(rows) == 2
    assert [row["year"] for row in rows] == [2021, 2022]
    assert calls[0] == [2019, 2020]
    assert calls[1:] == [[2021], [2022]]


def test_report_writes_strategy_summary_and_rejects_bad_results(tmp_path):
    split = split_fixture()
    path = tmp_path / "step7_sustainable_approach.md"
    sustainable.write_report(valid_results(), split, path, 0.001)
    text = path.read_text(encoding="utf-8")
    assert "# Step 7 Report: Sustainable ML Approach" in text
    assert "partial_fit" in text
    assert "alpha=0.001" in text

    with pytest.raises(ValueError, match="every test year"):
        sustainable.write_report(
            valid_results()[valid_results()["year"] != 2022],
            split,
            tmp_path / "bad.md",
            0.001,
        )
