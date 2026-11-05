"""Step 6: predict on held-out data and compare performance across test years."""
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.utils.validation import check_is_fitted

from src import config, evaluation, models, preprocessing
from src.data_loading import build_dataset


def _load_model(path, feature_columns):
    """Load and validate one fitted Step 4 pipeline for the current feature contract."""
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"Step 4 model artifact not found: {path}")
    estimator = joblib.load(path)
    if not callable(getattr(estimator, "predict", None)):
        raise TypeError(f"{path}: loaded artifact does not provide predict()")
    check_is_fitted(estimator)
    model_columns = getattr(estimator, "feature_names_in_", None)
    if model_columns is not None and list(model_columns) != list(feature_columns):
        raise ValueError(f"{path}: model feature names/order do not match the test contract")
    if model_columns is None:
        n_features = getattr(estimator, "n_features_in_", None)
        if n_features is None or int(n_features) != len(feature_columns):
            raise ValueError(
                f"{path}: model does not expose the expected feature count "
                f"({len(feature_columns)})"
            )
    return estimator


def _predict_all(models_dir, X_test, y_test, years_test):
    """Load each model once and return shared predictions plus validated inputs."""
    models_dir = Path(models_dir)
    years_test = pd.Series(years_test).reset_index(drop=True)
    y_test = pd.Series(y_test).reset_index(drop=True)
    X_test = X_test.reset_index(drop=True)
    if not (len(X_test) == len(y_test) == len(years_test)):
        raise ValueError("predict_by_year: features, labels, and years must have equal length")
    if len(X_test) == 0:
        raise ValueError("predict_by_year: test data cannot be empty")
    if years_test.empty or years_test.isna().any():
        raise ValueError("predict_by_year: test data must contain at least one valid year")

    predictions_by_model = {}
    for name in models.MODEL_SPECS:
        path = models_dir / f"{name}.joblib"
        estimator = _load_model(path, X_test.columns)
        predictions = estimator.predict(X_test)
        if len(predictions) != len(y_test):
            raise ValueError(
                f"{path}: produced {len(predictions)} predictions for {len(y_test)} rows"
            )
        predictions_by_model[name] = np.asarray(predictions)
    return predictions_by_model, X_test, y_test, years_test


def _metrics_from_predictions(predictions_by_model, y_test, years_test):
    """Build overall and per-year metrics from one shared prediction pass."""
    rows = []
    overall_rows = []
    for name in models.MODEL_SPECS:
        predictions = predictions_by_model[name]
        overall_scores = evaluation.classification_metrics(y_test, predictions)
        overall_rows.append({
            "classifier": name,
            "year": "all_test_years",
            "n_test_rows": len(y_test),
            **overall_scores,
        })
        for year in sorted(years_test.unique()):
            mask = years_test == year
            scores = evaluation.classification_metrics(
                y_test[mask],
                np.asarray(predictions)[mask.to_numpy()],
            )
            rows.append({
                "classifier": name,
                "year": int(year),
                "n_test_rows": int(mask.sum()),
                **scores,
            })

    result = pd.DataFrame(rows)
    expected_rows = len(models.MODEL_SPECS) * years_test.nunique()
    if len(result) != expected_rows:
        raise ValueError(
            f"predict_by_year: produced {len(result)} rows, expected {expected_rows}"
        )
    if not np.isfinite(result[list(evaluation.METRICS)].to_numpy()).all():
        raise ValueError("predict_by_year: non-finite metric value")
    overall = pd.DataFrame(overall_rows)
    if len(overall) != len(models.MODEL_SPECS):
        raise ValueError("predict_by_year: incomplete overall classifier coverage")
    return (
        result.sort_values(["year", "accuracy", "classifier"],
                           ascending=[True, False, True]).reset_index(drop=True),
        overall.sort_values("accuracy", ascending=False, kind="stable").reset_index(drop=True),
    )


def write_report(by_year, overall, split, path):
    """Write the Step 6 per-year prediction and drift analysis report."""
    required = {"classifier", "year", "n_test_rows", *evaluation.METRICS}
    if required - set(by_year.columns):
        raise ValueError("write_report: per-year results are missing required columns")
    expected_classifiers = set(models.MODEL_SPECS)
    expected_years = set(split.test_years)
    if set(by_year["classifier"]) != expected_classifiers:
        raise ValueError("write_report: per-year results do not cover every classifier")
    if set(by_year["year"]) != expected_years:
        raise ValueError("write_report: per-year results do not cover every test year")
    if by_year.duplicated(["classifier", "year"]).any():
        raise ValueError("write_report: duplicate classifier/year results")
    expected_combinations = len(expected_classifiers) * len(expected_years)
    if len(by_year) != expected_combinations:
        raise ValueError(
            f"write_report: expected {expected_combinations} classifier/year rows, "
            f"got {len(by_year)}"
        )
    overall_required = {"classifier", "year", "n_test_rows", *evaluation.METRICS}
    if overall_required - set(overall.columns):
        raise ValueError("write_report: overall results are missing required columns")
    if set(overall["classifier"]) != expected_classifiers or overall["classifier"].duplicated().any():
        raise ValueError("write_report: overall results do not cover each classifier exactly once")
    if not np.isfinite(by_year[list(evaluation.METRICS)].to_numpy()).all():
        raise ValueError("write_report: per-year metrics must be finite")
    if not np.isfinite(overall[list(evaluation.METRICS)].to_numpy()).all():
        raise ValueError("write_report: overall metrics must be finite")
    by_year = by_year.sort_values(
        ["year", "accuracy", "classifier"], ascending=[True, False, True]
    ).reset_index(drop=True)
    overall = overall.sort_values("accuracy", ascending=False, kind="stable")
    earliest = by_year["year"].min()
    latest = by_year["year"].max()
    first = by_year[by_year["year"] == earliest].set_index("classifier")
    last = by_year[by_year["year"] == latest].set_index("classifier")
    drift = pd.DataFrame({
        "classifier": sorted(models.MODEL_SPECS),
        "earliest_year": earliest,
        "latest_year": latest,
        "accuracy_change": [
            last.loc[name, "accuracy"] - first.loc[name, "accuracy"]
            for name in sorted(models.MODEL_SPECS)
        ],
        "precision_change": [
            last.loc[name, "precision"] - first.loc[name, "precision"]
            for name in sorted(models.MODEL_SPECS)
        ],
        "recall_change": [
            last.loc[name, "recall"] - first.loc[name, "recall"]
            for name in sorted(models.MODEL_SPECS)
        ],
        "f1_change": [
            last.loc[name, "f1"] - first.loc[name, "f1"]
            for name in sorted(models.MODEL_SPECS)
        ],
    })

    def format_table(frame):
        shown = frame.copy()
        for metric in evaluation.METRICS:
            shown[metric] = shown[metric].map(lambda value: f"{value:.4f}")
        return shown.to_markdown(index=False)

    lines = [
        "# Step 6 Report: Test Data Prediction",
        "",
        "Written by the builder agent. Generated by `python -m src.main --step 6`.",
        "",
        f"Models were trained on years {', '.join(map(str, split.train_years))} and predicted "
        f"on held-out years {', '.join(map(str, split.test_years))}. No model was fitted, "
        "updated, or tuned during this step.",
        "",
        "## Overall held-out prediction results",
        "",
        format_table(overall),
        "",
        "## Performance by test year",
        "",
        format_table(by_year),
        "",
        "## Drifted-data comparison",
        "",
        f"The change columns below compare the earliest test year ({earliest}) with the latest "
        f"test year ({latest}). A negative value indicates performance decreased as the data "
        "moved farther from the training period; a positive value indicates improvement.",
        "",
        drift.to_markdown(index=False, floatfmt=".4f"),
        "",
        "The per-year table separates ordinary classifier differences from changes associated "
        "with time. The drift comparison is descriptive: it does not prove that feature drift "
        "alone caused a performance change, because class composition and concept drift may "
        "also contribute.",
        "",
        "Machine-readable outputs:",
        "",
        "- `results/step6_test_predictions_by_year.csv`",
        "- `results/step6_test_prediction_overall.csv`",
        "- `results/step6_drift_comparison.csv`",
    ]
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    report = "\n".join(lines) + "\n"
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(report, encoding="utf-8")
    return drift


def run(input_dir=None, models_dir=None, results_dir=None, docs_dir=None):
    """Run Step 6 using Step 3's split and Step 4's fitted model artifacts."""
    results_path = Path(results_dir or config.RESULTS_DIR)
    docs_path = Path(docs_dir or config.DOCS_DIR)
    models_path = Path(models_dir or config.OUTPUTS_DIR / "models")
    results_path.mkdir(parents=True, exist_ok=True)
    docs_path.mkdir(parents=True, exist_ok=True)
    if not models_path.is_dir():
        raise FileNotFoundError(f"Step 4 model directory not found: {models_path}")

    df = build_dataset(input_dir)
    split = preprocessing.split_by_year(df)
    X_train, y_train, years_train = preprocessing.features_labels(split.train)
    X_test, y_test, years_test = preprocessing.features_labels(
        split.test, expected_columns=X_train.columns
    )
    preprocessing.check_feature_contract(X_train, X_test)
    preprocessing.check_labels_years(split, y_train, years_train, y_test, years_test)

    predictions_by_model, X_test, y_test, years_test = _predict_all(
        models_path, X_test, y_test, years_test
    )
    by_year, overall = _metrics_from_predictions(
        predictions_by_model, y_test, years_test
    )
    drift_path = results_path / "step6_drift_comparison.csv"
    by_year_path = results_path / "step6_test_predictions_by_year.csv"
    overall_path = results_path / "step6_test_prediction_overall.csv"
    by_year.to_csv(by_year_path, index=False)
    overall.to_csv(overall_path, index=False)
    drift = write_report(
        by_year,
        overall,
        split,
        docs_path / "step6_test_results.md",
    )
    drift.to_csv(drift_path, index=False)
    print(by_year.to_string(index=False))
    return by_year
