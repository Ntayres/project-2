"""Step 7: compare frozen, periodically retrained, and incrementally updated models."""
from pathlib import Path

import joblib
import json
import numpy as np
import pandas as pd
from sklearn.linear_model import SGDClassifier
from sklearn.utils.validation import check_is_fitted

from src import config, evaluation, preprocessing
from src.data_loading import build_dataset


MODEL_NAME = "SGDClassifier"
def _feature_frame(frame):
    X, y, years = preprocessing.features_labels(frame)
    return X, y.reset_index(drop=True), years.reset_index(drop=True)


def _sgd(alpha):
    return SGDClassifier(
        loss="log_loss",
        penalty="l2",
        alpha=alpha,
        max_iter=1000,
        tol=1e-3,
        random_state=config.RANDOM_STATE,
    )


def _selected_sgd_params(results_dir):
    """Read the Step 4-selected SGD alpha so all strategies use one configuration."""
    path = Path(results_dir) / "step4_best_params.csv"
    if not path.is_file():
        raise FileNotFoundError(f"Step 4 parameter results not found: {path}")
    table = pd.read_csv(path)
    rows = table[table["model"] == MODEL_NAME]
    if len(rows) != 1:
        raise ValueError(f"{path}: expected exactly one {MODEL_NAME} row")
    params = json.loads(rows.iloc[0]["best_params"])
    if "alpha" not in params:
        raise ValueError(f"{path}: selected {MODEL_NAME} parameters do not contain alpha")
    return float(params["alpha"])


def _score_predictions(strategy, year, y_true, predictions):
    scores = evaluation.classification_metrics(y_true, predictions)
    return {
        "strategy": strategy,
        "classifier": MODEL_NAME,
        "year": int(year),
        "n_test_rows": len(y_true),
        **scores,
    }


def _frozen_predictions(models_dir, X_test, y_test, years_test, alpha):
    path = Path(models_dir) / f"{MODEL_NAME}.joblib"
    if not path.is_file():
        raise FileNotFoundError(f"Step 4 model artifact not found: {path}")
    estimator = joblib.load(path)
    check_is_fitted(estimator)
    model_columns = getattr(estimator, "feature_names_in_", None)
    if model_columns is not None and list(model_columns) != list(X_test.columns):
        raise ValueError(f"{path}: model feature names/order do not match test features")
    if model_columns is None:
        n_features = getattr(estimator, "n_features_in_", None)
        if n_features is None or int(n_features) != X_test.shape[1]:
            raise ValueError(f"{path}: model does not expose {X_test.shape[1]} input features")
    fitted_model = getattr(estimator, "named_steps", {}).get("model", estimator)
    if not np.isclose(float(getattr(fitted_model, "alpha", alpha)), alpha):
        raise ValueError(f"{path}: frozen model alpha does not match Step 4 alpha {alpha}")
    years_test = np.asarray(years_test)
    if len(years_test) == 0 or np.isnan(years_test.astype(float)).any():
        raise ValueError("frozen baseline requires nonempty, valid test years")
    predictions = estimator.predict(X_test)
    if len(predictions) != len(y_test):
        raise ValueError("frozen baseline produced the wrong number of predictions")
    return [
        _score_predictions("frozen_baseline", year, y_test[years_test == year],
                           predictions[years_test == year])
        for year in sorted(np.unique(years_test))
    ]


def _periodic_retraining(df, test_years, alpha):
    """Fit a new model on all data available before each test year."""
    rows = []
    for year in test_years:
        prior = df[df[config.YEAR_COLUMN] < year]
        current = df[df[config.YEAR_COLUMN] == year]
        X_prior, y_prior, _ = _feature_frame(prior)
        X_current, y_current, _ = _feature_frame(current)
        preprocessing.check_feature_contract(X_prior, X_current)
        preprocessor = preprocessing.fit_preprocessor(
            preprocessing.preprocessor_for(MODEL_NAME),
            X_prior,
            sorted(prior[config.YEAR_COLUMN].unique()),
        )
        model = _sgd(alpha)
        model.fit(preprocessor.transform(X_prior), y_prior)
        predictions = model.predict(preprocessor.transform(X_current))
        if len(predictions) != len(y_current):
            raise ValueError(f"periodic retraining for {year} returned the wrong prediction count")
        rows.append(_score_predictions("periodic_retraining", year, y_current, predictions))
    return rows


def _incremental_update(split, alpha):
    """Predict each year, then update SGD with that year after scoring it."""
    X_train, y_train, _ = _feature_frame(split.train)
    preprocessor = preprocessing.fit_preprocessor(
        preprocessing.preprocessor_for(MODEL_NAME),
        X_train,
        split.train_years,
    )
    model = _sgd(alpha)
    model.partial_fit(
        preprocessor.transform(X_train),
        y_train,
        classes=np.array(sorted(config.LABEL_MAP.values())),
    )

    rows = []
    for year in split.test_years:
        current = split.test[split.test[config.YEAR_COLUMN] == year]
        X_current, y_current, _ = _feature_frame(current)
        transformed = preprocessor.transform(X_current)
        predictions = model.predict(transformed)
        if len(predictions) != len(y_current):
            raise ValueError(f"incremental update for {year} returned the wrong prediction count")
        allowed = set(config.LABEL_MAP.values())
        if set(y_current) - allowed:
            raise ValueError(f"incremental update for {year} contains invalid labels")
        rows.append(_score_predictions("incremental_partial_fit", year, y_current, predictions))
        model.partial_fit(transformed, y_current)
    return rows


def _validate_results(results, test_years):
    expected_strategies = {
        "frozen_baseline",
        "periodic_retraining",
        "incremental_partial_fit",
    }
    if set(results["strategy"]) != expected_strategies:
        raise ValueError("Step 7 results do not contain all sustainability strategies")
    if set(results["year"]) != set(test_years):
        raise ValueError("Step 7 results do not contain every test year")
    expected_rows = len(expected_strategies) * len(test_years)
    if len(results) != expected_rows:
        raise ValueError(f"Step 7 results expected {expected_rows} rows, got {len(results)}")
    if results.duplicated(["strategy", "year"]).any():
        raise ValueError("Step 7 results contain duplicate strategy/year rows")
    expected_pairs = {
        (strategy, int(year))
        for strategy in expected_strategies
        for year in test_years
    }
    if set(zip(results["strategy"], results["year"])) != expected_pairs:
        raise ValueError("Step 7 results do not contain every strategy/year combination")
    if not np.isfinite(results[list(evaluation.METRICS)].to_numpy()).all():
        raise ValueError("Step 7 results contain non-finite metrics")


def _report_text(results, split, alpha):
    """Validate report inputs and return the Step 7 report text."""
    _validate_results(results, split.test_years)
    shown = results.sort_values(["year", "strategy"]).copy()
    for metric in evaluation.METRICS:
        shown[metric] = shown[metric].map(lambda value: f"{value:.4f}")

    summary = (
        results.groupby("strategy", as_index=False)[list(evaluation.METRICS)]
        .mean()
        .sort_values("accuracy", ascending=False, kind="stable")
    )
    for metric in evaluation.METRICS:
        summary[metric] = summary[metric].map(lambda value: f"{value:.4f}")

    lines = [
        "# Step 7 Report: Sustainable ML Approach",
        "",
        "Written by the builder agent. Generated by `python -m src.main --step 7`.",
        "",
        f"The frozen training period is {', '.join(map(str, split.train_years))}; each "
        f"strategy is evaluated forward on test years {', '.join(map(str, split.test_years))}. "
        "The model used for the sustainability comparison is SGDClassifier because it supports "
        "incremental `partial_fit` updates. All three strategies use the Step 4-selected "
        f"SGD configuration with alpha={alpha:g}.",
        "",
        "## Strategies",
        "",
        "**Frozen baseline.** The Step 4 SGD model is never changed and predicts every test "
        "year. This represents a model that is deployed once and never retrained.",
        "",
        "**Periodic retraining.** Before each test year, a new SGD model and scaler are fitted "
        "using every year available before that test year. The current year is predicted before "
        "it can enter training, preventing temporal leakage.",
        "",
        "**Incremental adaptation.** An SGD model is initialized on the original training years. "
        "For each test year, it predicts first and then receives that year's labeled data through "
        "`partial_fit`, simulating an update after labels become available. Its preprocessing "
        "scaler remains fit on the original training years so the comparison isolates model "
        "adaptation rather than changing both the representation and classifier simultaneously.",
        "",
        "## Unweighted mean performance across test years",
        "",
        "Each test year contributes equally to this summary, regardless of its row count. "
        "The per-year table includes `n_test_rows` so these averages can be distinguished "
        "from pooled metrics over all test examples.",
        "",
        summary.to_markdown(index=False),
        "",
        "## Performance by year",
        "",
        shown.to_markdown(index=False),
        "",
        "## Conclusion",
        "",
        "The frozen baseline shows how performance changes when the detector is not updated. "
        "Periodic retraining adapts both preprocessing and model parameters to the accumulated "
        "data, while incremental SGD adapts parameters without a full retraining cycle. "
        "As a general operational consideration, periodic retraining may be preferable when "
        "labeled data can be collected in batches, while incremental updates may be useful "
        "when updates must be frequent and computationally inexpensive. This experiment "
        "measures predictive performance, not operational safety, label quality, or deployment "
        "cost; both approaches require monitoring for drift and regressions before deployment.",
        "",
        "Machine-readable results: `results/step7_sustainable_results.csv`.",
    ]
    return "\n".join(lines) + "\n"


def write_report(results, split, path, alpha):
    """Validate and write the Step 7 sustainable ML comparison report."""
    report = _report_text(results, split, alpha)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(report, encoding="utf-8")


def run(input_dir=None, models_dir=None, results_dir=None, docs_dir=None):
    """Run the sustainable ML comparison without using future-year labels early."""
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

    alpha = _selected_sgd_params(results_path)
    if sorted(split.train_years) != list(split.train_years):
        raise ValueError("Step 7 training years must be sorted")
    if sorted(split.test_years) != list(split.test_years):
        raise ValueError("Step 7 test years must be sorted")
    if not split.test_years or max(split.train_years) >= min(split.test_years):
        raise ValueError("Step 7 requires every test year to be later than every training year")

    rows = _frozen_predictions(
        models_path,
        X_test,
        y_test.to_numpy(),
        years_test.to_numpy(),
        alpha,
    )
    rows.extend(_periodic_retraining(df, split.test_years, alpha))
    rows.extend(_incremental_update(split, alpha))
    results = pd.DataFrame(rows)
    _validate_results(results, split.test_years)
    report_path = docs_path / "step7_sustainable_approach.md"
    report_text = _report_text(results, split, alpha)
    results.to_csv(results_path / "step7_sustainable_results.csv", index=False)
    report_path.write_text(report_text, encoding="utf-8")
    print(results.to_string(index=False))
    return results
