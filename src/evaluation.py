"""Step 5: evaluate the fitted classifiers on the held-out test years."""
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.utils.validation import check_is_fitted

from src import config, models, preprocessing
from src.data_loading import build_dataset


METRICS = ("accuracy", "precision", "recall", "f1")
METRIC_LABELS = {
    "accuracy": "Accuracy",
    "precision": "Precision",
    "recall": "Recall",
    "f1": "F1-score",
}


def classification_metrics(y_true, y_pred):
    """Return the required binary metrics with malware (label 1) as positive."""
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    if y_true.ndim != 1 or y_pred.ndim != 1:
        raise ValueError("classification_metrics: labels must be one-dimensional")
    if len(y_true) == 0:
        raise ValueError("classification_metrics: evaluation inputs cannot be empty")
    if y_true.shape[0] != y_pred.shape[0]:
        raise ValueError(
            f"classification_metrics: {len(y_true)} true labels but {len(y_pred)} predictions"
        )
    allowed = set(config.LABEL_MAP.values())
    if set(y_true) - allowed or set(y_pred) - allowed:
        raise ValueError(
            f"classification_metrics: labels must be {sorted(allowed)}"
        )
    if set(y_true) != allowed:
        raise ValueError(
            f"classification_metrics: test labels must contain both classes {sorted(allowed)}"
        )
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, pos_label=1, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, pos_label=1, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, pos_label=1, zero_division=0)),
    }


def evaluate_models(models_dir, X_test, y_test):
    """Load each Step 4 model and score it once on the held-out test matrix."""
    models_dir = Path(models_dir)
    rows = []
    for name in models.MODEL_SPECS:
        path = models_dir / f"{name}.joblib"
        if not path.is_file():
            raise FileNotFoundError(f"Step 4 model artifact not found: {path}")
        estimator = joblib.load(path)
        if not callable(getattr(estimator, "predict", None)):
            raise TypeError(f"{path}: loaded artifact does not provide predict()")
        check_is_fitted(estimator)
        model_columns = getattr(estimator, "feature_names_in_", None)
        if model_columns is not None and list(model_columns) != list(X_test.columns):
            raise ValueError(
                f"{path}: model feature names/order do not match the test feature contract"
            )
        predictions = estimator.predict(X_test)
        if len(predictions) != len(y_test):
            raise ValueError(
                f"{path}: produced {len(predictions)} predictions for {len(y_test)} test rows"
            )
        scores = classification_metrics(y_test, predictions)
        rows.append({"classifier": name, "n_test_rows": len(y_test), **scores})

    result = pd.DataFrame(rows)
    if result["classifier"].duplicated().any():
        raise ValueError("evaluate_models: duplicate classifier results")
    if not np.isfinite(result[list(METRICS)].to_numpy()).all():
        raise ValueError("evaluate_models: non-finite metric value")
    return result.sort_values("accuracy", ascending=False, kind="stable").reset_index(drop=True)


def write_report(metrics, split, path):
    """Validate and write the required Step 5 evaluation report."""
    report = _report_text(metrics, split)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(report, encoding="utf-8")


def _report_text(metrics, split):
    """Validate report inputs and return the report text without writing it."""
    required = {"classifier", "n_test_rows", *METRICS}
    missing = required - set(metrics.columns)
    if missing:
        raise ValueError(f"write_report: missing columns {sorted(missing)}")
    if set(metrics["classifier"]) != set(models.MODEL_SPECS):
        raise ValueError("write_report: results do not contain exactly the Step 4 classifiers")
    if metrics.empty or metrics["classifier"].duplicated().any():
        raise ValueError("write_report: classifier results must be nonempty and unique")
    if not np.isfinite(metrics[list(METRICS)].to_numpy()).all():
        raise ValueError("write_report: metric values must be finite")
    if not ((metrics[list(METRICS)] >= 0) & (metrics[list(METRICS)] <= 1)).all().all():
        raise ValueError("write_report: metric values must be between 0 and 1")

    table = metrics.copy()
    for metric in METRICS:
        table[metric] = table[metric].map(lambda value: f"{value:.4f}")
    table = table.rename(columns={
        "classifier": "Classifier",
        "n_test_rows": "Test rows",
        **METRIC_LABELS,
    })
    winner = metrics.sort_values("accuracy", ascending=False, kind="stable").iloc[0]
    metric_names = ", ".join(METRIC_LABELS[metric] for metric in METRICS)
    lines = [
        "# Step 5 Report: Classifier Evaluation",
        "",
        "Written by the builder agent. Generated by `python -m src.main --step 5`; "
        "all metrics come from predictions on the held-out test years.",
        "",
        f"Training years: {', '.join(map(str, split.train_years))}.  ",
        f"Test years: {', '.join(map(str, split.test_years))}.",
        "",
        f"{metric_names} are reported. Accuracy is the primary metric, as required by the "
        "assignment. Precision, recall, and F1-score are also reported because malware "
        "detection can be imbalanced. Malware is label `1` and is therefore the positive "
        "class for all three additional metrics. Undefined precision or F1 values are "
        "scored as zero when a classifier predicts no malware.",
        "",
        table.to_markdown(index=False),
        "",
        f"**Primary comparison.** {winner['classifier']} has the highest held-out accuracy "
        f"({winner['accuracy']:.4f}) on {int(winner['n_test_rows']):,} test rows.",
        "",
        "Step 5 does not fit or tune models and uses the Step 4 artifacts according to the "
        "intended training-year-only workflow. Artifact training history is established by "
        "the Step 4 pipeline and its reports, not independently reconstructed here.",
        "",
        "Machine-readable results: `results/step5_test_metrics.csv`.",
    ]
    return "\n".join(lines) + "\n"


def run(input_dir=None, models_dir=None, results_dir=None, docs_dir=None):
    """Run Step 5 using the validated Step 3 split and Step 4 model artifacts."""
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
    preprocessing.check_labels_years(
        split, y_train, years_train, y_test, years_test
    )
    metrics = evaluate_models(
        models_path,
        X_test,
        y_test,
    )
    report = _report_text(metrics, split)
    metrics.to_csv(results_path / "step5_test_metrics.csv", index=False)
    (docs_path / "step5_evaluation.md").write_text(report, encoding="utf-8")
    print(metrics.to_string(index=False))
    return metrics
