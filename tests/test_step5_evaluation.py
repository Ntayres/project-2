import pandas as pd
import pytest
import numpy as np
from sklearn.dummy import DummyClassifier

from src import config, evaluation, preprocessing


class WrongLengthEstimator:
    def __init__(self):
        self.feature_names_in_ = np.array(["api_a", "api_b"])
        self.fitted_ = True

    def fit(self, X, y):
        return self

    def predict(self, values):
        return np.array([0])


def test_classification_metrics_match_hand_checked_binary_example():
    scores = evaluation.classification_metrics([0, 0, 1, 1], [0, 1, 1, 1])

    assert scores == {
        "accuracy": 0.75,
        "precision": pytest.approx(2 / 3),
        "recall": 1.0,
        "f1": pytest.approx(0.8),
    }


def test_classification_metrics_rejects_bad_lengths_and_labels():
    with pytest.raises(ValueError, match="true labels"):
        evaluation.classification_metrics([0], [0, 1])
    with pytest.raises(ValueError, match="labels must be"):
        evaluation.classification_metrics([0, 2], [0, 1])
    with pytest.raises(ValueError, match="cannot be empty"):
        evaluation.classification_metrics([], [])
    with pytest.raises(ValueError, match="both classes"):
        evaluation.classification_metrics([0, 0], [0, 1])


def test_evaluate_models_scores_only_the_supplied_test_rows(tmp_path):
    X_train = pd.DataFrame({"api_a": [0, 1, 2, 3], "api_b": [1, 1, 1, 1]})
    y_train = pd.Series([0, 0, 1, 1])
    model = DummyClassifier(strategy="most_frequent", random_state=config.RANDOM_STATE)
    model.fit(X_train, y_train)
    model_dir = tmp_path / "models"
    model_dir.mkdir()
    import joblib
    for name in evaluation.models.MODEL_SPECS:
        joblib.dump(model, model_dir / f"{name}.joblib")

    result = evaluation.evaluate_models(
        model_dir,
        pd.DataFrame({"api_a": [99, 100], "api_b": [1, 1]}),
        pd.Series([0, 1]),
    )

    assert len(result) == len(evaluation.models.MODEL_SPECS)
    assert (result["n_test_rows"] == 2).all()
    assert (result["accuracy"] == 0.5).all()


def test_evaluate_models_rejects_a_model_with_wrong_feature_contract(tmp_path):
    model = DummyClassifier(strategy="most_frequent", random_state=config.RANDOM_STATE)
    model.fit(pd.DataFrame({"wrong_feature": [0, 1, 2, 3]}), [0, 0, 1, 1])
    model_dir = tmp_path / "models"
    model_dir.mkdir()
    import joblib
    for name in evaluation.models.MODEL_SPECS:
        joblib.dump(model, model_dir / f"{name}.joblib")

    with pytest.raises(ValueError, match="feature names/order"):
        evaluation.evaluate_models(
            model_dir,
            pd.DataFrame({"api_a": [0, 1], "api_b": [1, 1]}),
            pd.Series([0, 1]),
        )


def test_write_report_requires_all_classifiers(tmp_path):
    split = preprocessing.Split(
        train=pd.DataFrame(),
        test=pd.DataFrame(),
        train_years=[2019, 2020],
        test_years=[2021, 2022],
    )
    metrics = pd.DataFrame([{
        "classifier": "DummyClassifier",
        "n_test_rows": 2,
        "accuracy": 0.5,
        "precision": 0.0,
        "recall": 0.0,
        "f1": 0.0,
    }])

    with pytest.raises(ValueError, match="exactly"):
        evaluation.write_report(metrics, split, tmp_path / "report.md")


def test_write_report_selects_highest_accuracy_even_when_input_is_unsorted(tmp_path):
    split = preprocessing.Split(
        train=pd.DataFrame(),
        test=pd.DataFrame(),
        train_years=[2019, 2020],
        test_years=[2021, 2022],
    )
    metrics = pd.DataFrame([
        {
            "classifier": name,
            "n_test_rows": 4,
            "accuracy": accuracy,
            "precision": 0.5,
            "recall": 0.5,
            "f1": 0.5,
        }
        for name, accuracy in [
            ("DummyClassifier", 0.5),
            ("LogisticRegression", 0.7),
            ("SGDClassifier", 0.8),
            ("RandomForestClassifier", 0.9),
        ]
    ])

    report_path = tmp_path / "report.md"
    evaluation.write_report(metrics, split, report_path)

    assert "RandomForestClassifier has the highest held-out accuracy (0.9000)" in (
        report_path.read_text(encoding="utf-8")
    )


@pytest.mark.parametrize(
    "y_true, y_pred, message",
    [
        ([[0, 1]], [1], "one-dimensional"),
        ([0, 1], [0], "true labels but 1 predictions"),
        ([0, 1], [0, 2], "labels must be"),
    ],
)
def test_classification_metrics_rejects_shape_and_prediction_errors(
    y_true, y_pred, message
):
    with pytest.raises(ValueError, match=message):
        evaluation.classification_metrics(y_true, y_pred)


def test_evaluate_models_requires_all_artifacts(tmp_path):
    X = pd.DataFrame({"api_a": [0, 1], "api_b": [1, 0]})
    y = pd.Series([0, 1])
    model_dir = tmp_path / "models"
    model_dir.mkdir()

    with pytest.raises(FileNotFoundError, match="model artifact not found"):
        evaluation.evaluate_models(model_dir, X, y)


def test_evaluate_models_rejects_unfitted_artifact(tmp_path):
    import joblib

    X = pd.DataFrame({"api_a": [0, 1], "api_b": [1, 0]})
    y = pd.Series([0, 1])
    model_dir = tmp_path / "models"
    model_dir.mkdir()
    for name in evaluation.models.MODEL_SPECS:
        joblib.dump(DummyClassifier(), model_dir / f"{name}.joblib")

    with pytest.raises(Exception):
        evaluation.evaluate_models(model_dir, X, y)


def test_evaluate_models_rejects_prediction_length_mismatch(tmp_path):
    import joblib

    X_train = pd.DataFrame({"api_a": [0, 1, 2, 3], "api_b": [1, 1, 1, 1]})
    y_train = pd.Series([0, 0, 1, 1])
    model_dir = tmp_path / "models"
    model_dir.mkdir()
    for name in evaluation.models.MODEL_SPECS:
        joblib.dump(WrongLengthEstimator(), model_dir / f"{name}.joblib")

    with pytest.raises(ValueError, match="produced 1 predictions"):
        evaluation.evaluate_models(model_dir, X_train, y_train)


@pytest.mark.parametrize(
    "mutator, message",
    [
        (lambda frame: frame.drop(columns=["f1"]), "missing columns"),
        (lambda frame: frame.assign(accuracy=np.nan), "finite"),
        (lambda frame: frame.assign(f1=1.1), "between 0 and 1"),
        (lambda frame: pd.concat([frame, frame.iloc[[0]]]), "unique"),
    ],
)
def test_report_rejects_invalid_metric_tables(mutator, message):
    split = preprocessing.Split(
        train=pd.DataFrame(),
        test=pd.DataFrame(),
        train_years=[2019, 2020],
        test_years=[2021, 2022],
    )
    metrics = pd.DataFrame([
        {
            "classifier": name,
            "n_test_rows": 4,
            "accuracy": 0.5,
            "precision": 0.5,
            "recall": 0.5,
            "f1": 0.5,
        }
        for name in evaluation.models.MODEL_SPECS
    ])

    with pytest.raises(ValueError, match=message):
        evaluation._report_text(mutator(metrics), split)
