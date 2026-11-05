import numpy as np
import pandas as pd
import pytest

from src import config
from src import preprocessing


def make_dataset(years=(2019, 2020, 2021, 2022), rows_per_year=2):
    feature_names = [f"api_{i}" for i in range(config.EXPECTED_FEATURES)]
    rows = []
    for year in years:
        for offset in range(rows_per_year):
            values = np.zeros(config.EXPECTED_FEATURES, dtype=np.int32)
            values[0] = year - years[0]
            values[1] = offset
            values[2] = 1
            rows.append(
                [f"app-{year}-{offset}", year, offset % 2, *values, f"{year}.csv"]
            )
    return pd.DataFrame(
        rows,
        columns=[
            config.ID_COLUMN,
            config.YEAR_COLUMN,
            config.LABEL_COLUMN,
            *feature_names,
            "source_file",
        ],
    )


def test_split_by_year_is_time_ordered_and_preserves_rows():
    df = make_dataset()
    split = preprocessing.split_by_year(df)

    assert split.train_years == [2019, 2020]
    assert split.test_years == [2021, 2022]
    assert set(split.train[config.YEAR_COLUMN]) == {2019, 2020}
    assert set(split.test[config.YEAR_COLUMN]) == {2021, 2022}
    assert len(split.train) + len(split.test) == len(df)
    assert max(split.train_years) < min(split.test_years)


def test_check_split_rejects_overlap_missing_years_and_bad_order():
    df = make_dataset()
    valid = preprocessing.split_by_year(df)

    overlap = valid._replace(test_years=[2020, 2021])
    with pytest.raises(ValueError, match="both train and test"):
        preprocessing.check_split(overlap, all_years=[2019, 2020, 2021, 2022])

    missing = valid._replace(test_years=[2022])
    with pytest.raises(ValueError, match="neither split"):
        preprocessing.check_split(missing, all_years=[2019, 2020, 2021, 2022])

    bad_order = valid._replace(train_years=[2021, 2022], test_years=[2019, 2020])
    with pytest.raises(ValueError, match="not all later"):
        preprocessing.check_split(bad_order)


def test_features_labels_excludes_metadata_and_enforces_feature_contract():
    df = make_dataset()
    X, y, years = preprocessing.features_labels(df)

    assert X.shape == (len(df), config.EXPECTED_FEATURES)
    assert config.ID_COLUMN not in X.columns
    assert config.YEAR_COLUMN not in X.columns
    assert config.LABEL_COLUMN not in X.columns
    assert "source_file" not in X.columns
    assert y.tolist() == df[config.LABEL_COLUMN].tolist()
    assert years.tolist() == df[config.YEAR_COLUMN].tolist()

    with pytest.raises(ValueError, match="do not match"):
        preprocessing.features_labels(df, expected_columns=list(reversed(X.columns)))


def test_features_labels_rejects_negative_non_numeric_and_non_finite_values():
    df = make_dataset()

    negative = df.copy()
    negative.loc[0, "api_0"] = -1
    with pytest.raises(ValueError, match="negative count"):
        preprocessing.features_labels(negative)

    text = df.copy()
    text["api_3"] = "count"
    with pytest.raises(ValueError, match="non-numeric"):
        preprocessing.features_labels(text)

    nonfinite = df.copy()
    nonfinite["api_4"] = 1.0
    nonfinite.loc[0, "api_4"] = np.nan
    with pytest.raises(ValueError, match="non-finite"):
        preprocessing.features_labels(nonfinite)


def test_pipeline_drops_training_constants_and_scales_training_statistics_only():
    df = make_dataset()
    split = preprocessing.split_by_year(df)
    X_train, _, _ = preprocessing.features_labels(split.train)
    X_test, _, _ = preprocessing.features_labels(
        split.test, expected_columns=X_train.columns
    )

    pipeline = preprocessing.fit_preprocessor(
        preprocessing.make_preprocessor(scale=True),
        X_train,
        split.train_years,
    )
    kept = preprocessing.kept_features(pipeline, split.train_years)
    dropped = [column for column in X_train.columns if column not in kept]

    assert "api_2" in dropped
    assert "api_0" in kept
    preprocessing.check_feature_contract(X_train, X_test)
    preprocessing.check_feature_lists(X_train, X_test, kept, dropped)

    check = preprocessing.verify_scaler(pipeline, X_train)
    assert check["max_abs_mean_difference"] < 1e-12
    assert check["max_abs_scale_difference"] < 1e-12

    transformed = pipeline.transform(X_train)
    assert transformed.shape == (len(X_train), len(kept))
    assert np.allclose(transformed.mean(axis=0), 0.0)


def test_unscaled_pipeline_and_model_scaling_decisions():
    unscaled = preprocessing.make_preprocessor(scale=False)
    assert "scale" not in unscaled.named_steps
    assert preprocessing.verify_scaler(unscaled, pd.DataFrame()) is None

    assert preprocessing.preprocessor_for("LogisticRegression").named_steps["scale"]
    assert "scale" not in preprocessing.preprocessor_for(
        "RandomForestClassifier"
    ).named_steps
    with pytest.raises(KeyError, match="no scaling decision"):
        preprocessing.preprocessor_for("UnknownModel")

    configs = preprocessing.artifact_configs()
    assert configs["scaled"][0] is True
    assert configs["unscaled"][0] is False
    assert set(configs["scaled"][1]) == {
        "LogisticRegression",
        "LinearSVC",
        "SGDClassifier",
    }
    assert set(configs["unscaled"][1]) == {
        "DummyClassifier",
        "RandomForestClassifier",
    }


def test_forward_chaining_folds_and_indices_respect_time_order():
    folds = preprocessing.forward_chaining_folds([2022, 2020, 2021])
    assert folds == [([2020], 2021), ([2020, 2021], 2022)]
    for train_years, validation_year in folds:
        assert max(train_years) < validation_year

    years = np.array([2020, 2020, 2021, 2021, 2022, 2022])
    indices = preprocessing.fold_indices(years, folds)
    assert [(train.tolist(), valid.tolist()) for train, valid in indices] == [
        ([0, 1], [2, 3]),
        ([0, 1, 2, 3], [4, 5]),
    ]

    with pytest.raises(ValueError, match="at least 2 training years"):
        preprocessing.forward_chaining_folds([2020])


def test_split_summary_and_report_input_validation():
    df = make_dataset()
    split = preprocessing.split_by_year(df)
    summary = preprocessing.split_summary(df, split)

    assert summary["rows"].sum() == len(df)
    assert (summary["benign"] + summary["malware"] == summary["rows"]).all()
    assert set(summary["split"]) == {"train", "test"}
    preprocessing.check_report_inputs(summary, split, total_rows=len(df))

    invalid = df.copy()
    invalid.loc[0, config.LABEL_COLUMN] = 7
    with pytest.raises(ValueError, match="label value"):
        preprocessing.split_summary(invalid, split)

    missing_test = summary[summary["split"] == "train"]
    with pytest.raises(ValueError, match="no \\['test'\\]"):
        preprocessing.check_report_inputs(missing_test, split)


def test_all_constant_training_features_fail_loudly_with_year_context():
    df = make_dataset()
    split = preprocessing.split_by_year(df)
    X_train, _, _ = preprocessing.features_labels(split.train)
    X_train = X_train.copy()
    X_train.loc[:, :] = 1

    with pytest.raises(ValueError, match="all .* features are constant.*2019"):
        preprocessing.fit_preprocessor(
            preprocessing.make_preprocessor(),
            X_train,
            split.train_years,
        )
