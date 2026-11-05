import numpy as np
import pandas as pd
import pytest

from src import config
from src.data_loading import (
    build_dataset,
    feature_columns,
    load_file,
    parse_filename,
    split_features,
)
from src.data_summary import counts_per_year, duplicate_summary
from src.main import run_step1


FEATURE_NAMES = [f"api_{i}" for i in range(config.EXPECTED_FEATURES)]


def write_csv(directory, year, class_name, rows):
    columns = [config.ID_COLUMN, *FEATURE_NAMES]
    pd.DataFrame(rows, columns=columns).to_csv(
        directory / f"sampled_{year}_{class_name}_api.csv", index=False
    )


def row(identifier, first_feature=0):
    return [identifier, first_feature, *([0] * (config.EXPECTED_FEATURES - 1))]


def test_parse_filename_and_rejects_unknown_name(tmp_path):
    assert parse_filename("sampled_2019_malware_api.csv") == (2019, 1)
    assert parse_filename("sampled_2020_benign_api.csv") == (2020, 0)

    with pytest.raises(ValueError, match="does not match"):
        parse_filename("sampled_2020_goodware_api.csv")


def test_load_file_keeps_identifier_as_metadata_and_labels_from_filename(tmp_path):
    write_csv(tmp_path, 2019, "malware", [row("app-a", 3)])

    loaded = load_file(tmp_path / "sampled_2019_malware_api.csv")

    assert loaded.shape == (1, config.EXPECTED_FEATURES + 4)
    assert loaded[config.ID_COLUMN].tolist() == ["app-a"]
    assert loaded[config.YEAR_COLUMN].tolist() == [2019]
    assert loaded[config.LABEL_COLUMN].tolist() == [1]
    assert len(feature_columns(loaded)) == config.EXPECTED_FEATURES
    assert config.ID_COLUMN not in feature_columns(loaded)

    X, y, meta = split_features(loaded)
    assert X.shape == (1, config.EXPECTED_FEATURES)
    assert config.ID_COLUMN not in X.columns
    assert y.tolist() == [1]
    assert meta[config.ID_COLUMN].tolist() == ["app-a"]


def test_build_dataset_requires_both_classes_and_matching_columns(tmp_path):
    write_csv(tmp_path, 2019, "benign", [row("benign-app")])
    with pytest.raises(ValueError, match="missing a class file"):
        build_dataset(tmp_path)

    write_csv(tmp_path, 2019, "malware", [row("malware-app", 1)])
    write_csv(tmp_path, 2020, "benign", [row("benign-2020")])
    write_csv(tmp_path, 2020, "malware", [row("malware-2020", 1)])

    dataset = build_dataset(tmp_path)
    assert len(dataset) == 4
    assert sorted(dataset[config.YEAR_COLUMN].unique()) == [2019, 2020]
    assert set(dataset[config.LABEL_COLUMN]) == {0, 1}
    assert dataset.columns.tolist() == (
        [config.ID_COLUMN, config.YEAR_COLUMN, config.LABEL_COLUMN]
        + FEATURE_NAMES
        + ["source_file"]
    )


def test_loader_rejects_missing_negative_float_and_overflow_features(tmp_path):
    cases = [
        ("missing", [[np.nan] + [0] * (config.EXPECTED_FEATURES - 1)]),
        ("non_integer", [[0.5] + [0] * (config.EXPECTED_FEATURES - 1)]),
        ("negative", [[-1] + [0] * (config.EXPECTED_FEATURES - 1)]),
        ("overflow", [[np.iinfo(np.int32).max + 1] + [0] * (config.EXPECTED_FEATURES - 1)]),
    ]
    for name, features in cases:
        path = tmp_path / f"sampled_2019_benign_api.csv"
        pd.DataFrame(
            [["app", *features[0]]], columns=[config.ID_COLUMN, *FEATURE_NAMES]
        ).to_csv(path, index=False)
        expected = {
            "missing": "missing values",
            "non_integer": "non-integer feature values",
            "negative": "negative",
            "overflow": "int32 range",
        }[name]
        with pytest.raises(ValueError, match=expected):
            load_file(path)


def test_duplicate_identifier_and_vector_counts_are_separate():
    df = pd.DataFrame(
        {
            config.ID_COLUMN: ["same-app", "same-app", "other-app", "third-app"],
            config.YEAR_COLUMN: [2019, 2019, 2019, 2020],
            config.LABEL_COLUMN: [0, 1, 1, 1],
            "api_0": [1, 1, 1, 1],
            "api_1": [0, 0, 0, 2],
            "source_file": ["a.csv", "a.csv", "a.csv", "b.csv"],
        }
    )

    per_file, cross, totals = duplicate_summary(df)

    first_file = per_file.set_index("file").loc["a.csv"]
    assert first_file["duplicate_apkname_rows"] == 1
    assert first_file["duplicated_apkname_keys"] == 1
    assert first_file["duplicate_vector_rows"] == 2
    assert first_file["duplicated_vector_keys"] == 1
    assert totals["unique_apkname"] == 3
    assert totals["unique_feature_vectors"] == 2
    assert cross.set_index("measure").loc[
        "apkname appearing in both classes", "unique_keys"
    ] == 1
    assert cross.set_index("measure").loc[
        "feature vector appearing in both classes", "unique_keys"
    ] == 1


def test_counts_per_year_and_step1_entry_point(tmp_path):
    for year in (2019, 2020):
        write_csv(tmp_path, year, "benign", [row(f"b-{year}")])
        write_csv(tmp_path, year, "malware", [row(f"m-{year}", 1)])

    dataset = build_dataset(tmp_path)
    counts = counts_per_year(dataset)
    assert counts["rows"].tolist() == [2, 2]
    assert counts["malware"].tolist() == [1, 1]
    assert counts["benign"].tolist() == [1, 1]
    assert counts["malware_share"].tolist() == [0.5, 0.5]

    result = run_step1(tmp_path)
    assert len(result) == 4
