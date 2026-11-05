import numpy as np
import pandas as pd
import pytest

from src import config
from src.drift import (
    ALPHA,
    class_cancellation,
    count_bins,
    feature_drift,
    js_distance,
    pair_row,
    pair_summary,
    short_name,
    symmetric_matrix,
    top_features,
    unique_vector_index,
    year_classifier,
    year_stats,
)


def test_count_bins_and_js_distance_are_valid_for_count_features():
    assert count_bins([0, 1, 2, 3, 4, 8, 9]).tolist() == [0, 1, 2, 3, 3, 4, 5]

    identical = np.array([[3, 0], [0, 3]], dtype=float)
    assert np.allclose(js_distance(identical, identical), 0.0)

    different = np.array([[4, 0], [0, 4]], dtype=float)
    distance = js_distance(identical, different)
    assert np.all(distance >= 0)
    assert np.all(distance <= 1)


def test_feature_drift_detects_shift_and_preserves_input():
    X = np.array([
        [1, 0],
        [1, 0],
        [2, 1],
        [2, 1],
        [8, 0],
        [8, 0],
        [9, 1],
        [9, 1],
    ], dtype=np.int32)
    years = np.array([2019] * 4 + [2020] * 4)
    labels = np.array([0, 0, 1, 1, 0, 0, 1, 1])
    names = ["shifted", "stable"]
    before = X.copy()

    drift = feature_drift(X, years, labels, names, pairs=[(2019, 2020)])

    assert np.array_equal(X, before)
    assert set(drift["class"]) == {"all", "benign", "malware"}
    assert len(drift) == 6
    pooled = drift[drift["class"] == "all"].set_index("feature")
    assert pooled.loc["shifted", "ks"] > 0.9
    assert pooled.loc["shifted", "prevalence_change"] == 0
    assert (drift["q_value"] >= drift["p_value"]).all()
    assert drift["p_value"].between(0, 1).all()


def test_pair_summary_and_symmetric_matrix_require_complete_unique_pairs():
    X = np.array([[0], [1], [0], [2], [1], [3]], dtype=np.int32)
    years = np.array([2019, 2019, 2020, 2020, 2021, 2021])
    labels = np.array([0, 1, 0, 1, 0, 1])
    drift = feature_drift(X, years, labels, ["api"], pairs=[(2019, 2020), (2019, 2021), (2020, 2021)])
    summary = pair_summary(drift)

    matrix = symmetric_matrix(summary, "mean_ks", "all", [2019, 2020, 2021])
    assert matrix.shape == (3, 3)
    assert np.allclose(np.diag(matrix), 0)
    assert np.allclose(matrix.to_numpy(), matrix.to_numpy().T)

    with pytest.raises(KeyError, match="no row"):
        pair_row(summary, 2018, 2020)

    duplicate = pd.concat([summary, summary.iloc[[0]]], ignore_index=True)
    with pytest.raises(ValueError, match="duplicate rows"):
        symmetric_matrix(duplicate, "mean_ks", "all", [2019, 2020, 2021])

    missing = summary[~((summary["year_a"] == 2019) & (summary["year_b"] == 2021))]
    with pytest.raises(ValueError, match="missing"):
        symmetric_matrix(missing, "mean_ks", "all", [2019, 2020, 2021])


def test_top_features_are_ranked_within_each_comparison():
    table = pd.DataFrame({
        "year_a": [2019, 2019, 2019, 2019],
        "year_b": [2020, 2020, 2020, 2020],
        "class": ["all", "all", "all", "benign"],
        "feature": ["a", "b", "c", "a"],
        "ks": [0.2, 0.9, 0.5, 0.8],
    })
    result = top_features(table, k=2)
    pooled = result[result["class"] == "all"]
    assert pooled["feature"].tolist() == ["b", "c"]
    assert pooled["rank"].tolist() == [1, 2]


def test_class_cancellation_counts_opposite_prevalence_changes():
    rows = []
    for cls, ks, change in (
        ("all", 0.1, 0.0),
        ("benign", 0.3, 0.4),
        ("malware", 0.4, -0.2),
    ):
        rows.append({
            "year_a": 2019, "year_b": 2020, "class": cls, "feature": "api",
            "ks": ks, "prevalence_change": change,
        })
    result = class_cancellation(pd.DataFrame(rows), [(2019, 2020)])
    row = result.iloc[0]
    assert row["n_features_compared"] == 1
    assert row["n_pooled"] == 0
    assert row["n_in_a_class"] == 1
    assert row["n_in_a_class_opposite_direction"] == 1

    incomplete = pd.DataFrame(rows[:-1])
    with pytest.raises(KeyError, match="no rows"):
        class_cancellation(incomplete, [(2019, 2020)])


def test_vector_deduplication_and_year_stats_keep_identifiers_separate():
    X = np.array([[1, 0], [1, 0], [2, 0], [1, 0]], dtype=np.int32)
    years = np.array([2019, 2019, 2020, 2020])
    labels = np.array([0, 1, 0, 1])
    names = np.array(["same-a", "same-b", "other", "same-c"])

    assert unique_vector_index(X, np.array([0, 1, 2])).tolist() == [0, 2]
    stats = year_stats(X, years, labels, names)
    assert stats.loc[2019, "available_rows"] == 2
    assert stats.loc[2019, "available_unique_vectors"] == 1
    assert stats.loc[2019, "unique_apknames"] == 2
    assert stats.loc[2020, "n_benign"] == 1
    assert stats.loc[2020, "n_malware"] == 1


def test_year_classifier_reports_small_blocks_instead_of_failing():
    X = np.array([[0], [1], [2], [3]], dtype=np.int32)
    years = np.array([2019, 2019, 2020, 2020])
    labels = np.array([0, 1, 0, 1])
    result = year_classifier(
        X, years, labels, [(2019, 2020)], "all", min_vectors=3, n_splits=2
    )
    row = result.iloc[0]
    assert bool(row["skipped"])
    assert row["classifier_sample_size"] == 0
    assert "fewer than 3" in row["skip_reason"]

    with pytest.raises(ValueError, match="negative counts"):
        year_classifier(
            np.array([[-1], [0], [1], [2]]),
            years,
            labels,
            [(2019, 2020)],
            "all",
            min_vectors=1,
        )


def test_config_split_years_validates_boundary():
    assert config.split_years([2021, 2019, 2020, 2022]) == ([2019, 2020], [2021, 2022])

    with pytest.raises(ValueError, match="at least 4 years"):
        config.split_years([2019, 2020, 2021])

    with pytest.raises(ValueError, match="repeated years"):
        config.split_years([2019, 2019, 2020, 2021])


def test_short_name_removes_signature_suffix_and_truncates():
    assert short_name("Lpkg/Class;.method:(I)V") == "Lpkg/Class;.method"
    assert short_name("x" * 80, width=10) == "xxxxxxx..."
