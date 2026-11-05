"""Entry point: run project steps.

Usage:
    python -m src.main --step 1
    python -m src.main --step 2
    python -m src.main --step 3
    python -m src.main --step 4
    python -m src.main --step 5
    python -m src.main --step 6
    python -m src.main --step 7
    python -m src.main --all
    python -m src.main --step 1 --input-dir path/to/input_data
"""
import argparse

import numpy as np

from src import config, drift, evaluation, predictions, preprocessing, sustainable
from src.data_loading import build_dataset, feature_columns
from src.data_summary import counts_per_year, duplicate_summary, write_report


def run_step1(input_dir=None):
    """Step 1: load, drop the first column, label from file name, combine, summarise."""
    config.RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    config.DOCS_DIR.mkdir(parents=True, exist_ok=True)
    df = build_dataset(input_dir)
    per_year = counts_per_year(df)
    per_file, cross, totals = duplicate_summary(df)
    per_year.to_csv(config.RESULTS_DIR / "step1_counts_per_year.csv")
    per_file.to_csv(config.RESULTS_DIR / "step1_duplicates_per_file.csv")
    write_report(df, per_year, per_file, cross, totals, config.DOCS_DIR / "step1_data_summary.md")
    print(per_year.to_string())
    print(cross)
    return df


def run_step2(input_dir=None):
    """Step 2: measure feature drift across all years (descriptive only; see PLAN.md)."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import pandas as pd

    for d in (config.RESULTS_DIR, config.DOCS_DIR, config.OUTPUTS_DIR):
        d.mkdir(parents=True, exist_ok=True)
    df = build_dataset(input_dir)
    names = feature_columns(df)
    X = df[names].to_numpy()                       # read-only use; df is never modified
    years = df[config.YEAR_COLUMN].to_numpy()
    labels = df[config.LABEL_COLUMN].to_numpy()
    year_list = [int(y) for y in sorted(np.unique(years))]
    all_pairs = list(drift.combinations(year_list, 2))
    consecutive = list(zip(year_list[:-1], year_list[1:]))
    vs_first = [(year_list[0], y) for y in year_list[1:]]
    # Validate the year structure against the split policy before anything else, so a
    # year range that cannot support the train/test boundary fails here (review R7).
    train_years, test_years = config.split_years(year_list)
    print(f"years {year_list}: training {train_years}, test {test_years}")

    # 0. What the data actually holds per year: rows, distinct feature vectors, distinct
    #    app identifiers and class balance. The report quotes these instead of assuming
    #    row counts or a 50/50 split (review R3/R4).
    stats = drift.year_stats(X, years, labels, df[config.ID_COLUMN].to_numpy())
    stats.to_csv(config.RESULTS_DIR / "step2_year_stats.csv")

    # 1. Per-feature drift for every year pair and class (BH within year pair x class).
    table = drift.feature_drift(X, years, labels, names, all_pairs)
    table.to_csv(config.OUTPUTS_DIR / "step2_feature_drift.csv.gz", index=False)
    summary = drift.pair_summary(table)
    summary.to_csv(config.RESULTS_DIR / "step2_pair_summary.csv", index=False)
    for cls in drift.CLASSES:
        suffix = "" if cls == "all" else f"_{cls}"
        drift.symmetric_matrix(summary, "mean_ks", cls, year_list).to_csv(
            config.RESULTS_DIR / f"step2_ks_matrix{suffix}.csv")
    drift.symmetric_matrix(summary, "mean_js", "all", year_list).to_csv(
        config.RESULTS_DIR / "step2_js_matrix.csv")

    # 2. Top drifting features for the first-year and year-to-year comparisons.
    wanted = set(consecutive) | set(vs_first)
    sub = table[np.array([(a, b) in wanted for a, b in zip(table["year_a"], table["year_b"])])]
    top = drift.top_features(sub, k=10)
    top["short_name"] = top["feature"].map(drift.short_name)
    top.to_csv(config.RESULTS_DIR / "step2_top_features.csv", index=False)

    # 3. Feature-level summary (all apps): mean KS over consecutive pairs and first vs last year.
    is_consecutive = np.array([(a, b) in set(consecutive)
                               for a, b in zip(table["year_a"], table["year_b"])])
    cons = table[(table["class"] == "all").to_numpy() & is_consecutive]
    first_last = table[(table["class"] == "all") & (table["year_a"] == year_list[0])
                       & (table["year_b"] == year_list[-1])].set_index("feature")
    feat = cons.groupby("feature", sort=False)["ks"].mean().rename("mean_ks_consecutive").to_frame()
    feat["ks_first_last"] = first_last["ks"]
    feat["prevalence_first"] = first_last["prevalence_a"]
    feat["prevalence_last"] = first_last["prevalence_b"]
    feat["js_first_last"] = first_last["js_distance"]
    feat["q_first_last"] = first_last["q_value"]
    feat = feat.sort_values("mean_ks_consecutive", ascending=False)
    feat.insert(0, "short_name", [drift.short_name(f) for f in feat.index])
    feat.to_csv(config.RESULTS_DIR / "step2_feature_summary.csv")

    # 4. Sensitivity check: the same year-to-year comparison with identical feature vectors
    #    collapsed inside each (year, class), because Step 1 found many repeated rows.
    #    This is feature-vector deduplication, not application deduplication (review R2).
    keep = np.zeros(len(df), dtype=bool)
    for (y, lab), idx in pd.Series(np.arange(len(df))).groupby([years, labels]):
        keep[drift.unique_vector_index(X, idx.to_numpy())] = True
    dedup = drift.pair_summary(drift.feature_drift(X, years, labels, names, consecutive, keep))
    dedup.to_csv(config.RESULTS_DIR / "step2_dedup_sensitivity.csv", index=False)

    # 5. Temporal distinguishability (year classifier). Descriptive, not concept drift.
    #    Identical feature vectors are grouped across the two compared years and the split
    #    is group-aware and repeated, so no vector straddles train/test (review R1/R6).
    clf = [drift.year_classifier(X, years, labels, all_pairs, "all")]
    for cls in ("benign", "malware"):
        clf.append(drift.year_classifier(X, years, labels, consecutive + vs_first, cls))
    clf = pd.concat(clf, ignore_index=True).drop_duplicates(["year_a", "year_b", "class"])
    clf.to_csv(config.RESULTS_DIR / "step2_year_classifier.csv", index=False)
    if clf["skipped"].any():
        print("year classifier: comparisons not run ->")
        print(clf[clf["skipped"]][["year_a", "year_b", "class", "skip_reason"]].to_string(index=False))

    # 6. Heatmaps over all years (all apps): mean KS and mean Jensen-Shannon distance.
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.6))
    for ax, value, title in ((axes[0], "mean_ks", "Mean KS statistic"),
                             (axes[1], "mean_js", "Mean Jensen-Shannon distance")):
        mat = drift.symmetric_matrix(summary, value, "all", year_list)
        im = ax.imshow(mat.to_numpy(), cmap="viridis")
        ax.set_xticks(range(len(year_list)), year_list, rotation=45)
        ax.set_yticks(range(len(year_list)), year_list)
        for i in range(len(year_list)):
            for j in range(len(year_list)):
                v = mat.to_numpy()[i, j]
                ax.text(j, i, f"{v:.3f}", ha="center", va="center", fontsize=6,
                        color="white" if v < mat.to_numpy().max() * 0.6 else "black")
        ax.set_title(f"{title} over {len(names):,} features (all apps)")
        fig.colorbar(im, ax=ax, fraction=0.046)
    fig.tight_layout()
    fig.savefig(config.RESULTS_DIR / "step2_drift_heatmap.png", dpi=150)
    plt.close(fig)

    # 7. Class cancellation, then the report. Every claim in the report is taken from
    #    these tables (review R3/R8).
    cancel = drift.class_cancellation(table, consecutive)
    cancel.to_csv(config.RESULTS_DIR / "step2_class_cancellation.csv", index=False)
    drift.write_report(summary, feat, top, clf, dedup, year_list,
                       config.DOCS_DIR / "step2_drift_analysis.md", cancel, stats)
    print(summary[summary["class"] == "all"].to_string())
    return table


def run_step3(input_dir=None):
    """Step 3: split by year, then fit the preprocessing pipeline on training years only.

    Nothing is fitted on a test year. The pipeline (constant-column removal + scaling) is
    saved so Step 4 can wrap each classifier in it instead of transforming once up front.
    """
    import joblib
    import pandas as pd

    for d in (config.RESULTS_DIR, config.DOCS_DIR, config.OUTPUTS_DIR):
        d.mkdir(parents=True, exist_ok=True)
    df = build_dataset(input_dir)

    # Everything that can fail is validated first; the artifacts and the report are only
    # written at the end, so a failed run cannot leave Step 4 with half-updated files
    # (review R13).

    # 1. Time-based split by year (first half trains, later half tests). check_split
    #    verifies the two year lists partition the dataset's years, that every test year
    #    is later than every training year, and that no row is lost (review R2).
    split = preprocessing.split_by_year(df)
    summary = preprocessing.split_summary(df, split)
    print(f"train years {split.train_years} ({len(split.train):,} rows), "
          f"test years {split.test_years} ({len(split.test):,} rows)")

    # 2. Feature matrices with no metadata column (apkname, year, label, source_file),
    #    and the same features in the same order on both sides (review R4). The test
    #    matrix is required to match the training columns exactly.
    X_train, y_train, years_train = preprocessing.features_labels(split.train)
    X_test, y_test, years_test = preprocessing.features_labels(
        split.test, expected_columns=X_train.columns)
    preprocessing.check_feature_contract(X_train, X_test)
    #    The labels and years travel separately from X, so confirm they describe these
    #    rows and hold only the known label values before fitting anything (review R15).
    preprocessing.check_labels_years(split, y_train, years_train, y_test, years_test)

    # 3. Forward-chaining folds are built before the fit, so a year range that cannot
    #    support time-ordered validation fails before any work is done (review R7).
    folds = preprocessing.forward_chaining_folds(split.train_years)
    preprocessing.fold_indices(years_train, folds)   # fails here if a fold would be empty
    print("validation folds: " + "; ".join(
        f"train {f_train} -> validate {f_val}" for f_train, f_val in folds))

    # 4. Fit the pipeline on the training years ONLY, then read back what it keeps.
    #    kept_features raises if every feature was constant (review R3).
    pipeline = preprocessing.fit_preprocessor(
        preprocessing.make_preprocessor(scale=True), X_train, split.train_years)
    kept = preprocessing.kept_features(pipeline, split.train_years)
    dropped = [c for c in X_train.columns if c not in set(kept)]
    preprocessing.check_feature_lists(X_train, X_test, kept, dropped)
    print(f"features: {X_train.shape[1]} -> {len(kept)} kept, {len(dropped)} constant in "
          f"the training years and dropped")

    # 5. Verify the scaler stores the training set's own statistics, and that transforming
    #    the training rows centres and scales them. Both numbers go into the report, which
    #    states what each one does and does not establish (review R6).
    scaler_check = preprocessing.verify_scaler(pipeline, X_train)
    if scaler_check is not None:
        print(f"scaler check: max |mean difference| {scaler_check['max_abs_mean_difference']:.3e}, "
              f"max |scale difference| {scaler_check['max_abs_scale_difference']:.3e}")
    #    Taken on ALL training rows, not a subsample: a subsample of sparse count features
    #    misses their rare large values, so its standard deviation sits below 1 even when
    #    the scaler is correct, which would read as a failed check.
    transformed = pipeline.transform(X_train)
    sample_stats = {"n_rows": len(X_train), "mean": float(transformed.mean(axis=0).mean()),
                    "std": float(transformed.std(axis=0).mean())}
    del transformed
    print(f"transformed training rows: average feature mean {sample_stats['mean']:+.3f}, "
          f"average feature std {sample_stats['std']:.3f}")

    # 6. Fit one pipeline per distinct preprocessing configuration, so the saved artifacts
    #    match the per-model scaling policy instead of implying every model is scaled
    #    (review R1). These are diagnostic artifacts: Step 4 builds fresh pipelines with
    #    preprocessor_for() and refits them inside each fold (review R14).
    artifacts = preprocessing.artifact_configs()
    fitted = {}
    for name, (scale, models) in artifacts.items():
        fitted[name] = (pipeline if scale else preprocessing.fit_preprocessor(
            preprocessing.make_preprocessor(scale=False), X_train, split.train_years))
        if preprocessing.kept_features(fitted[name], split.train_years) != kept:
            raise ValueError(f"run_step3: the {name} pipeline kept a different feature set "
                             f"than the scaled one; the dropped-column step must not depend "
                             f"on scaling")
        print(f"artifact {name}: scale={scale}, models {', '.join(models)}")

    # 7. Validate the report's inputs, then write every file (review R8/R13).
    preprocessing.check_report_inputs(summary, split, total_rows=len(df))
    summary.to_csv(config.RESULTS_DIR / "step3_split_summary.csv", index=False)
    for name, pipe in fitted.items():
        joblib.dump(pipe, config.OUTPUTS_DIR / f"step3_preprocessor_{name}.joblib")
    pd.DataFrame([{"model": model, "artifact": f"step3_preprocessor_{name}.joblib",
                   "scaled": scale, "reason": preprocessing.SCALING_BY_MODEL[model][1]}
                  for name, (scale, models) in artifacts.items() for model in models]).to_csv(
        config.OUTPUTS_DIR / "step3_preprocessor_models.csv", index=False)
    pd.DataFrame({"feature": kept}).to_csv(
        config.OUTPUTS_DIR / "step3_kept_features.csv", index=False)
    pd.DataFrame({"feature": dropped,
                  "varies_in_test_years": [bool(X_test[c].max() != X_test[c].min())
                                           for c in dropped]}).to_csv(
        config.OUTPUTS_DIR / "step3_dropped_features.csv", index=False)
    preprocessing.write_report(split, summary, pipeline, X_train, X_test, dropped,
                               scaler_check, folds, sample_stats,
                               config.DOCS_DIR / "step3_preprocessing.md", artifacts)
    print(summary.to_string(index=False))
    return split, pipeline


def run_step4(input_dir=None):
    """Step 4: choose hyperparameters on training-year folds and fit each classifier.

    Reuses Step 3's split, feature contract and folds. No test-year row is loaded into
    the search, scored, or used to choose anything; prediction is Step 5 and Step 6.
    """
    from src import models

    for d in (config.RESULTS_DIR, config.DOCS_DIR, config.OUTPUTS_DIR):
        d.mkdir(parents=True, exist_ok=True)
    df = build_dataset(input_dir)

    # 1. The same validated split and feature contract as Step 3.
    split = preprocessing.split_by_year(df)
    summary = preprocessing.split_summary(df, split)
    X_train, y_train, years_train = preprocessing.features_labels(split.train)
    X_test, y_test, years_test = preprocessing.features_labels(
        split.test, expected_columns=X_train.columns)
    preprocessing.check_feature_contract(X_train, X_test)
    preprocessing.check_labels_years(split, y_train, years_train, y_test, years_test)
    del X_test, y_test, years_test   # Step 4 must not touch the test years at all
    print(f"training on {split.train_years} ({len(X_train):,} rows, "
          f"{X_train.shape[1]:,} features before preprocessing)")

    # 2. Forward-chaining folds inside the training years, validated by models.check_folds
    #    before any model is fitted.
    folds = preprocessing.forward_chaining_folds(split.train_years)
    cv = preprocessing.fold_indices(years_train, folds)
    print("validation folds: " + "; ".join(
        f"train {f_train} -> validate {f_val}" for f_train, f_val in folds))

    # 3. Search each model's small grid, refit the winner on the whole training half,
    #    save the models and write the report.
    best, cv_table = models.run(X_train, y_train, years_train, split, summary, cv, folds,
                                config.RESULTS_DIR, config.OUTPUTS_DIR / "models",
                                config.DOCS_DIR)
    print(best[["model", "best_params", "cv_accuracy_mean", "cv_f1_mean",
                "fit_seconds", "reproducible"]].to_string(index=False))
    return best, cv_table


def run_step5(input_dir=None):
    """Step 5: evaluate each fitted classifier on the held-out test years."""
    return evaluation.run(input_dir)


def run_step6(input_dir=None):
    """Step 6: predict on held-out test data and compare results by year."""
    return predictions.run(input_dir)


def run_step7(input_dir=None):
    """Step 7: compare frozen, periodically retrained, and incremental models."""
    return sustainable.run(input_dir)


def main():
    parser = argparse.ArgumentParser(description="Project 2 pipeline")
    parser.add_argument("--step", type=int, choices=[1, 2, 3, 4, 5, 6, 7], help="run one step")
    parser.add_argument("--all", action="store_true",
                        help="run all implemented steps (currently 1 to 7)")
    parser.add_argument("--input-dir", default=None, help="folder with the CSV files")
    args = parser.parse_args()
    if args.all or args.step == 1:
        run_step1(args.input_dir)
    if args.all or args.step == 2:
        run_step2(args.input_dir)
    if args.all or args.step == 3:
        run_step3(args.input_dir)
    if args.all or args.step == 4:
        run_step4(args.input_dir)
    if args.all or args.step == 5:
        run_step5(args.input_dir)
    if args.all or args.step == 6:
        run_step6(args.input_dir)
    if args.all or args.step == 7:
        run_step7(args.input_dir)
    if not args.all and args.step is None:
        parser.print_help()


if __name__ == "__main__":
    main()
