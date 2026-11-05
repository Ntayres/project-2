"""Step 3: split the data by year and build the preprocessing pipeline.

The split is by year, never a random shuffle across years (PLAN.md year split policy):
the first half of the years trains, the later half tests. config.split_years decides the
boundary, so no module picks it with a positional index.

Everything that is *learned* from data lives inside a scikit-learn Pipeline:
  * `drop_constant` (VarianceThreshold) removes features that never vary, and
  * `scale` (StandardScaler) centres and scales them.
Keeping both inside the pipeline is what prevents leakage (PLAN.md leakage rule 1 and
review R5 on the plan): when Step 4 refits the pipeline on a forward-chaining fold, the
dropped-column set and the scaler statistics come from that fold's training years alone.
Nothing here ever sees a test year during fitting.

Feature names and their order are preserved: the pipeline is fitted on a DataFrame, so the
retained columns can be read back by name and applied unchanged to the test years.

Contract for Step 4 (review R14). The pipelines saved by Step 3 in `outputs/` are
**diagnostic artifacts**: they record what fitting on the training years produced, for
inspection and for the report. Step 4 must NOT load one and transform everything before
model selection, because that would fit preprocessing once on all four training years and
then reuse it inside every validation fold. Instead Step 4 builds a fresh pipeline per
model with `preprocessor_for(model_name)`, puts the classifier in the same Pipeline, and
passes `fold_indices(...)` as `cv`, so each fold refits the dropped-column set and the
scaler on that fold's training years alone and only transforms its validation year.
"""
from collections import namedtuple

import numpy as np
import pandas as pd
from sklearn.feature_selection import VarianceThreshold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from src import config
from src.data_loading import feature_columns

# Which models get StandardScaler, and why. Step 4 reads this so the choice is recorded
# once instead of being repeated per model (PLAN.md Step 3, item 4).
SCALING_BY_MODEL = {
    "LogisticRegression": (True, "regularised linear model: the penalty treats all "
                                 "coefficients alike, so features must share a scale"),
    "LinearSVC": (True, "margin depends on distances, so unscaled count ranges dominate"),
    "SGDClassifier": (True, "gradient steps need comparable feature scales to converge"),
    "RandomForestClassifier": (False, "splits are on thresholds per feature; monotone "
                                      "rescaling does not change the tree"),
    "DummyClassifier": (False, "predicts a constant class and never reads the features, so "
                               "scaling cannot change its output"),
}

Split = namedtuple("Split", "train test train_years test_years")

# Forward-chaining validation needs an earlier year to train on and a later one to
# validate on, so the training half cannot be a single year (review R7).
MIN_TRAIN_YEARS = 2


def split_by_year(df):
    """Split the combined dataset into training and test years.

    Returns a Split with the two frames and the two year lists. The `year` and `label`
    columns are kept on both frames so later steps can report per year; they are removed
    from the feature matrix by features_labels(), not here.
    """
    years = sorted(int(y) for y in df[config.YEAR_COLUMN].unique())
    train_years, test_years = config.split_years(years)
    year_col = df[config.YEAR_COLUMN]
    split = Split(train=df[year_col.isin(train_years)].copy(),
                  test=df[year_col.isin(test_years)].copy(),
                  train_years=train_years, test_years=test_years)
    check_split(split, all_years=years, total_rows=len(df))
    return split


def check_split(split, all_years=None, total_rows=None):
    """Fail loudly if the split could leak, lose rows, or misdescribe the year partition.

    Given `all_years` (every year in the dataset being split), this establishes that the
    two declared year lists *partition* those years: each year is on exactly one side,
    none is omitted, and none is claimed that the dataset does not have (review R2). A
    matching row count alone would not establish that. Also checks that every test year
    is later than every training year, so the split is time-based and not a shuffle, and
    that the frames really hold the years they claim.
    """
    overlap = set(split.train_years) & set(split.test_years)
    if overlap:
        raise ValueError(f"check_split: year(s) {sorted(overlap)} are in both train and test")
    if not split.train_years or not split.test_years:
        raise ValueError("check_split: one side of the split is empty")
    if len(split.train_years) < MIN_TRAIN_YEARS:
        raise ValueError(f"check_split: {len(split.train_years)} training year(s) "
                         f"({split.train_years}); forward-chaining validation needs at least "
                         f"{MIN_TRAIN_YEARS}")
    if min(split.test_years) <= max(split.train_years):
        raise ValueError(f"check_split: test years {split.test_years} are not all later than "
                         f"training years {split.train_years}")
    if all_years is not None:
        all_years = sorted(int(y) for y in all_years)
        declared = sorted(int(y) for y in list(split.train_years) + list(split.test_years))
        missing = sorted(set(all_years) - set(declared))
        extra = sorted(set(declared) - set(all_years))
        if missing:
            raise ValueError(f"check_split: year(s) {missing} of the dataset are in neither "
                             f"split")
        if extra:
            raise ValueError(f"check_split: year(s) {extra} are declared but are not in the "
                             f"dataset ({all_years})")
        if declared != all_years:
            raise ValueError(f"check_split: declared years {declared} do not partition the "
                             f"dataset's years {all_years}")
    for name, frame, expected in (("train", split.train, split.train_years),
                                  ("test", split.test, split.test_years)):
        found = sorted(int(y) for y in frame[config.YEAR_COLUMN].unique())
        if found != sorted(expected):
            raise ValueError(f"check_split: the {name} frame holds years {found}, "
                             f"expected {sorted(expected)}")
    if total_rows is not None and len(split.train) + len(split.test) != total_rows:
        raise ValueError(f"check_split: {len(split.train)} + {len(split.test)} rows do not add "
                         f"up to the {total_rows} rows loaded")


def features_labels(df, expected_columns=None):
    """Return (X, y, years): X holds only the API-call features, in their file order.

    The feature contract this enforces (review R4), because Step 4 reuses the column
    order and a silent mismatch there would misalign every coefficient:
      * no metadata column (apkname, year, label, source_file) is present;
      * the column count is config.EXPECTED_FEATURES;
      * column names are unique;
      * every column is a numeric dtype, and no float column holds a non-finite value
        (integer dtypes cannot, so they are not scanned);
      * counts are non-negative;
      * if `expected_columns` is given, the names match it exactly, in the same order.
    """
    X = df[feature_columns(df)]
    leaked = [c for c in config.METADATA_COLUMNS + ["source_file"] if c in X.columns]
    if leaked:
        raise ValueError(f"features_labels: metadata column(s) {leaked} in the feature matrix")
    if X.shape[1] != config.EXPECTED_FEATURES:
        raise ValueError(f"features_labels: {X.shape[1]} feature columns, "
                         f"expected {config.EXPECTED_FEATURES}")
    if X.columns.has_duplicates:
        repeated = X.columns[X.columns.duplicated()].unique().tolist()
        raise ValueError(f"features_labels: duplicate feature name(s), e.g. {repeated[:3]}")
    non_numeric = [c for c, t in X.dtypes.items() if not pd.api.types.is_numeric_dtype(t)]
    if non_numeric:
        raise ValueError(f"features_labels: non-numeric feature column(s), e.g. "
                         f"{non_numeric[:3]}")
    float_cols = [c for c, t in X.dtypes.items() if pd.api.types.is_float_dtype(t)]
    if float_cols and not np.isfinite(X[float_cols].to_numpy()).all():
        raise ValueError("features_labels: non-finite values in the feature matrix")
    minimum = X.min().min()
    if minimum < 0:
        raise ValueError(f"features_labels: negative count ({minimum}) in the feature matrix")
    if expected_columns is not None and list(X.columns) != list(expected_columns):
        raise ValueError("features_labels: feature names or their order do not match the "
                         "expected columns")
    return X, df[config.LABEL_COLUMN], df[config.YEAR_COLUMN]


def check_feature_contract(X_train, X_test):
    """The two matrices must offer the same features in the same order (review R4)."""
    if list(X_train.columns) != list(X_test.columns):
        train_only = [c for c in X_train.columns if c not in set(X_test.columns)]
        test_only = [c for c in X_test.columns if c not in set(X_train.columns)]
        detail = (f"train-only {train_only[:3]}, test-only {test_only[:3]}"
                  if train_only or test_only else "same names in a different order")
        raise ValueError(f"check_feature_contract: train and test feature columns differ "
                         f"({detail})")


def check_labels_years_side(frame, y, years, expected_years, name="train"):
    """Check one side of the split: labels and years must describe `frame`'s own rows.

    Separated from check_labels_years so a step that must not build the test matrix at
    all can still validate its training side (Step 4 review R1).
    """
    if len(y) != len(frame) or len(years) != len(frame):
        raise ValueError(f"check_labels_years: the {name} labels/years have "
                         f"{len(y)}/{len(years)} rows, expected {len(frame)}")
    if not y.equals(frame[config.LABEL_COLUMN]) or not years.equals(frame[config.YEAR_COLUMN]):
        raise ValueError(f"check_labels_years: the {name} labels or years do not match "
                         f"the {name} frame")
    unknown = sorted(set(y.unique()) - set(config.LABEL_MAP.values()))
    if unknown:
        raise ValueError(f"check_labels_years: unexpected label value(s) {unknown} in the "
                         f"{name} rows; expected {sorted(config.LABEL_MAP.values())}")
    found_years = sorted(int(v) for v in years.unique())
    if found_years != sorted(expected_years):
        raise ValueError(f"check_labels_years: the {name} years are {found_years}, "
                         f"expected {sorted(expected_years)}")


def check_labels_years(split, y_train, years_train, y_test, years_test):
    """The labels and years returned with both matrices must match the split frames.

    features_labels() returns them as separate objects, so this confirms they were taken
    from the rows they claim to describe before anything is fitted (review R15).
    """
    check_labels_years_side(split.train, y_train, years_train, split.train_years, "train")
    check_labels_years_side(split.test, y_test, years_test, split.test_years, "test")


def make_preprocessor(scale=True):
    """Build the preprocessing pipeline: drop never-varying features, then scale.

    Both steps are estimators inside the Pipeline on purpose, so refitting on a subset of
    years refits them on that subset only (review R5 on the plan). `scale=False` is for
    models that do not need it, such as Random Forest; see SCALING_BY_MODEL.
    """
    steps = [("drop_constant", VarianceThreshold(threshold=0.0))]
    if scale:
        # with_mean=True is safe here: the matrix is dense, not sparse.
        steps.append(("scale", StandardScaler()))
    return Pipeline(steps)


def preprocessor_for(model_name):
    """Fresh, unfitted pipeline for a named model, per the recorded scaling decision.

    Step 4 calls this once per model and lets the cross-validation refit it on each fold
    (review R14); it must not load a pipeline Step 3 already fitted and transform
    everything up front. See the module docstring for the full contract.
    """
    if model_name not in SCALING_BY_MODEL:
        raise KeyError(f"preprocessor_for: no scaling decision recorded for {model_name!r}; "
                       f"known models: {sorted(SCALING_BY_MODEL)}")
    return make_preprocessor(scale=SCALING_BY_MODEL[model_name][0])


def artifact_configs():
    """The distinct preprocessing configurations to fit and save, and which models use each.

    SCALING_BY_MODEL has four models but only two configurations, so Step 3 saves two
    artifacts rather than duplicating one of them twice (review R1).
    """
    configs = {}
    for name, (scale, _) in SCALING_BY_MODEL.items():
        configs.setdefault(bool(scale), []).append(name)
    return {("scaled" if scale else "unscaled"): (scale, models)
            for scale, models in sorted(configs.items(), reverse=True)}


def fit_preprocessor(pipeline, X_train, train_years=None):
    """Fit the pipeline on the training rows, naming the years if nothing survives.

    When every feature is constant, `VarianceThreshold` raises during *fit* ("No feature
    in X meets the variance threshold"), before any retained-feature list can be read.
    That message does not say which rows were involved, so it is re-raised here with the
    training years and the feature count (review R3). kept_features() keeps an equivalent
    check for a pipeline that returns an empty selection instead of raising.
    """
    try:
        return pipeline.fit(X_train)
    except ValueError as exc:
        if "variance threshold" not in str(exc).lower():
            raise
        where = f" in the training years {list(train_years)}" if train_years else ""
        raise ValueError(f"fit_preprocessor: all {X_train.shape[1]} features are constant"
                         f"{where}, so the pipeline retains no feature to scale or train "
                         f"on") from exc


def kept_features(pipeline, train_years=None):
    """Names of the features the fitted pipeline keeps, in order.

    Raises when the pipeline kept nothing, which happens if every feature is constant in
    the years it was fitted on. Failing here, with the years named, is clearer than a
    shape error from a classifier or an empty-array error in the report (review R3).
    """
    kept = list(pipeline.named_steps["drop_constant"].get_feature_names_out())
    if not kept:
        where = f" in the training years {list(train_years)}" if train_years else ""
        raise ValueError(f"kept_features: the pipeline retained no features{where}: every "
                         "feature was constant, so there is nothing to scale or train on")
    return kept


def check_feature_lists(X_train, X_test, kept, dropped):
    """The kept and dropped lists must account for the feature set exactly (review R10).

    Step 4 consumes these lists, so this checks they are disjoint, that together they are
    the training columns, that every dropped feature also exists in the test matrix, and
    that the kept order is the training order rather than an arbitrary one.
    """
    train_cols, test_cols = list(X_train.columns), set(X_test.columns)
    kept_set, dropped_set = set(kept), set(dropped)
    if kept_set & dropped_set:
        raise ValueError(f"check_feature_lists: {len(kept_set & dropped_set)} feature(s) are in "
                         f"both the kept and dropped lists, e.g. "
                         f"{sorted(kept_set & dropped_set)[:3]}")
    if kept_set | dropped_set != set(train_cols):
        unaccounted = sorted(set(train_cols) - (kept_set | dropped_set))
        unknown = sorted((kept_set | dropped_set) - set(train_cols))
        raise ValueError(f"check_feature_lists: kept + dropped do not account for the training "
                         f"columns (unaccounted {unaccounted[:3]}, unknown {unknown[:3]})")
    missing_in_test = [c for c in dropped if c not in test_cols]
    if missing_in_test:
        raise ValueError(f"check_feature_lists: dropped feature(s) absent from the test matrix, "
                         f"e.g. {missing_in_test[:3]}")
    if kept != [c for c in train_cols if c in kept_set]:
        raise ValueError("check_feature_lists: the kept features are not in training-column "
                         "order")


def constant_columns(X):
    """Names of columns that never vary in X (max == min), cheaper than a full variance."""
    spread = X.max() - X.min()
    return list(spread.index[spread == 0])


def forward_chaining_folds(train_years):
    """Time-ordered validation folds inside the training years.

    [(Y1,), Y2], [(Y1, Y2), Y3], ... Each fold trains on earlier years and validates on
    the next one, so a fold never validates on a year it trained on and never touches a
    test year (PLAN.md: no random k-fold, which would mix years).
    """
    train_years = sorted(int(y) for y in train_years)
    if len(train_years) < 2:
        raise ValueError(f"forward_chaining_folds: need at least 2 training years, "
                         f"got {train_years}")
    return [(train_years[:k], train_years[k]) for k in range(1, len(train_years))]


def fold_indices(years, folds):
    """Turn year folds into (train_idx, validation_idx) positions for scikit-learn `cv`.

    Positions are relative to `years` as passed in, so this must be given the training
    frame's year column, not the whole dataset's.
    """
    years = np.asarray(years)
    out = []
    for fold_train_years, val_year in folds:
        train_idx = np.flatnonzero(np.isin(years, fold_train_years))
        val_idx = np.flatnonzero(years == val_year)
        if len(train_idx) == 0 or len(val_idx) == 0:
            raise ValueError(f"fold_indices: fold {fold_train_years} -> {val_year} is empty "
                             f"for the years given")
        out.append((train_idx, val_idx))
    return out


def split_summary(df, split):
    """Rows and class balance per year, with the split each year belongs to.

    Refuses to summarise a label column holding anything other than the known label
    values: an unexpected label would be counted as neither benign nor malware, and the
    report would then show class counts that do not add up to the rows (review R9).
    """
    labels_all = df[config.LABEL_COLUMN]
    unknown = sorted(set(labels_all.unique()) - set(config.LABEL_MAP.values()))
    if unknown:
        n_unknown = int((~labels_all.isin(config.LABEL_MAP.values())).sum())
        raise ValueError(f"split_summary: {n_unknown} row(s) carry label value(s) {unknown}; "
                         f"expected only {sorted(config.LABEL_MAP.values())} "
                         f"({config.LABEL_MAP})")
    rows = []
    for year in sorted(int(y) for y in df[config.YEAR_COLUMN].unique()):
        part = df[df[config.YEAR_COLUMN] == year]
        labels = part[config.LABEL_COLUMN]
        n_malware = int((labels == config.LABEL_MAP["malware"]).sum())
        n_benign = int((labels == config.LABEL_MAP["benign"]).sum())
        if n_benign + n_malware != len(part):
            raise ValueError(f"split_summary: year {year} has {len(part)} rows but "
                             f"{n_benign} + {n_malware} classified")
        rows.append({"year": year,
                     "split": "train" if year in split.train_years else "test",
                     "rows": len(part),
                     "benign": n_benign,
                     "malware": n_malware,
                     "malware_share": n_malware / len(part)})
    return pd.DataFrame(rows)


def check_report_inputs(summary, split, total_rows=None):
    """The report reads totals by split label, so make sure they are really there.

    Validates that both 'train' and 'test' appear, that each side has rows, that the
    per-year benign and malware counts add up, that the years match the split, and that
    the summary covers the whole dataset. Without this the report can fail with a bare
    KeyError from totals.loc['train'] (review R8).
    """
    missing = [s for s in ("train", "test") if s not in set(summary["split"])]
    if missing:
        raise ValueError(f"check_report_inputs: the split summary has no {missing} row(s)")
    totals = summary.groupby("split")["rows"].sum()
    for side in ("train", "test"):
        if totals.loc[side] <= 0:
            raise ValueError(f"check_report_inputs: the {side} split has {totals.loc[side]} rows")
    bad = summary[summary["benign"] + summary["malware"] != summary["rows"]]
    if len(bad):
        raise ValueError(f"check_report_inputs: benign + malware do not add up to rows for "
                         f"year(s) {bad['year'].tolist()}")
    for side, expected in (("train", split.train_years), ("test", split.test_years)):
        found = sorted(int(y) for y in summary[summary["split"] == side]["year"])
        if found != sorted(expected):
            raise ValueError(f"check_report_inputs: the summary's {side} years {found} do not "
                             f"match the split's {sorted(expected)}")
    if total_rows is not None and int(summary["rows"].sum()) != total_rows:
        raise ValueError(f"check_report_inputs: the summary covers {int(summary['rows'].sum())} "
                         f"rows, expected {total_rows}")


def verify_scaler(pipeline, X_train):
    """Check the scaler's statistics really are the training set's own.

    Recomputes the mean and the population standard deviation of the retained columns
    from X_train and compares them with what the fitted StandardScaler holds. Returns the
    largest absolute difference for each, so the report can quote a number instead of
    claiming the property. StandardScaler divides by the population standard deviation
    (ddof=0), and it stores scale_=1 for a column whose variance is 0.
    """
    if "scale" not in pipeline.named_steps:
        return None
    scaler = pipeline.named_steps["scale"]
    kept = kept_features(pipeline)   # raises if the pipeline retained nothing (R3/R5)
    for attr in ("mean_", "scale_"):
        values = getattr(scaler, attr)
        if len(values) != len(kept):
            raise ValueError(f"verify_scaler: the scaler's {attr} has {len(values)} entries but "
                             f"the pipeline retained {len(kept)} features")
    expected_mean = X_train[kept].mean().to_numpy(dtype=float)
    expected_scale = X_train[kept].std(ddof=0).to_numpy(dtype=float)
    expected_scale = np.where(expected_scale == 0, 1.0, expected_scale)
    return {"max_abs_mean_difference": float(np.abs(scaler.mean_ - expected_mean).max()),
            "max_abs_scale_difference": float(np.abs(scaler.scale_ - expected_scale).max())}


def write_report(split, summary, pipeline, X_train, X_test, dropped, scaler_check,
                 folds, sample_stats, path, artifacts=None):
    """Write docs/step3_preprocessing.md. Every number comes from the objects passed in."""
    path.parent.mkdir(parents=True, exist_ok=True)
    artifacts = artifacts if artifacts is not None else artifact_configs()
    check_report_inputs(summary, split)
    kept = kept_features(pipeline, split.train_years)
    train_years, test_years = split.train_years, split.test_years
    # Features with no variation in the training years that DO vary in the test years:
    # they are dropped on training evidence, which is the point of fitting on train only.
    dropped_active_in_test = [c for c in dropped
                              if X_test[c].max() != X_test[c].min()]
    col_means = X_train.mean()
    totals = summary.groupby("split")[["rows", "benign", "malware"]].sum()
    t_split = summary.assign(malware_share=summary["malware_share"].map(lambda v: f"{100*v:.1f}%"))
    fold_lines = [f"  {i}. train on {', '.join(map(str, f_train))} -> validate on {f_val}"
                  for i, (f_train, f_val) in enumerate(folds, start=1)]
    scaling_rows = pd.DataFrame(
        [{"model": name, "StandardScaler": "yes" if use else "no", "reason": why}
         for name, (use, why) in SCALING_BY_MODEL.items()])

    L = [
        "# Step 3 Report: Preprocessing and the Train/Test Split", "",
        "Written by the builder agent. Generated by `python -m src.main --step 3`; all numbers "
        "come from code (`results/step3_split_summary.csv`, fitted pipeline and feature list in "
        "`outputs/`).", "",
        "## The split: which years train and which years test", "",
        f"The split is **by year**, not a random shuffle: **{', '.join(map(str, train_years))} "
        f"train** and **{', '.join(map(str, test_years))} test**. The boundary is the project's "
        "fixed policy (the first half of the years available train, the later half test), applied "
        "by `config.split_years`, so no module chooses it with a positional index. It was fixed "
        "before the Step 2 drift results were examined and those results did not change it.", "",
        f"- Training rows: **{int(totals.loc['train', 'rows']):,}** "
        f"({int(totals.loc['train', 'benign']):,} benign, "
        f"{int(totals.loc['train', 'malware']):,} malware).",
        f"- Test rows: **{int(totals.loc['test', 'rows']):,}** "
        f"({int(totals.loc['test', 'benign']):,} benign, "
        f"{int(totals.loc['test', 'malware']):,} malware).",
        f"- Together that is {int(totals['rows'].sum()):,} rows, the whole dataset loaded in "
        "Step 1, so no row is dropped or counted twice.", "",
        t_split.to_markdown(index=False), "",
        "The test years are all later than every training year, which is what makes this a "
        "time-based split: the models are asked to do what a deployed detector has to do, namely "
        "judge apps from years it has never seen. Step 6 reports the test years one at a time, so "
        "the effect of the drift measured in Step 2 is visible year by year.", "",
        "## Scaling", "",
        f"The features are non-negative integer API-call counts over {config.EXPECTED_FEATURES} "
        "columns, and their ranges differ by orders of magnitude: in the training years the "
        f"largest single count is {int(X_train.max().max()):,}, while the per-feature mean count "
        f"runs from {col_means.min():.3g} to {col_means.max():.3g}. A model that treats all "
        "coefficients alike would therefore be dominated by a handful of common calls. Two "
        "preprocessing steps are applied, in this order:", "",
        "1. **Drop never-varying features** (`VarianceThreshold(threshold=0.0)`): a column that "
        "holds one value across the training years carries no information for a model and would "
        "give a zero standard deviation to divide by.",
        "2. **Standardise** (`StandardScaler`): subtract the mean and divide by the standard "
        "deviation, so every feature is on a comparable scale.", "",
        f"Of the {config.EXPECTED_FEATURES} features, **{len(dropped)} never vary in the training "
        f"years and are dropped, leaving {len(kept)}**. The same {len(kept)} columns, in the same "
        "order, are used for the test years.", "",
        (f"{len(dropped_active_in_test)} of those {len(dropped)} dropped features are constant "
         "in the training years but vary in the test years. That is expected and is not a "
         "mistake: the decision is made on training evidence only, because a deployed model "
         "cannot consult future data. What this shows is a change in feature *variability* "
         "across the split, which is consistent with the temporal changes Step 2 examined; on "
         "its own it does not establish the magnitude or the statistical significance of any "
         "feature's drift (the KS statistics and adjusted p-values in Step 2 do that). It is a "
         "reason to compare a static model with retraining in Step 7."
         if dropped_active_in_test else
         (f"None of those {len(dropped)} features vary in the test years either, so dropping them "
          "on training evidence costs nothing here."
          if dropped else
          "No feature is constant across the training years, so nothing is dropped and the full "
          "feature set reaches the scaler.")), "",
        "Scaling is not forced on every model, because not every model benefits from it:", "",
        scaling_rows.to_markdown(index=False), "",
        "Random Forest is fitted on the same dropped-column set but without the scaler, so the "
        "comparison between models differs only where the model actually requires it.", "",
        "## How leakage between years is prevented", "",
        "1. **Both learned steps live inside one `Pipeline`.** Dropping constant columns and "
        "scaling are estimators, not a one-off transformation of the whole table, so refitting the "
        "pipeline refits both on whatever rows it is given and nothing else.",
        f"2. **The pipeline is fitted on the training years only** "
        f"({', '.join(map(str, train_years))}) and applied unchanged to "
        f"{', '.join(map(str, test_years))}. The test years are transformed, never fitted on, so "
        "no test-year mean, standard deviation or column decision reaches the model.",
        "3. **Model selection uses time-ordered (forward-chaining) validation inside the training "
        "years**, never random k-fold, which would put later years in a fold that trains on them. "
        "The folds are:",
        *fold_lines,
        "   Each fold refits the pipeline on its own training years, so in the last fold nothing "
        f"is fitted on {train_years[-1]}, and no fold touches a test year.",
        "4. **No row ever crosses the boundary.** The split is taken by the `year` column with "
        "`config.split_years`, and `check_split` fails if a year appears on both sides, if a test "
        "year is not later than every training year, or if the two parts do not add up to the "
        "rows loaded.",
        "5. **Metadata cannot reach a model.** `features_labels` refuses to return a feature "
        f"matrix that still holds `{config.ID_COLUMN}`, `{config.YEAR_COLUMN}`, "
        f"`{config.LABEL_COLUMN}` or `source_file`, so the app identifier and the year itself are "
        "never features.",
        "6. **Step 2 did not influence any of this.** The drift statistics are descriptive; they "
        "were not used to choose the boundary, select features or set any parameter.", "",
        "## Checks run on this step", "",
        "These are the builder's own checks, printed by the step and repeated here; the tester "
        "agent's independent tests are in `tests/`. Each one is listed with what it does and "
        "does not establish, so no single number is read as proof that the pipeline is "
        "leakage-free.", ""]
    if scaler_check is not None:
        L += [
            "- **The fitted scaler stores the training set's own statistics.** Recomputing the "
            "mean and the population standard deviation of the retained columns from the "
            "training rows and comparing them with the fitted `StandardScaler` gives a largest "
            f"absolute difference of {scaler_check['max_abs_mean_difference']:.3e} (mean) and "
            f"{scaler_check['max_abs_scale_difference']:.3e} (scale). *Establishes:* the stored "
            "statistics are those of the training rows. *Does not establish:* that no test row "
            "influenced fitting - that follows from fitting on the training frame only, which is "
            "point 2 of the previous section, not from this comparison."]
    L += [
        f"- **Transforming the training rows gives what standardising should give.** Across the "
        f"{sample_stats['n_rows']:,} training rows, the average of the retained features' "
        f"transformed means is {sample_stats['mean']:+.3f} and the average of their transformed "
        f"standard deviations is {sample_stats['std']:.3f}. These are averages over per-feature "
        "statistics, not the mean and standard deviation of the whole matrix taken as one pool. "
        "*Establishes:* the transform does centre and scale the data it was fitted on. *Does not "
        "establish:* anything about the test years, which are deliberately not standardised to "
        "their own statistics.",
        f"- **Shapes and column identity line up.** The training matrix is {X_train.shape[0]:,} x "
        f"{X_train.shape[1]:,} before preprocessing and {X_train.shape[0]:,} x {len(kept):,} "
        f"after; the test matrix is {X_test.shape[0]:,} x {len(kept):,} after. "
        "`check_feature_contract` confirms train and test offer the same feature names in the "
        "same order, and `check_feature_lists` confirms the kept and dropped lists are disjoint "
        "and together account for every feature.",
        "- **The year lists partition the dataset.** `check_split` confirms each year is on "
        "exactly one side, that none is omitted or invented, that every test year is later than "
        "every training year, and that the two frames add up to the rows loaded in Step 1.",
        "- **Labels and years belong to the rows they describe.** `check_labels_years` confirms "
        "the label and year series returned with each matrix match that split's frame and hold "
        "only the two known label values, and `split_summary` refuses to summarise an unexpected "
        "label rather than quietly reporting counts that do not add up.", "",
        "## Files written", "",
        "Everything that can fail is validated before any of these is written, so a file present "
        "here means the step completed; a failed run leaves the previous files untouched.", "",
        "- `results/step3_split_summary.csv`: rows and class balance per year, with the split.",
        *[f"- `outputs/step3_preprocessor_{name}.joblib`: fitted "
          f"{'drop-constant + StandardScaler' if scale else 'drop-constant only'}, used by "
          f"{', '.join(models)}." for name, (scale, models) in artifacts.items()],
        "- `outputs/step3_preprocessor_models.csv`: which model maps to which artifact.",
        "- `outputs/step3_kept_features.csv`: the retained feature names, in order.",
        "- `outputs/step3_dropped_features.csv`: the dropped features, and whether each varies in "
        "the test years.", "",
        "Two pipelines are saved rather than one, because the scaling policy above has exactly "
        "two distinct configurations: the saved artifacts therefore represent the policy instead "
        "of implying that every model is scaled.", "",
        "## What Step 4 uses from here", "",
        "`split_by_year` for the frames, `features_labels` for a metadata-free feature matrix, "
        "`preprocessor_for(model_name)` for the pipeline that model needs, and "
        "`forward_chaining_folds` / `fold_indices` for the `cv` passed to model selection.", "",
        "**The saved pipelines above are diagnostic artifacts, not inputs to model selection.** "
        "Step 4 must not load one and transform the data before selecting a model: that would "
        f"fit the dropped-column set and the scaler once on all of "
        f"{', '.join(map(str, train_years))} and then reuse those statistics inside every "
        "validation fold, so each fold's validation year would have helped set them. Instead "
        "Step 4 builds a fresh pipeline per model with `preprocessor_for(model_name)`, puts the "
        "classifier in the same `Pipeline`, and passes the folds as `cv`, so scikit-learn refits "
        "the preprocessing on each fold's training years and only transforms that fold's "
        "validation year. The test years take no part in model selection at all; the final "
        "pipeline is fitted on the full training half once, after the model is chosen.", ""]
    path.write_text("\n".join(L))
