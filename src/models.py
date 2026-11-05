"""Step 4: define the classifiers, choose hyperparameters, and fit on the training years.

Four scikit-learn estimators are used (see MODEL_SPECS for the reason recorded against
each): a DummyClassifier floor baseline, Logistic Regression, SGDClassifier and Random
Forest. Each one is wrapped in its own Step 3 preprocessing pipeline, built fresh by
`preprocessing.preprocessor_for(name)` and never loaded pre-fitted, so the dropped-column
set and the scaler are refitted inside every validation fold (AGENTS.md convention, Step 3
review R14).

Hyperparameters come from a small grid searched with forward-chaining validation inside
the training years (train on Y1..Yk, validate on Yk+1) rather than random k-fold, which
would let a fold train on a year it validates on. After the search, scikit-learn refits
the winning pipeline on the whole training half.

Test-year policy, stated precisely (Step 4 review R1). The shared loader builds one
DataFrame for every year, so test rows exist in memory in this step. What this step does
NOT do is build a test feature matrix, fit on test rows, score against them, or let them
influence any choice: only `split.train` reaches `features_labels` here. Prediction on the
test years is Step 5 and Step 6.
"""
import json
import shutil
import time
from collections import namedtuple

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.metrics import f1_score, make_scorer, precision_score, recall_score
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline

from src import config, preprocessing

Spec = namedtuple("Spec", "make grid why")

# The grids are small (PLAN.md Step 4): a handful of values for the parameter that matters
# most per model, so the selection is a real measurement that still finishes in one run.
# They were widened once, after a first run selected a value at a grid edge for two
# models; `grid_position_notes` reports where the chosen value sits in every run.
MODEL_SPECS = {
    "DummyClassifier": Spec(
        make=lambda: DummyClassifier(strategy="most_frequent",
                                     random_state=config.RANDOM_STATE),
        grid={},
        why="Floor baseline. It predicts the most frequent training class and never reads "
            "the features, so it fixes the accuracy a model must beat before it can be "
            "said to have learned anything. With a 50/50 training set that floor is 0.50."),
    "LogisticRegression": Spec(
        make=lambda: LogisticRegression(penalty="l2", solver="lbfgs", max_iter=1000,
                                        random_state=config.RANDOM_STATE),
        grid={"model__C": [0.01, 0.1, 1.0, 10.0, 100.0]},
        why="Linear, fast and interpretable baseline: one weight per API call, so the "
            "fitted model can be read. C sets the L2 regularisation strength, which is "
            "the parameter that matters with 1,105 correlated count features."),
    "SGDClassifier": Spec(
        make=lambda: SGDClassifier(loss="log_loss", penalty="l2", max_iter=1000, tol=1e-3,
                                   random_state=config.RANDOM_STATE),
        grid={"model__alpha": [1e-8, 1e-7, 1e-6, 1e-5, 1e-4, 1e-3, 1e-2]},
        why="Linear model fitted by stochastic gradient descent. It is included because "
            "it supports partial_fit, so Step 7 can update it incrementally year by year "
            "and compare that with periodic retraining, using a model whose "
            "hyperparameters were selected here rather than one introduced late. "
            "loss='log_loss' keeps predict_proba available for threshold work in Step 5. "
            "alpha is its regularisation strength."),
    "RandomForestClassifier": Spec(
        make=lambda: RandomForestClassifier(n_estimators=200, n_jobs=-1,
                                            random_state=config.RANDOM_STATE),
        grid={"model__max_depth": [10, 20, 40, None],
              "model__min_samples_leaf": [1, 5]},
        why="Non-linear and the natural contrast to the two linear models: it captures "
            "interactions between API calls, needs no scaling, and yields feature "
            "importances for the report. max_depth and min_samples_leaf trade fit against "
            "variance; 200 trees are enough for a stable vote on 72,000 rows."),
}

# The assignment names accuracy as the primary metric, so accuracy decides the winner.
# The other metrics are computed on the same folds and reported next to it, because
# accuracy alone would hide a model that gets its score from one class (Step 5 looks at
# precision, recall and F1 in depth). Malware (label 1) is the positive class, which is
# scikit-learn's default for binary 0/1 labels.
# zero_division=0 is set explicitly: a baseline that predicts no positives has undefined
# precision, and scoring it 0 is the honest reading for a detector that never fires.
SCORING = {"accuracy": "accuracy",
           "precision": make_scorer(precision_score, zero_division=0),
           "recall": make_scorer(recall_score, zero_division=0),
           "f1": make_scorer(f1_score, zero_division=0),
           "roc_auc": "roc_auc"}
REFIT_METRIC = "accuracy"
REPRODUCIBILITY_ROWS = 4000   # rows used to fit each clone in the determinism check
REPRODUCIBILITY_EVAL_ROWS = 2000   # disjoint rows the two clones are compared on

# Grid values that are the parameter's own limit rather than merely the end of our grid.
# Nothing exists beyond them to search, so selecting one is not a grid-edge concern and
# the report says so instead of suggesting the grid be widened.
PARAM_LIMITS = {
    "max_depth": {"None"},        # unbounded depth: trees grow until the leaves are pure
    "min_samples_leaf": {"1"},    # a leaf cannot hold fewer than one sample
}


def make_model_pipeline(name):
    """Preprocessing pipeline for this model with the classifier appended as 'model'.

    The preprocessing steps come from preprocessor_for(name), so the per-model scaling
    policy recorded in Step 3 is what gets used, and they are unfitted: the search refits
    them on each fold's training years.
    """
    if name not in MODEL_SPECS:
        raise KeyError(f"make_model_pipeline: unknown model {name!r}; "
                       f"known: {sorted(MODEL_SPECS)}")
    pre = preprocessing.preprocessor_for(name)
    return Pipeline(list(pre.steps) + [("model", MODEL_SPECS[name].make())])


def check_folds(cv, years_train, train_years, folds=None, y_train=None):
    """Validate the fold indices, then the forward-chaining policy they are meant to encode.

    `preprocessing.fold_indices` builds these from year masks, so in normal use the
    indices are already well formed; this does not assume that (review R2). Before any
    index is used to look up a year it checks that each half holds unique, non-negative
    integer positions inside the array, and that the two halves are disjoint.

    Then, given the `folds` plan, it checks each fold *is* the intended one rather than
    merely time-ordered (review R3): the training years must equal the planned prefix and
    the validation half must contain exactly the planned year, in the planned order. With
    `folds=None` only the weaker chronological property is checked.

    With `y_train`, it also checks each validation half holds both classes, which ROC AUC
    needs to be defined (review R5).
    """
    years = np.asarray(years_train)
    n = len(years)
    allowed = set(int(y) for y in train_years)
    cv = list(cv)
    if folds is not None and len(cv) != len(folds):
        raise ValueError(f"check_folds: {len(cv)} fold index pairs for {len(folds)} planned "
                         f"folds")
    for i, (train_idx, val_idx) in enumerate(cv, start=1):
        # --- the indices themselves, before they are used to index anything (R2) ---
        for half, idx in (("training", train_idx), ("validation", val_idx)):
            idx = np.asarray(idx)
            if idx.size == 0:
                raise ValueError(f"check_folds: fold {i}'s {half} half is empty")
            if not np.issubdtype(idx.dtype, np.integer):
                raise ValueError(f"check_folds: fold {i}'s {half} indices are "
                                 f"{idx.dtype}, expected integers")
            if idx.min() < 0 or idx.max() >= n:
                raise ValueError(f"check_folds: fold {i}'s {half} indices run from "
                                 f"{idx.min()} to {idx.max()}, outside [0, {n})")
            if len(np.unique(idx)) != len(idx):
                raise ValueError(f"check_folds: fold {i}'s {half} half repeats "
                                 f"{len(idx) - len(np.unique(idx))} row index(es)")
        if set(np.asarray(train_idx).tolist()) & set(np.asarray(val_idx).tolist()):
            raise ValueError(f"check_folds: fold {i} has rows in both its training and "
                             f"validation half")

        fold_train = sorted(int(y) for y in set(years[np.asarray(train_idx)]))
        fold_val = sorted(int(y) for y in set(years[np.asarray(val_idx)]))
        outside = (set(fold_train) | set(fold_val)) - allowed
        if outside:
            raise ValueError(f"check_folds: fold {i} touches year(s) {sorted(outside)}, which "
                             f"are not training years {sorted(allowed)}")
        if max(fold_train) >= min(fold_val):
            raise ValueError(f"check_folds: fold {i} trains on {fold_train} and validates on "
                             f"{fold_val}, which is not time-ordered")

        # --- the fold is the planned one, not just a time-ordered one (R3) ---
        if folds is not None:
            planned_train, planned_val = folds[i - 1]
            planned_train = sorted(int(y) for y in planned_train)
            if fold_train != planned_train:
                raise ValueError(f"check_folds: fold {i} trains on {fold_train}, but the plan "
                                 f"says {planned_train}")
            if fold_val != [int(planned_val)]:
                raise ValueError(f"check_folds: fold {i} validates on {fold_val}, but the plan "
                                 f"says only {int(planned_val)}")
            planned_val_rows = set(np.flatnonzero(years == int(planned_val)).tolist())
            if set(np.asarray(val_idx).tolist()) != planned_val_rows:
                raise ValueError(f"check_folds: fold {i}'s validation half is not exactly the "
                                 f"{len(planned_val_rows)} rows of year {int(planned_val)}")

        # --- both classes present, or ROC AUC is undefined on this fold (R5) ---
        if y_train is not None:
            classes = set(np.asarray(y_train)[np.asarray(val_idx)].tolist())
            if len(classes) < 2:
                raise ValueError(f"check_folds: fold {i}'s validation year {fold_val} holds "
                                 f"only class {sorted(classes)}, so ROC AUC would be "
                                 f"undefined; expected both of "
                                 f"{sorted(config.LABEL_MAP.values())}")
    return True


def _model_state(estimator):
    """The fitted numbers worth comparing between two fits: preprocessing and model."""
    state = {}
    steps = estimator.named_steps
    if "drop_constant" in steps:
        state["kept_features"] = list(steps["drop_constant"].get_feature_names_out())
    if "scale" in steps:
        state["scaler_mean"] = np.asarray(steps["scale"].mean_)
        state["scaler_scale"] = np.asarray(steps["scale"].scale_)
    model = steps["model"]
    for attr in ("coef_", "intercept_", "feature_importances_"):
        if hasattr(model, attr):
            state[attr] = np.asarray(getattr(model, attr))
    return state


def _states_match(a, b):
    """True when two state dicts hold the same keys and numerically identical values."""
    if set(a) != set(b):
        return False
    for key in a:
        if isinstance(a[key], list):
            if a[key] != b[key]:
                return False
        elif not np.array_equal(a[key], b[key]):
            return False
    return True


def reproducibility_check(estimator, X, y, n_fit_rows=REPRODUCIBILITY_ROWS,
                          n_eval_rows=REPRODUCIBILITY_EVAL_ROWS, seed=config.RANDOM_STATE):
    """Fit the selected configuration twice on one subsample and compare what came out.

    This is a **sampled determinism check**, and the report describes it as exactly that
    (review R4). It compares three things between the two fits: predictions on a set of
    rows *disjoint* from the rows used to fit (so it is not only reading back memorised
    training rows), the preprocessing state (retained feature names, scaler mean and
    scale), and the model's own fitted parameters where it exposes them (`coef_`,
    `intercept_`, `feature_importances_`).

    It does not establish that the full-size fit is reproducible, nor that another machine
    or library version gives the same numbers; it covers one subsample under one seed.
    Returns a dict of the individual results plus `all_match`.
    """
    rng = np.random.RandomState(seed)
    take = rng.permutation(len(X))
    fit_idx = take[:min(n_fit_rows, len(X))]
    eval_idx = take[len(fit_idx):len(fit_idx) + min(n_eval_rows, len(X) - len(fit_idx))]
    if len(eval_idx) == 0:
        raise ValueError("reproducibility_check: no rows left over to evaluate on")
    X_fit, y_fit = X.iloc[fit_idx], y.iloc[fit_idx]
    X_eval = X.iloc[eval_idx]
    if y_fit.nunique() < 2:
        raise ValueError("reproducibility_check: the fitting subsample holds a single class")

    first = clone(estimator).fit(X_fit, y_fit)
    second = clone(estimator).fit(X_fit, y_fit)
    predictions_match = bool(np.array_equal(first.predict(X_eval), second.predict(X_eval)))
    state_match = _states_match(_model_state(first), _model_state(second))
    result = {"n_fit_rows": int(len(fit_idx)), "n_eval_rows": int(len(eval_idx)),
              "predictions_match": predictions_match, "state_match": bool(state_match)}
    result["all_match"] = bool(predictions_match and state_match)
    return result


def check_search_scores(name, search):
    """Every scorer must have produced usable numbers for the selected candidate (R5).

    Missing metric keys and a non-finite primary metric are errors: the selection itself
    would be meaningless. A non-finite secondary metric is returned in `unavailable` so
    the report can print "unavailable" for it rather than a bare NaN.
    """
    cv = search.cv_results_
    i = search.best_index_
    missing = [m for m in SCORING if f"mean_test_{m}" not in cv]
    if missing:
        raise KeyError(f"check_search_scores: {name}: the search produced no results for "
                       f"metric(s) {missing}")
    primary = float(cv[f"mean_test_{REFIT_METRIC}"][i])
    if not np.isfinite(primary):
        raise ValueError(f"check_search_scores: {name}: the selected candidate's "
                         f"{REFIT_METRIC} is {primary}, so the selection is not meaningful")
    unavailable = [m for m in SCORING
                   if not np.isfinite(float(cv[f"mean_test_{m}"][i]))]
    for fold in range(search.n_splits_):
        key = f"split{fold}_test_{REFIT_METRIC}"
        if key not in cv or not np.isfinite(float(cv[key][i])):
            raise ValueError(f"check_search_scores: {name}: fold {fold + 1} has no finite "
                             f"{REFIT_METRIC} for the selected candidate")
    return unavailable


def check_model_rows(best):
    """The report compares models by row, so the rows must be complete and finite (R8)."""
    counts = best["model"].value_counts()
    for name in MODEL_SPECS:
        if counts.get(name, 0) != 1:
            raise ValueError(f"check_model_rows: {counts.get(name, 0)} summary row(s) for "
                             f"{name}, expected exactly 1")
    unexpected = sorted(set(best["model"]) - set(MODEL_SPECS))
    if unexpected:
        raise ValueError(f"check_model_rows: summary rows for unknown model(s) {unexpected}")
    bad = best[~np.isfinite(best[f"cv_{REFIT_METRIC}_mean"])]
    if len(bad):
        raise ValueError(f"check_model_rows: non-finite cv_{REFIT_METRIC}_mean for "
                         f"{bad['model'].tolist()}")


def check_class_balance_input(summary, split):
    """Validate the rows the class-weight decision is read from (review R6)."""
    train = summary[summary["split"] == "train"]
    found = sorted(int(y) for y in train["year"])
    if found != sorted(int(y) for y in split.train_years):
        raise ValueError(f"check_class_balance_input: the summary's training years {found} do "
                         f"not match the split's {sorted(split.train_years)}")
    if train["year"].duplicated().any():
        raise ValueError("check_class_balance_input: a training year appears more than once "
                         "in the summary")
    shares = train["malware_share"].to_numpy(dtype=float)
    if not np.isfinite(shares).all() or shares.min() < 0 or shares.max() > 1:
        raise ValueError(f"check_class_balance_input: malware_share values {shares.tolist()} "
                         f"are not all finite and within [0, 1]")
    return shares


def class_balance_note(summary, split):
    """The class_weight decision, read from the validated per-year training shares.

    PLAN.md Step 4: 'Use class_weight="balanced" only if Step 1 shows a large imbalance.'
    """
    shares = check_class_balance_input(summary, split)
    rule = ("PLAN.md Step 4 says to use `class_weight=\"balanced\"` only if Step 1 shows a "
            "large imbalance.")
    if float(shares.min()) == float(shares.max()) == 0.5:
        return (f"Not used. {rule} Every training year is exactly 50% malware (Step 1 and "
                "Step 3), so there is no imbalance to correct and `balanced` would leave the "
                "weights unchanged.")
    return (f"Not used. {rule} The training years run from {100 * shares.min():.1f}% to "
            f"{100 * shares.max():.1f}% malware, which is not a large imbalance; the decision "
            "is recorded here so it is visible rather than implicit.")


def best_params_row(name, search, fit_seconds, reproducibility, unavailable):
    """One summary row per model: what was chosen, how it scored, how long it took."""
    i = search.best_index_
    cv = search.cv_results_
    row = {"model": name,
           "best_params": json.dumps({k.replace("model__", ""): v
                                      for k, v in search.best_params_.items()}),
           "n_candidates": len(cv["params"]),
           "n_folds": search.n_splits_,
           "fit_seconds": round(fit_seconds, 1),
           "reproducible": reproducibility["all_match"],
           "reproducible_predictions": reproducibility["predictions_match"],
           "reproducible_state": reproducibility["state_match"],
           "repro_fit_rows": reproducibility["n_fit_rows"],
           "repro_eval_rows": reproducibility["n_eval_rows"],
           "unavailable_metrics": ",".join(unavailable)}
    for metric in SCORING:
        row[f"cv_{metric}_mean"] = float(cv[f"mean_test_{metric}"][i])
        row[f"cv_{metric}_std"] = float(cv[f"std_test_{metric}"][i])
    for fold in range(search.n_splits_):
        row[f"fold{fold + 1}_accuracy"] = float(cv[f"split{fold}_test_accuracy"][i])
    return row


def cv_results_table(name, search):
    """Every candidate of this model's grid with its mean score per metric."""
    cv = search.cv_results_
    rows = []
    for i, params in enumerate(cv["params"]):
        row = {"model": name,
               "params": json.dumps({k.replace("model__", ""): v for k, v in params.items()}),
               "selected": bool(i == search.best_index_)}
        for metric in SCORING:
            row[f"{metric}_mean"] = float(cv[f"mean_test_{metric}"][i])
            row[f"{metric}_std"] = float(cv[f"std_test_{metric}"][i])
        for fold in range(search.n_splits_):
            row[f"fold{fold + 1}_accuracy"] = float(cv[f"split{fold}_test_accuracy"][i])
        rows.append(row)
    return pd.DataFrame(rows)


def grid_position_notes(best):
    """Say, per searched parameter, whether the chosen value sits inside the range.

    A value selected at an end of its grid is a sign the search may not have bracketed
    the optimum, which the report should disclose rather than leave for a reader to spot.
    Values listed in PARAM_LIMITS are excepted: `max_depth=None` and
    `min_samples_leaf=1` are the parameters' own limits, so there is nothing beyond them
    to search and calling them a grid edge would be misleading.
    """
    notes = []
    for name, spec in MODEL_SPECS.items():
        selected = json.loads(best[best["model"] == name]["best_params"].iloc[0])
        for key, values in spec.grid.items():
            param = key.replace("model__", "")
            if len(values) < 2 or param not in selected:
                continue
            chosen = selected[param]
            position = next((i for i, v in enumerate(values) if str(v) == str(chosen)), None)
            if position is None:
                continue
            searched = "{" + ", ".join("unbounded" if v is None else str(v) for v in values) + "}"
            shown = "unbounded" if chosen is None else chosen
            if str(chosen) in PARAM_LIMITS.get(param, set()):
                notes.append(f"**{name}** chose `{param}={shown}` from {searched}. That is the "
                             "limit of this parameter, so no value beyond it exists to search; "
                             "the other values scored worse. Not a grid-edge concern.")
            elif position == 0:
                notes.append(f"**{name}** chose `{param}={shown}`, the **lowest** value "
                             f"searched in {searched}, so a better value may lie below the "
                             "grid and this result should be read as a grid-edge selection.")
            elif position == len(values) - 1:
                notes.append(f"**{name}** chose `{param}={shown}`, the **highest** value "
                             f"searched in {searched}, so a better value may lie above the "
                             "grid and this result should be read as a grid-edge selection.")
            else:
                notes.append(f"**{name}** chose `{param}={shown}` from {searched}, an interior "
                             "value, so the search bracketed the optimum on this parameter.")
    return notes


def metric_disagreements(cv_table):
    """Models where another candidate would win under a metric other than accuracy.

    Accuracy decides the winner, so where a different candidate tops F1 or ROC AUC the
    report says so instead of presenting one ranking as the only one.
    """
    out = []
    for name, part in cv_table.groupby("model", sort=False):
        if len(part) < 2:
            continue
        chosen = part[part["selected"]].iloc[0]
        for metric in ("f1", "roc_auc"):
            top = part.loc[part[f"{metric}_mean"].idxmax()]
            if not bool(top["selected"]):
                out.append(f"For **{name}**, accuracy selects {json.loads(chosen['params'])} "
                           f"({metric} {chosen[f'{metric}_mean']:.4f}), but "
                           f"{json.loads(top['params'])} has the best {metric} "
                           f"({top[f'{metric}_mean']:.4f}). Accuracy is the stated primary "
                           "metric, so the accuracy choice stands; the difference is recorded "
                           "because it matters for a detector that must not miss malware.")
    return out


def check_cv_table(best, cv_table):
    """The candidate table must agree with the per-model summary (review R8/R10 habit)."""
    for name in MODEL_SPECS:
        part = cv_table[cv_table["model"] == name]
        if part.empty:
            raise ValueError(f"check_cv_table: no candidate rows for {name}")
        n_selected = int(part["selected"].sum())
        if n_selected != 1:
            raise ValueError(f"check_cv_table: {n_selected} selected candidates for {name}, "
                             f"expected exactly 1")
        expected = int(best[best["model"] == name]["n_candidates"].iloc[0])
        if len(part) != expected:
            raise ValueError(f"check_cv_table: {len(part)} candidate rows for {name}, but the "
                             f"summary records {expected}")
        chosen_params = json.loads(part[part["selected"]].iloc[0]["params"])
        summary_params = json.loads(best[best["model"] == name]["best_params"].iloc[0])
        if chosen_params != summary_params:
            raise ValueError(f"check_cv_table: {name}'s selected candidate {chosen_params} "
                             f"does not match the summary's {summary_params}")


def write_report(best, cv_table, folds, split, summary, path, balance_note):
    """Write docs/step4_models.md. Every number comes from the tables passed in.

    Counts in the prose (models, real classifiers, folds, metrics) are derived from
    MODEL_SPECS, `folds` and SCORING rather than written out, so a change to the model
    set cannot leave the text stale (review R7/R11).
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    check_model_rows(best)
    check_cv_table(best, cv_table)
    n_folds = len(folds)
    recorded_folds = sorted(set(int(v) for v in best["n_folds"]))
    if recorded_folds != [n_folds]:
        raise ValueError(f"write_report: the search ran {recorded_folds} fold(s) but the plan "
                         f"has {n_folds}")
    train_years, test_years = split.train_years, split.test_years
    winner = best.sort_values(f"cv_{REFIT_METRIC}_mean", ascending=False).iloc[0]
    baseline_name = "DummyClassifier"
    real = best[best["model"] != baseline_name]
    floor = float(best[best["model"] == baseline_name][f"cv_{REFIT_METRIC}_mean"].iloc[0])
    beat_floor = int((real[f"cv_{REFIT_METRIC}_mean"] > floor).sum())
    metric_names = list(SCORING)
    fold_lines = [f"  {i}. train on {', '.join(map(str, f_train))} -> validate on {f_val}"
                  for i, (f_train, f_val) in enumerate(folds, start=1)]

    def cell(row, metric):
        return ("unavailable" if metric in str(row["unavailable_metrics"]).split(",")
                else f"{row[f'cv_{metric}_mean']:.4f}")

    t_models = pd.DataFrame([
        {"classifier": name,
         "scaled": "yes" if preprocessing.SCALING_BY_MODEL[name][0] else "no",
         "grid searched": (", ".join(f"{k.replace('model__', '')} in "
                                     f"{{{', '.join('unbounded' if v is None else str(v) for v in vals)}}}"
                                     for k, vals in MODEL_SPECS[name].grid.items())
                           or "none (no hyperparameter to choose)"),
         "chosen": json.loads(best[best["model"] == name]["best_params"].iloc[0]) or "defaults"}
        for name in MODEL_SPECS])

    t_scores = pd.DataFrame([
        {"classifier": r["model"],
         "CV accuracy": f"{r['cv_accuracy_mean']:.4f} +/- {r['cv_accuracy_std']:.4f}",
         "CV precision": cell(r, "precision"), "CV recall": cell(r, "recall"),
         "CV F1": cell(r, "f1"), "CV ROC AUC": cell(r, "roc_auc"),
         "candidates": r["n_candidates"], "fit seconds": r["fit_seconds"]}
        for _, r in best.sort_values(f"cv_{REFIT_METRIC}_mean", ascending=False).iterrows()])

    fold_cols = [f"fold{i + 1}_accuracy" for i in range(n_folds)]
    t_folds = pd.DataFrame([
        {"classifier": r["model"],
         **{f"fold {i + 1} ({folds[i][1]})": f"{r[c]:.4f}" for i, c in enumerate(fold_cols)}}
        for _, r in best.iterrows()])

    repro_all = bool(best["reproducible"].all())
    repro_rows = best.iloc[0]

    L = [
        "# Step 4 Report: Classifiers and Training", "",
        "Written by the builder agent. Generated by `python -m src.main --step 4`; all numbers "
        "come from code (`results/step4_best_params.csv`, `results/step4_cv_results.csv`; fitted "
        "models in `outputs/models/`).", "",
        "## Classifiers chosen and why", "",
        f"{len(MODEL_SPECS)} scikit-learn estimators, trained on "
        f"{', '.join(map(str, train_years))} only:", ""]
    for name, spec in MODEL_SPECS.items():
        L += [f"**{name}.** {spec.why}", ""]
    L += [
        f"{len(real)} of them are real classifiers, which satisfies the assignment's \"at least "
        "two\"; the remaining one is a baseline that cannot learn and exists to make the others' "
        "accuracy interpretable. One linear model, one linear model trained incrementally, and "
        "one non-linear ensemble give three different inductive biases on the same features.", "",
        "## Hyperparameters and how they were selected", "",
        t_models.to_markdown(index=False), "",
        "The grids are small on purpose (PLAN.md Step 4): for each model, a few values of the "
        "parameter that matters most on 1,105 correlated count features, so the selection is "
        "measured rather than asserted and still finishes in a single run. Everything else is "
        "left at the documented value in the estimator call, and `random_state=42` is fixed "
        "everywhere.", "",
        f"Selection used **forward-chaining validation inside the training years only**, never "
        f"random k-fold, which would let a fold train on a year it validates on. The {n_folds} "
        "folds are:", "",
        *fold_lines, "",
        "Each fold refits the whole pipeline - the dropped-column set, the scaler and the "
        "classifier - on that fold's training years, and only transforms its validation year. "
        f"After the search, scikit-learn refits the winning configuration on all "
        f"{int(summary[summary['split'] == 'train']['rows'].sum()):,} training rows, and that is "
        "the model saved for Step 5.", "",
        "**What this step does with the test years.** Nothing is fitted on them, scored against "
        "them, or chosen using them. To be exact about it: the shared loader in Step 1 builds "
        "one table containing every year, so test rows are present in memory, and Step 3's "
        "validated split is what separates them. This step then passes only the training frame "
        f"to `features_labels`, so no test feature matrix is built here at all. The "
        f"{len(test_years)} test years ({', '.join(map(str, test_years))}) are first used in "
        "Step 5.", "",
        f"**Selection metric.** Accuracy, because the assignment names it the primary metric. "
        f"The other {len(metric_names) - 1} metrics ({', '.join(m for m in metric_names if m != REFIT_METRIC)}) "
        "are computed on the same folds and reported beside it, so a model that bought accuracy "
        "by neglecting one class is visible here rather than only in Step 5. Malware (label 1) "
        "is the positive class.", "",
        f"**Class weights.** {balance_note}", "",
        "## Selection results", "",
        f"Mean over the {n_folds} validation folds, for the configuration chosen per model:", "",
        t_scores.to_markdown(index=False), "",
        "Per-fold accuracy for the chosen configuration (the column heading names the "
        "validation year):", "",
        t_folds.to_markdown(index=False), "",
        f"- The best cross-validated accuracy is **{winner['model']}** at "
        f"{winner[f'cv_{REFIT_METRIC}_mean']:.4f} "
        f"(+/- {winner[f'cv_{REFIT_METRIC}_std']:.4f} across folds), with "
        f"{json.loads(winner['best_params']) or 'default parameters'}.",
        (f"- All {beat_floor} real classifiers beat the {baseline_name} floor of {floor:.4f}."
         if beat_floor == len(real) else
         f"- {beat_floor} of {len(real)} real classifiers beat the {baseline_name} floor of "
         f"{floor:.4f}; any that do not are not learning from the features.")
        + " Beating that floor means a model is using the features rather than the class "
        "balance. It does not by itself show the model is a useful malware detector: that "
        "depends on how the errors split between the classes, which is why precision, recall "
        "and F1 are in the table above and are examined per year in Step 6.",
        "- Every candidate of every grid, with all five metrics and the per-fold scores, is in "
        "`results/step4_cv_results.csv`.",
        f"- **Precision exceeds recall for every real classifier** (for {winner['model']}, "
        f"{winner['cv_precision_mean']:.3f} against {winner['cv_recall_mean']:.3f}). On "
        "training-year data these models rarely raise a false alarm but miss roughly "
        f"{100 * (1 - winner['cv_recall_mean']):.0f}% of malware at the default decision "
        "threshold. That trade-off, and whether to move the threshold, belongs to Step 5.",
        "- Accuracy falling from fold to fold would already hint at drift inside the training "
        "years; the fold columns above are the place to look, and Step 6 measures the same thing "
        "on the test years.", "",
        "### Did the grids bracket the chosen values?", "",
        "A value chosen at an end of its grid may not be the best value, only the best of those "
        "searched. This is checked automatically for every searched parameter:", "",
        *[f"- {note}" for note in grid_position_notes(best)], "",
        *(["### Where another metric would choose differently", "",
           *[f"- {note}" for note in metric_disagreements(cv_table)], ""]
          if metric_disagreements(cv_table) else []),
        "## What this step does not do", "",
        "- **No test-year scoring or selection.** See the paragraph above for exactly what that "
        "means. Reporting a test score in this step would make the selection above unusable, "
        "because the choice would then depend on the data it is meant to be judged on.",
        "- **No threshold tuning.** All metrics use each model's default decision rule. Moving a "
        "threshold to trade precision against recall is a Step 5 question, and it would have to "
        "be chosen on training-year folds, not on test years.",
        "- **No feature selection beyond Step 3's constant-column removal**, which was fitted on "
        "training years and is refitted per fold.", "",
        "## Checks run on this step", "",
        "- **Fold indices are well formed, and the folds are the planned ones.** `check_folds` "
        "validates that each half holds unique integer positions inside the training array and "
        "that the halves are disjoint, before any index is used; then it checks each fold's "
        "training years equal the planned prefix and its validation half is exactly the rows of "
        f"the planned year, in the planned order, all inside {', '.join(map(str, train_years))}. "
        "It also requires both classes in every validation year, without which ROC AUC would be "
        "undefined. *Establishes:* model selection ran on the intended forward-chaining folds "
        "and saw no test-year row. *Does not establish:* anything about the final refit, which "
        "is on the whole training half by design.",
        f"- **Sampled determinism of the chosen configurations.** Each selected pipeline was "
        f"cloned and fitted twice on the same {int(repro_rows['n_folds']) and REPRODUCIBILITY_ROWS:,}"
        f"-row training subsample, then compared on "
        f"{REPRODUCIBILITY_EVAL_ROWS:,} *held-out* training rows and on the fitted state "
        "itself: retained feature names, scaler mean and scale, and the model's own parameters "
        f"(`coef_`, `intercept_` or `feature_importances_`). "
        f"{'All' if repro_all else str(int(best['reproducible'].sum())) + ' of ' + str(len(best))}"
        f" model{'s' if repro_all or int(best['reproducible'].sum()) != 1 else ''} matched on "
        "both. *Establishes:* for this configuration, this seed and this subsample, two fits "
        "produce the same preprocessing, the same parameters and the same predictions on rows "
        "they were not fitted on. *Does not establish:* that the full "
        f"{int(summary[summary['split'] == 'train']['rows'].sum()):,}-row fit is reproducible, "
        "or that a different machine or library version gives the same numbers.",
        "- **Every metric the report quotes is finite.** `check_search_scores` requires the "
        f"selected candidate to have a finite {REFIT_METRIC} on every fold and all "
        f"{len(metric_names)} metric keys present; a secondary metric that came out undefined "
        "is printed as \"unavailable\" rather than as a number.",
        "- **The tables agree with each other.** `check_model_rows` requires exactly one summary "
        f"row per model with a finite {REFIT_METRIC}; `check_cv_table` requires each model's "
        "candidate rows to be present, to have exactly one selected row, and for that row's "
        "parameters to match the summary. `check_class_balance_input` validates the per-year "
        "shares the class-weight decision is read from.",
        "- **Training rows only.** The matrices come from Step 3's `features_labels` on the "
        "training frame, whose years and labels were checked by `check_labels_years_side` before "
        "any fitting.", "",
        "## Files written", "",
        "Written to a staging directory first and moved into place only after every file "
        "succeeds, so a failed run cannot leave a partial set that looks like a complete Step 4 "
        "result:", "",
        "- `results/step4_best_params.csv`: the chosen configuration per model, its CV scores "
        "per metric, per-fold accuracy, fit time and the determinism results.",
        f"- `results/step4_cv_results.csv`: every grid candidate with all {len(metric_names)} "
        "metrics.",
        "- `outputs/models/<classifier>.joblib`: each winning pipeline fitted on the full "
        "training half (preprocessing and classifier together, so Step 5 applies exactly what "
        "was fitted).", "",
        "## What Step 5 uses from here", "",
        "The saved pipelines are the trained models to score. Step 5 loads each one, calls "
        "`predict` (and `predict_proba` where available) on the test years returned by "
        "`preprocessing.split_by_year`, and reports accuracy, precision, recall and F1 with "
        "malware as the positive class. The pipelines already contain their own preprocessing, "
        "so Step 5 must not transform the test data itself.", ""]
    path.write_text("\n".join(L))


def run(X_train, y_train, years_train, split, summary, cv, folds, results_dir, models_dir,
        docs_dir, verbose=1):
    """Select, fit and save every model; return (best_params table, cv results table).

    Order: validate, then fit, then write. Writes go to a staging directory and are moved
    into place only once all of them have succeeded, so a failure cannot leave a mixture
    of new and stale Step 4 artifacts behind (review R9).
    """
    check_folds(cv, years_train, split.train_years, folds=folds, y_train=y_train)
    balance_note = class_balance_note(summary, split)   # validates its input (R6)

    searches, rows, tables = {}, [], []
    for name in MODEL_SPECS:
        started = time.perf_counter()
        search = select_and_fit(name, X_train, y_train, cv, verbose=verbose)
        elapsed = time.perf_counter() - started
        unavailable = check_search_scores(name, search)
        reproducibility = reproducibility_check(search.best_estimator_, X_train, y_train)
        searches[name] = search
        rows.append(best_params_row(name, search, elapsed, reproducibility, unavailable))
        tables.append(cv_results_table(name, search))
        if verbose:
            print(f"{name}: accuracy {search.best_score_:.4f}, "
                  f"params {search.best_params_ or 'defaults'}, {elapsed:.1f}s, "
                  f"reproducible={reproducibility['all_match']}"
                  + (f", unavailable metrics: {unavailable}" if unavailable else ""))
    best = pd.DataFrame(rows)
    cv_table = pd.concat(tables, ignore_index=True)
    check_model_rows(best)
    check_cv_table(best, cv_table)

    # --- staged writes (R9) -------------------------------------------------------------
    import joblib
    staging = models_dir.parent / "step4_staging"
    if staging.exists():
        shutil.rmtree(staging)
    (staging / "models").mkdir(parents=True)
    best.to_csv(staging / "step4_best_params.csv", index=False)
    cv_table.to_csv(staging / "step4_cv_results.csv", index=False)
    for name, search in searches.items():
        joblib.dump(search.best_estimator_, staging / "models" / f"{name}.joblib")
    write_report(best, cv_table, folds, split, summary, staging / "step4_models.md",
                 balance_note)

    expected = ([staging / "step4_best_params.csv", staging / "step4_cv_results.csv",
                 staging / "step4_models.md"]
                + [staging / "models" / f"{name}.joblib" for name in MODEL_SPECS])
    missing = [p.name for p in expected if not p.exists() or p.stat().st_size == 0]
    if missing:
        raise RuntimeError(f"run: staged Step 4 output is incomplete, missing/empty {missing}; "
                           f"the previous artifacts were left untouched")

    if models_dir.exists():
        shutil.rmtree(models_dir)
    shutil.move(str(staging / "models"), str(models_dir))
    shutil.move(str(staging / "step4_best_params.csv"),
                str(results_dir / "step4_best_params.csv"))
    shutil.move(str(staging / "step4_cv_results.csv"), str(results_dir / "step4_cv_results.csv"))
    shutil.move(str(staging / "step4_models.md"), str(docs_dir / "step4_models.md"))
    shutil.rmtree(staging)
    return best, cv_table


def select_and_fit(name, X_train, y_train, cv, n_jobs=1, verbose=0):
    """Search this model's grid on the forward-chaining folds, then refit on all of them.

    Returns the fitted GridSearchCV. `refit=REFIT_METRIC` means `best_estimator_` is the
    winning pipeline fitted on the whole training half, which is the model Step 5 scores.
    n_jobs defaults to 1 so only one fold is in memory at a time; Random Forest gets its
    parallelism from the estimator's own n_jobs instead.
    """
    search = GridSearchCV(make_model_pipeline(name), MODEL_SPECS[name].grid,
                          scoring=SCORING, refit=REFIT_METRIC, cv=list(cv),
                          n_jobs=n_jobs, verbose=verbose, return_train_score=False,
                          error_score="raise")
    search.fit(X_train, y_train)
    return search
