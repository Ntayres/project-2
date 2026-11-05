"""Step 2: measure feature drift across years (descriptive only).

The features are non-negative integer API-call counts, so drift is measured with
measures that suit counts. For every pair of years, and separately for all apps,
benign apps and malware apps, each of the 1,848 features is compared with:
  * the two-sample Kolmogorov-Smirnov (KS) statistic and its asymptotic p-value
    (integer counts contain many ties, so the p-value is only approximate);
  * the change in prevalence (share of apps with a nonzero count of that call);
  * the Jensen-Shannon distance between the binned count distributions (base 2, so it
    lies in [0, 1]).
Benjamini-Hochberg (BH) correction is applied across the 1,848 features, separately for
each year pair and class. A "year classifier" shows how well a simple model can tell two
years apart (temporal distinguishability; this is NOT evidence of concept drift).

Terminology (Step 2 review R2). This module works on feature vectors, never on app
identity. "Identical feature vectors" are rows whose 1,848 counts are equal, and
collapsing them is "feature-vector deduplication". Two identical feature vectors are not
necessarily the same application: only `apkname` can establish that, and Step 1 counts
duplicate identifiers separately. The three situations kept apart here are
(a) duplicate `apkname` values (Step 1), (b) identical feature vectors inside one
(year, class) block, and (c) identical feature vectors shared by two compared years.

Nothing here is used to choose the train/test split, select features or tune models
(PLAN.md leakage rule 2). Input data frames are never modified.
"""
from itertools import combinations

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GroupShuffleSplit
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from src import config

CLASSES = ["all", "benign", "malware"]
LARGE_EFFECT_KS = 0.2   # descriptive convention for a "large" KS effect, not a tuned value
ALPHA = 0.05            # BH false discovery rate level
MIN_CLASSIFIER_VECTORS = 30  # fewest identical-vector-free rows per year to run the classifier
N_CLASSIFIER_SPLITS = 5      # repeated group-aware splits, so one split cannot set the AUC
CLASSIFIER_TEST_SIZE = 0.3


def bh_adjust(p):
    """Return Benjamini-Hochberg adjusted p-values (never smaller than the raw p-values)."""
    p = np.asarray(p, dtype=float)
    n = len(p)
    order = np.argsort(p)
    scaled = p[order] * n / np.arange(1, n + 1)
    scaled = np.minimum.accumulate(scaled[::-1])[::-1]  # enforce monotonicity
    adjusted = np.empty(n)
    adjusted[order] = np.clip(scaled, 0.0, 1.0)
    return adjusted


def count_bins(values):
    """Map counts to log2 bins: 0, 1, 2, 3-4, 5-8, 9-16, ... (returns an int8 array)."""
    v = np.asarray(values)
    idx = np.zeros(v.shape, dtype=np.int8)
    nz = v > 0
    idx[nz] = 1 + np.ceil(np.log2(v[nz])).astype(np.int8)
    return idx


def bin_histogram(values, n_bins):
    """Return an (n_features, n_bins) matrix of how many apps fall in each count bin."""
    idx = count_bins(values)
    return np.stack([(idx == b).sum(axis=0) for b in range(n_bins)], axis=1)


def js_distance(hist_a, hist_b):
    """Row-wise Jensen-Shannon distance (base 2) between two count-histogram matrices."""
    p = hist_a / hist_a.sum(axis=1, keepdims=True)
    q = hist_b / hist_b.sum(axis=1, keepdims=True)
    m = 0.5 * (p + q)

    def kl(a, b):
        with np.errstate(divide="ignore", invalid="ignore"):
            term = np.where(a > 0, a * np.log2(a / b), 0.0)
        return term.sum(axis=1)

    return np.sqrt(np.clip(0.5 * kl(p, m) + 0.5 * kl(q, m), 0.0, 1.0))


def compare_years(a, b, hist_a, hist_b):
    """Compare two row blocks (rows x features) feature by feature.

    Returns arrays (ks, p, prevalence_a, prevalence_b, js). p is the asymptotic KS p-value.
    """
    n_features = a.shape[1]
    ks = np.empty(n_features)
    p = np.empty(n_features)
    for j in range(n_features):
        res = stats.ks_2samp(a[:, j], b[:, j], method="asymp")
        ks[j], p[j] = res.statistic, res.pvalue
    return ks, p, (a > 0).mean(axis=0), (b > 0).mean(axis=0), js_distance(hist_a, hist_b)


def _blocks(X, years, labels, keep=None):
    """Split the matrix into per (year, class) blocks; class 'all' joins both labels.

    keep is an optional boolean mask of rows to use (used for the de-duplicated check).
    Returns {(year, class): (matrix, histogram)}.
    """
    n_bins = int(np.ceil(np.log2(max(int(X.max()), 2)))) + 2
    out, hists = {}, {}
    for year in np.unique(years):
        for cls in ("benign", "malware"):
            mask = class_mask(years, labels, year, cls)   # same rule as the classifier uses
            if keep is not None:
                mask = mask & keep
            out[(year, cls)] = X[mask]
            hists[(year, cls)] = bin_histogram(out[(year, cls)], n_bins)
        out[(year, "all")] = np.vstack([out[(year, "benign")], out[(year, "malware")]])
        hists[(year, "all")] = hists[(year, "benign")] + hists[(year, "malware")]
    return out, hists


def feature_drift(X, years, labels, names, pairs=None, keep=None):
    """Return one long table: one row per (year pair, class, feature).

    Columns: year_a, year_b, class, feature, ks, p_value, q_value (BH within the family of
    all features for that year pair and class), prevalence_a, prevalence_b,
    prevalence_change (b minus a), js_distance, n_a, n_b.
    """
    year_list = sorted(np.unique(years))
    pairs = pairs if pairs is not None else list(combinations(year_list, 2))
    blocks, hists = _blocks(X, years, labels, keep)
    frames = []
    for ya, yb in pairs:
        for cls in CLASSES:
            a, b = blocks[(ya, cls)], blocks[(yb, cls)]
            ks, p, pa, pb, js = compare_years(a, b, hists[(ya, cls)], hists[(yb, cls)])
            frames.append(pd.DataFrame({
                "year_a": int(ya), "year_b": int(yb), "class": cls, "feature": names,
                "ks": ks, "p_value": p, "q_value": bh_adjust(p),
                "prevalence_a": pa, "prevalence_b": pb, "prevalence_change": pb - pa,
                "js_distance": js, "n_a": len(a), "n_b": len(b)}))
    return pd.concat(frames, ignore_index=True)


def pair_summary(drift):
    """Summarise per (year pair, class): significant features, mean KS, mean JS, etc."""
    g = drift.groupby(["year_a", "year_b", "class"])
    out = g.agg(mean_ks=("ks", "mean"), mean_js=("js_distance", "mean"),
                mean_abs_prevalence_change=("prevalence_change", lambda s: s.abs().mean()),
                n_significant=("q_value", lambda s: int((s < ALPHA).sum())),
                n_large_effect=("ks", lambda s: int((s >= LARGE_EFFECT_KS).sum())),
                n_features=("ks", "size"), n_a=("n_a", "first"), n_b=("n_b", "first"))
    return out.reset_index()


def pair_row(summary, year_a, year_b, cls="all", what="pair summary"):
    """Return the one row of a pair table for (year_a, year_b, class).

    Fails with a message naming the missing or repeated comparison instead of letting
    an .iloc[0] pick the wrong row or raise an obscure IndexError (review R10).
    """
    sub = summary[(summary["year_a"] == year_a) & (summary["year_b"] == year_b)
                  & (summary["class"] == cls)]
    if len(sub) == 0:
        raise KeyError(f"{what}: no row for year pair {year_a}-{year_b}, class {cls!r}")
    if len(sub) > 1:
        raise ValueError(f"{what}: {len(sub)} rows for year pair {year_a}-{year_b}, "
                         f"class {cls!r}; expected exactly one")
    return sub.iloc[0]


def symmetric_matrix(summary, value, cls, years):
    """Turn the pair summary into a symmetric years x years matrix with a zero diagonal.

    Validates that the summary holds the expected years and no repeated year pair, so a
    stale or duplicated table cannot quietly produce a half-filled matrix (review R10).
    """
    years = [int(y) for y in years]
    if value not in summary.columns:
        raise KeyError(f"symmetric_matrix: {value!r} is not a column of the pair summary")
    sub = summary[summary["class"] == cls]
    if sub.empty:
        raise KeyError(f"symmetric_matrix: the pair summary has no rows for class {cls!r}")
    known = set(years)
    mat = pd.DataFrame(0.0, index=years, columns=years)
    seen = set()
    for row in sub.itertuples():
        ya, yb = int(row.year_a), int(row.year_b)
        unknown = {ya, yb} - known
        if unknown:
            raise ValueError(f"symmetric_matrix: year pair {ya}-{yb} refers to year(s) "
                             f"{sorted(unknown)}, which are not in {years}")
        if (ya, yb) in seen:
            raise ValueError(f"symmetric_matrix: duplicate rows for year pair {ya}-{yb}, "
                             f"class {cls!r}")
        seen.add((ya, yb))
        mat.loc[ya, yb] = mat.loc[yb, ya] = getattr(row, value)
    missing = set(combinations(years, 2)) - seen
    if missing:
        raise ValueError(f"symmetric_matrix: the pair summary is missing {len(missing)} "
                         f"year pair(s) for class {cls!r}, e.g. {sorted(missing)[:3]}")
    return mat


def class_mask(years, labels, year, cls):
    """Boolean mask of the rows of one year for one class ('all' joins both labels)."""
    if cls not in CLASSES:
        raise ValueError(f"class_mask: unknown class {cls!r}, expected one of {CLASSES}")
    mask = years == year
    if cls != "all":
        mask = mask & (labels == config.LABEL_MAP[cls])
    return mask


def unique_vector_index(X, idx):
    """Row indices of the first occurrence of each distinct feature vector within idx.

    This is feature-vector deduplication, not application deduplication: identical
    vectors may belong to different apps (review R2).
    """
    if len(idx) == 0:
        return idx
    _, first = np.unique(X[idx], axis=0, return_index=True)
    return idx[np.sort(first)]


def year_classifier(X, years, labels, pairs, cls, per_year=2000, seed=config.RANDOM_STATE,
                    n_splits=N_CLASSIFIER_SPLITS, min_vectors=MIN_CLASSIFIER_VECTORS,
                    test_size=CLASSIFIER_TEST_SIZE):
    """Temporal distinguishability: how well can a model tell year A from year B?

    Descriptive only. A high AUC says the feature representation separates the two years;
    it is not a temporal-generalisation result and is never used to tune anything
    (review R6). The split is a random split inside the two-year sample on purpose: the
    question is "do the years differ", not "does a model generalise to later years".

    Guards against an optimistic AUC (review R1): identical feature vectors are collapsed
    inside each (year, class) block, both years are sampled to the same size, and the
    train/test split is group-aware, with one group per distinct feature vector across the
    two years. So a vector that occurs in both compared years stays on one side of every
    split. `n_cross_year_identical` reports how many such vectors there are.

    Columns are named for what they count (review R4): `available_rows_*` are all rows of
    that year and class, `available_unique_vectors_*` are its distinct feature vectors, and
    `classifier_sample_size` is how many were drawn per year for this comparison. A
    comparison with too few vectors, or with a split that cannot hold both years, is
    recorded with `skipped=True` and a reason rather than dropped or half-computed
    (review R5).
    """
    rng = np.random.RandomState(seed)
    rows = []
    for ya, yb in pairs:
        ya, yb = int(ya), int(yb)
        record = {"year_a": ya, "year_b": yb, "class": cls}
        available, unique = {}, {}
        for y in (ya, yb):
            idx = np.flatnonzero(class_mask(years, labels, y, cls))
            available[y] = idx
            unique[y] = unique_vector_index(X, idx)
        record.update({
            "available_rows_a": len(available[ya]), "available_rows_b": len(available[yb]),
            "available_unique_vectors_a": len(unique[ya]),
            "available_unique_vectors_b": len(unique[yb])})

        # R5: refuse to run on a year/class block that cannot support a split.
        too_small = [y for y in (ya, yb) if len(unique[y]) < min_vectors]
        if too_small:
            sizes = ", ".join(f"{y}: {len(unique[y])}" for y in too_small)
            rows.append({**record, "classifier_sample_size": 0, "n_cross_year_identical": 0,
                         "n_groups": 0, "n_splits_used": 0, "roc_auc_mean": np.nan,
                         "roc_auc_std": np.nan, "skipped": True,
                         "skip_reason": f"fewer than {min_vectors} distinct feature vectors "
                                        f"({sizes})"})
            continue

        n = min(per_year, len(unique[ya]), len(unique[yb]))
        rows_idx = np.concatenate([rng.choice(unique[y], n, replace=False) for y in (ya, yb)])
        target = np.repeat([0, 1], n)

        # R1: one group per distinct feature vector across both years, so an identical
        # vector cannot be in the training and the test half of the same split.
        _, groups = np.unique(X[rows_idx], axis=0, return_inverse=True)
        groups = groups.ravel()
        # A group holding both targets is one feature vector shared by the two years.
        shared = int((pd.Series(target).groupby(groups).nunique() > 1).sum())
        record.update({"classifier_sample_size": int(n), "n_cross_year_identical": int(shared),
                       "n_groups": int(len(np.unique(groups)))})

        # R11: counts must be non-negative for log1p, and the result must be finite.
        raw = X[rows_idx].astype(float)
        if raw.min() < 0:
            raise ValueError(f"year_classifier: negative counts in {ya} vs {yb} ({cls})")
        Xp = np.log1p(raw)
        if not np.isfinite(Xp).all():
            raise ValueError(f"year_classifier: non-finite features after log1p in "
                             f"{ya} vs {yb} ({cls})")

        # R6: repeat the group-aware split so the AUC comes with a spread, not one draw.
        splitter = GroupShuffleSplit(n_splits=n_splits, test_size=test_size, random_state=seed)
        aucs = []
        for train_idx, test_idx in splitter.split(Xp, target, groups=groups):
            # Both years must be present on both sides or the AUC is undefined (R5).
            if len(np.unique(target[train_idx])) < 2 or len(np.unique(target[test_idx])) < 2:
                continue
            model = make_pipeline(StandardScaler(),      # fitted inside the split only (R11)
                                  LogisticRegression(C=0.1, max_iter=300, random_state=seed))
            model.fit(Xp[train_idx], target[train_idx])
            aucs.append(roc_auc_score(target[test_idx],
                                      model.predict_proba(Xp[test_idx])[:, 1]))
        if not aucs:
            rows.append({**record, "n_splits_used": 0, "roc_auc_mean": np.nan,
                         "roc_auc_std": np.nan, "skipped": True,
                         "skip_reason": f"no split of the {n} sampled vectors per year kept "
                                        "both years on both sides"})
            continue
        rows.append({**record, "n_splits_used": len(aucs),
                     "roc_auc_mean": float(np.mean(aucs)),
                     "roc_auc_std": float(np.std(aucs, ddof=1)) if len(aucs) > 1 else 0.0,
                     "skipped": False, "skip_reason": ""})
    return pd.DataFrame(rows)


def class_cancellation(table, pairs, strict=False):
    """For each year pair: features with KS >= the large-effect cut-off in the pooled data,
    in at least one class, and in at least one class while the benign and malware prevalence
    move in opposite directions (so the pooled distribution can hide the change).

    "Opposite direction" means the two prevalence changes have strictly opposite signs. A
    change of exactly zero in either class is NOT counted as an opposite direction.

    Before pivoting, every feature must have an 'all', a 'benign' and a 'malware' row for
    that year pair, exactly once (review R9). Features missing a class are excluded and
    counted in `n_features_excluded`; pass strict=True to raise instead. A year pair with
    no rows at all, or a class missing for the whole pair, is always an error.
    """
    rows = []
    for ya, yb in pairs:
        ya, yb = int(ya), int(yb)
        sub = table[(table["year_a"] == ya) & (table["year_b"] == yb)]
        if sub.empty:
            raise KeyError(f"class_cancellation: no drift rows for year pair {ya}-{yb}")
        counts = sub.pivot_table(index="feature", columns="class", values="ks", aggfunc="size")
        absent = [c for c in CLASSES if c not in counts.columns]
        if absent:
            raise KeyError(f"class_cancellation: year pair {ya}-{yb} has no rows for "
                           f"class(es) {absent}")
        counts = counts[CLASSES].fillna(0)
        if (counts > 1).to_numpy().any():
            repeated = counts.index[(counts > 1).any(axis=1)][:3].tolist()
            raise ValueError(f"class_cancellation: year pair {ya}-{yb} has repeated "
                             f"(feature, class) rows, e.g. {repeated}")
        complete = (counts == 1).all(axis=1)
        if not complete.all():
            incomplete = counts.index[~complete]
            if strict:
                raise ValueError(f"class_cancellation: year pair {ya}-{yb} is missing a class "
                                 f"for {len(incomplete)} feature(s), e.g. "
                                 f"{incomplete[:3].tolist()}")
        w = sub[sub["feature"].isin(counts.index[complete])].pivot(
            index="feature", columns="class", values=["ks", "prevalence_change"])
        any_class = (w["ks"]["benign"] >= LARGE_EFFECT_KS) | (w["ks"]["malware"] >= LARGE_EFFECT_KS)
        opposite = np.sign(w["prevalence_change"]["benign"]) * np.sign(
            w["prevalence_change"]["malware"]) < 0
        rows.append({"year_a": ya, "year_b": yb,
                     "n_features_compared": int(complete.sum()),
                     "n_features_excluded": int((~complete).sum()),
                     "n_pooled": int((w["ks"]["all"] >= LARGE_EFFECT_KS).sum()),
                     "n_in_a_class": int(any_class.sum()),
                     "n_in_a_class_opposite_direction": int((any_class & opposite).sum())})
    return pd.DataFrame(rows)


def short_name(signature, width=70):
    """Shorten an API signature to 'Class;.method' for tables."""
    s = signature.split(":(")[0]
    return s if len(s) <= width else s[: width - 3] + "..."


def top_features(drift, k=10):
    """Top-k features by KS for each (year pair, class) in the table passed in."""
    keep = drift.sort_values("ks", ascending=False).groupby(
        ["year_a", "year_b", "class"], sort=True).head(k)
    keep = keep.sort_values(["year_a", "year_b", "class", "ks"],
                            ascending=[True, True, True, False])
    keep.insert(3, "rank", keep.groupby(["year_a", "year_b", "class"]).cumcount() + 1)
    return keep.reset_index(drop=True)


def year_stats(X, years, labels, apknames=None):
    """Per-year availability and class balance, measured from the loaded data.

    The report quotes these instead of assuming row counts or a 50/50 class split
    (review R3). The three quantities are kept apart (review R2/R4):
    `available_rows` (all rows of that year), `available_unique_vectors` (its distinct
    feature vectors) and `unique_apknames` (its distinct app identifiers).
    """
    rows = []
    for y in sorted(np.unique(years)):
        idx = np.flatnonzero(years == y)
        rec = {"year": int(y), "available_rows": len(idx),
               "available_unique_vectors": len(unique_vector_index(X, idx)),
               "n_benign": int((labels[idx] == config.LABEL_MAP["benign"]).sum()),
               "n_malware": int((labels[idx] == config.LABEL_MAP["malware"]).sum())}
        rec["malware_share"] = rec["n_malware"] / rec["available_rows"]
        if apknames is not None:
            rec["unique_apknames"] = int(pd.unique(np.asarray(apknames)[idx]).size)
        rows.append(rec)
    return pd.DataFrame(rows).set_index("year")


def _pct(x, decimals=0):
    """Format a share as a percentage."""
    return f"{100 * x:.{decimals}f}%"


def _year_runs(years):
    """Describe a list of years compactly: [2014, 2015, 2016, 2019] -> '2014-2016, 2019'."""
    years = sorted(int(y) for y in years)
    parts, start = [], years[0]
    for prev, cur in zip(years, years[1:] + [None]):
        if cur != prev + 1:
            parts.append(f"{start}" if start == prev else f"{start}-{prev}")
            start = cur
    return ", ".join(parts)


def _size_phrase(stats, column="available_rows"):
    """Describe the per-year sizes as the data has them, equal or not.

    One size: 'every year has 120 rows'. Several: '18,000 rows in 2014-2018 and 3,000 in
    2019-2021'. The report must not call the years unequal when they are not (review R3).
    """
    groups = {}
    for year, value in stats[column].items():
        groups.setdefault(int(value), []).append(int(year))
    if len(groups) == 1:
        return f"every year has {next(iter(groups)):,} rows"
    items = [f"{value:,} rows in {_year_runs(ys)}" for value, ys in sorted(groups.items(),
                                                                          reverse=True)]
    joined = (" and ".join(items) if len(items) == 2
              else ", ".join(items[:-1]) + f" and {items[-1]}")
    return f"the years are not the same size: {joined}"


def write_report(summary, feat, top, clf, dedup, years, path, cancel=None, stats=None):
    """Write docs/step2_drift_analysis.md using the course template headings.

    Every number in the text and tables is computed from the result tables passed in, and
    every claim about a particular year pair is taken from that pair's computed row
    (reviews R3, R8). `stats` is the per-year table from year_stats(); it supplies the row
    counts and class balance that the text used to assume.

    The year structure is validated first (review R7): the train/test boundary comes from
    config.split_years, not from a positional index, and the pair tables must hold every
    comparison the report quotes (review R10, through pair_row).
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    years = [int(y) for y in years]
    if len(years) != len(set(years)) or years != sorted(years):
        raise ValueError(f"write_report: years must be unique and ordered, got {years}")
    first, last = years[0], years[-1]
    pairs_consec = list(zip(years[:-1], years[1:]))
    n_feat = int(summary["n_features"].iloc[0])
    # R7: the boundary is the project's split policy, validated for this year list.
    train_years, test_years = config.split_years(years)
    if stats is None:
        raise ValueError("write_report: stats (from year_stats) is required for the "
                         "per-year row counts and class balance")
    missing_stats = [y for y in years if y not in stats.index]
    if missing_stats:
        raise KeyError(f"write_report: year_stats has no row for {missing_stats}")

    def row(df, ya, yb, cls="all", what="pair summary"):
        return pair_row(df, ya, yb, cls, what)

    # --- Table: top features by mean KS over consecutive year pairs --------------------
    t1 = feat.head(10).reset_index()
    t1["feature"] = t1["feature"].map(lambda f: f if len(f) <= 95 else f[:92] + "...")
    t1 = pd.DataFrame({
        "feature (API signature)": t1["feature"],
        "mean KS (consecutive years)": t1["mean_ks_consecutive"].round(3),
        f"KS {first} vs {last}": t1["ks_first_last"].round(3),
        f"prevalence {first}": t1["prevalence_first"].round(3),
        f"prevalence {last}": t1["prevalence_last"].round(3),
        f"JS dist. {first} vs {last}": t1["js_first_last"].round(3)})

    # --- Table: top features by KS between first and last year -------------------------
    t2 = top[(top.year_a == first) & (top.year_b == last) & (top["class"] == "all")].copy()
    t2["feature"] = t2["feature"].map(lambda f: f if len(f) <= 95 else f[:92] + "...")
    t2 = pd.DataFrame({
        "rank": t2["rank"], "feature (API signature)": t2["feature"], "KS": t2["ks"].round(3),
        "adj. p (BH)": t2["q_value"].map(lambda q: "<0.001" if q < 0.001 else f"{q:.3f}"),
        f"prevalence {first}": t2["prevalence_a"].round(3),
        f"prevalence {last}": t2["prevalence_b"].round(3),
        "JS dist.": t2["js_distance"].round(3)})

    # --- Table: consecutive-year summary -----------------------------------------------
    rows = []
    for ya, yb in pairs_consec:
        a, b, m = row(summary, ya, yb), row(summary, ya, yb, "benign"), row(summary, ya, yb, "malware")
        rows.append({"years": f"{ya} to {yb}", "rows compared (A / B)": f"{a.n_a} / {a.n_b}",
                     "mean KS": round(a.mean_ks, 4), "mean JS": round(a.mean_js, 4),
                     "mean abs. prevalence change": round(a.mean_abs_prevalence_change, 4),
                     "significant features (BH q<0.05)": a.n_significant,
                     "features with KS>=0.2": a.n_large_effect,
                     "KS>=0.2 benign": b.n_large_effect, "KS>=0.2 malware": m.n_large_effect})
    t3 = pd.DataFrame(rows)

    # --- Table: years compared with the first year -------------------------------------
    rows = []
    for y in years[1:]:
        a, b, m = row(summary, first, y), row(summary, first, y, "benign"), row(summary, first, y, "malware")
        rows.append({"years": f"{first} vs {y}", "mean KS": round(a.mean_ks, 4),
                     "mean JS": round(a.mean_js, 4), "significant features": a.n_significant,
                     "features with KS>=0.2": a.n_large_effect,
                     "KS>=0.2 benign": b.n_large_effect, "KS>=0.2 malware": m.n_large_effect})
    t4 = pd.DataFrame(rows)

    # --- Table: data available per year (R3/R4: availability, not sampled sizes) --------
    t0 = pd.DataFrame({
        "year": [int(y) for y in years],
        "rows available": [int(stats.loc[y, "available_rows"]) for y in years],
        "distinct feature vectors": [int(stats.loc[y, "available_unique_vectors"])
                                     for y in years],
        **({"distinct apknames": [int(stats.loc[y, "unique_apknames"]) for y in years]}
           if "unique_apknames" in stats.columns else {}),
        "benign rows": [int(stats.loc[y, "n_benign"]) for y in years],
        "malware rows": [int(stats.loc[y, "n_malware"]) for y in years],
        "malware share": [_pct(stats.loc[y, "malware_share"], 1) for y in years],
        "used for": ["training" if y in train_years else "test" for y in years]})

    # --- Table: year classifier ---------------------------------------------------------
    # R4: these columns are the sizes the classifier sampled, not the sizes available.
    c = clf.copy()
    c["skipped"] = c["skipped"].astype(bool)
    sel = list(dict.fromkeys(pairs_consec + [(first, y) for y in years[1:]]))

    def auc_cell(ya, yb, cls):
        r = row(c, ya, yb, cls, "year classifier")
        if bool(r.skipped):
            return "not run"
        return f"{r.roc_auc_mean:.2f} +/- {r.roc_auc_std:.2f}"

    t5_rows = []
    for ya, yb in sel:
        r_all = row(c, ya, yb, "all", "year classifier")
        r_mal = row(c, ya, yb, "malware", "year classifier")
        t5_rows.append({
            "years": f"{ya} vs {yb}",
            "AUC all (mean +/- sd)": auc_cell(ya, yb, "all"),
            "AUC benign": auc_cell(ya, yb, "benign"),
            "AUC malware": auc_cell(ya, yb, "malware"),
            "classifier sample size per year (all)": int(r_all.classifier_sample_size),
            "classifier sample size per year (malware)": int(r_mal.classifier_sample_size),
            "identical vectors shared by the pair (all)": int(r_all.n_cross_year_identical)})
    t5 = pd.DataFrame(t5_rows)

    # --- Numbers quoted in the text -----------------------------------------------------
    allc = summary[summary["class"] == "all"]
    consec_all = allc[[(a, b) in set(pairs_consec) for a, b in zip(allc.year_a, allc.year_b)]]
    top_step = consec_all.sort_values("mean_ks", ascending=False).iloc[0]
    low_step = consec_all.sort_values("mean_ks").iloc[0]
    # R3/R8: the most different pair is the one with the largest mean KS, not the pair
    # that happens to be farthest apart in time.
    max_pair = allc.loc[allc["mean_ks"].idxmax()]
    max_is_first_last = (int(max_pair.year_a), int(max_pair.year_b)) == (first, last)
    gap_corr = float(np.corrcoef(allc["year_b"] - allc["year_a"], allc["mean_ks"])[0, 1])
    far = row(summary, first, last)
    near = row(summary, first, years[1])
    far_b, far_m = row(summary, first, last, "benign"), row(summary, first, last, "malware")
    # R7: the boundary step is the last training year against the first test year.
    boundary_a, boundary_b = train_years[-1], test_years[0]
    boundary = row(summary, boundary_a, boundary_b)
    # Rank the boundary step among the consecutive steps instead of calling it "ordinary".
    boundary_rank = int((consec_all["mean_ks"] > boundary.mean_ks).sum()) + 1
    # R8: the class breakdown and the duplicate-sensitivity check describe the largest
    # consecutive step found above, not a fixed pair of years.
    step_a, step_b = int(top_step.year_a), int(top_step.year_b)
    m_step = row(summary, step_a, step_b, "malware")
    b_step = row(summary, step_a, step_b, "benign")
    step_class = "malware" if m_step.n_large_effect >= b_step.n_large_effect else "benign"
    step_hi = m_step if step_class == "malware" else b_step
    step_lo = b_step if step_class == "malware" else m_step
    has_dedup = len(dedup[(dedup["year_a"] == step_a) & (dedup["year_b"] == step_b)
                          & (dedup["class"] == step_class)]) == 1
    if has_dedup:
        step_dedup = row(dedup, step_a, step_b, step_class, "dedup sensitivity")
        dedup_n = int(step_dedup.n_a), int(step_dedup.n_b)
        # Only call the step duplicate-driven if collapsing identical vectors really
        # shrinks it: a quarter off the mean KS or the large-effect count (review R3/R8).
        dedup_shrinks = (step_dedup.mean_ks <= 0.75 * step_hi.mean_ks
                         or step_dedup.n_large_effect <= 0.75 * step_hi.n_large_effect)
    zero = int((feat["mean_ks_consecutive"] == 0).sum())
    emerging = int(((feat["prevalence_first"] < 0.01) & (feat["prevalence_last"] >= 0.10)).sum())
    faded = int(((feat["prevalence_first"] >= 0.10) & (feat["prevalence_last"] < 0.01)).sum())
    big_prev = int((feat["ks_first_last"] >= LARGE_EFFECT_KS).sum())
    # Classifier summaries, skipped comparisons excluded.
    ran = c[~c["skipped"]]
    all_ran = ran[ran["class"] == "all"]
    auc_far = all_ran[all_ran["year_b"] - all_ran["year_a"] >= 3]["roc_auc_mean"]
    auc_adj = all_ran[all_ran["year_b"] - all_ran["year_a"] == 1]["roc_auc_mean"]
    n_skipped = int(c["skipped"].sum())
    max_shared = int(c["n_cross_year_identical"].max())
    # R8: look for a standout consecutive benign pair instead of naming one in advance.
    ben_cons = ran[(ran["class"] == "benign").to_numpy()
                   & np.array([(a, b) in set(pairs_consec)
                               for a, b in zip(ran["year_a"], ran["year_b"])])]
    ben_ranked = ben_cons.sort_values("roc_auc_mean", ascending=False)
    JUMP = 0.05  # descriptive gap needed before the text calls a pair a standout
    has_jump = len(ben_ranked) >= 2 and (ben_ranked.iloc[0].roc_auc_mean
                                         - ben_ranked.iloc[1].roc_auc_mean) >= JUMP
    # Prose about the top features is generated from the ranked table itself (R3).
    top_fl = top[(top.year_a == first) & (top.year_b == last) & (top["class"] == "all")]
    named = ", ".join(f"`{short_name(f)}`" for f in top_fl["feature"].head(5))
    n_rising = int((top_fl["prevalence_b"] > top_fl["prevalence_a"]).sum())
    n_top = len(top_fl)
    n_platform = int(sum(f.startswith(("Landroid/", "Ljava/", "Ljavax/"))
                         for f in top_fl["feature"]))
    shares = stats["malware_share"]
    balance = (f"every year is {_pct(shares.iloc[0])} malware"
               if np.allclose(shares, shares.iloc[0])
               else f"the malware share runs from {_pct(shares.min(), 1)} to "
                    f"{_pct(shares.max(), 1)} across years")

    if cancel is not None:
        # R8: quote the consecutive pair where cancellation is actually largest.
        cz = cancel.loc[cancel["n_in_a_class_opposite_direction"].idxmax()]
    if has_jump:
        jump = ben_ranked.iloc[0]
        jump_a, jump_b = int(jump.year_a), int(jump.year_b)
        jump_second = float(ben_ranked.iloc[1].roc_auc_mean)
        ben_cons_ks = summary[(summary["class"] == "benign").to_numpy()
                              & np.array([(a, b) in set(pairs_consec) for a, b
                                          in zip(summary["year_a"], summary["year_b"])])]
        jump_ks = float(row(summary, jump_a, jump_b, "benign").mean_ks)
        # Compare with the OTHER consecutive benign pairs, not with a maximum that
        # includes this pair itself, or the two numbers are the same by construction.
        others_ks = ben_cons_ks[~((ben_cons_ks["year_a"] == jump_a)
                                  & (ben_cons_ks["year_b"] == jump_b))]["mean_ks"]
        jump_ks_other = float(others_ks.max()) if len(others_ks) else float("nan")
        jump_ks_is_top = len(others_ks) > 0 and jump_ks >= jump_ks_other
    mal_ran = ran[ran["class"] == "malware"]
    min_mal = int(mal_ran["classifier_sample_size"].min()) if len(mal_ran) else 0
    cap = int(c["classifier_sample_size"].max())

    L = [
        "# Step 2 Report: Data Drift Analysis", "",
        "Written by the builder agent. Generated by `python -m src.main --step 2`; all numbers "
        "come from code (tables in `results/step2_*.csv`, figure `results/step2_drift_heatmap.png`).", "",
        "## Method", "",
        f"All {len(years)} years ({', '.join(map(str, years))}) and all {n_feat} API-call features "
        "were used. The features are non-negative integer counts, so drift was measured with "
        "measures designed for counts, not binary data. For **every one of the "
        f"{len(years) * (len(years) - 1) // 2} year pairs** (each year against the first year, "
        "year against the next year, and every other pair) and separately for **all apps, benign "
        "apps and malware apps**, each feature was compared with:", "",
        "- **Two-sample Kolmogorov-Smirnov (KS) statistic** (the assignment's suggestion): the largest "
        "gap between the two empirical distributions, from 0 (same) to 1 (completely different). The "
        "p-value is the asymptotic one. Integer counts contain many ties (most features are 0 for most "
        "apps), so this p-value is only approximate and is not treated as exact.",
        "- **Change in prevalence**: the share of apps with a nonzero count of the API call in the later "
        "year minus the earlier year. It is easy to interpret.",
        "- **Jensen-Shannon (JS) distance** between the binned count distributions (log2 bins: 0, 1, 2, "
        "3-4, 5-8, 9-16, ...; base 2, so 0 to 1). It reacts to changes in how often a call is made, "
        "not only to whether it is used.",
        f"- **Benjamini-Hochberg (BH) correction** across the {n_feat} features, applied separately for "
        "each year pair and each class (that is the correction family). A feature is called "
        f"\"significant\" when its BH-adjusted p-value is below {ALPHA}.",
        f"- **Effect sizes next to significance.** Because the samples are large (up to "
        f"{int(stats['available_rows'].max()):,} rows per year), many small differences are "
        "statistically significant, so the tables also give the number of features with KS >= "
        f"{LARGE_EFFECT_KS}. That cut-off is a descriptive convention chosen in advance, not a "
        "tuned value.", "",
        "**Data available per year** (measured from the loaded files; "
        "`results/step2_year_stats.csv`). The three counts are different things: rows, distinct "
        "feature vectors, and distinct app identifiers. The classifier sample sizes further "
        "below are none of these, they are how many vectors that comparison drew.", "",
        t0.to_markdown(index=False), "",
        "Two more checks:", "",
        "- **Identical feature vectors.** The main numbers keep every row, because Step 1 found many "
        "repeated rows and the student has not yet decided whether to drop them. As a sensitivity "
        "check the year-to-year comparison was repeated with identical feature vectors collapsed "
        "inside each year and class (`results/step2_dedup_sensitivity.csv`). This is "
        "feature-vector deduplication, not application deduplication: two different apps can share "
        "a feature vector, so an identical vector is not evidence of a repeated app. Duplicate "
        "`apkname` identifiers are counted separately in Step 1, and the table above reports rows, "
        "distinct vectors and distinct identifiers side by side.",
        "- **Temporal distinguishability (year classifier).** For a pair of years, a logistic regression "
        "(on log(1 + count) features, standardised inside each training split) was trained to tell "
        "the two years apart, and the held-out ROC AUC is reported (0.5 = the years look the same, "
        "1.0 = perfectly separable). Identical feature vectors are collapsed inside each year and "
        f"class first, both years are sampled to the same size (at most {cap:,} vectors each in this "
        "run), and the train/test split is **group-aware**: one group per distinct feature vector "
        "across the two compared years, so a vector that occurs in both years stays on one side of "
        f"every split. Each comparison is repeated over {N_CLASSIFIER_SPLITS} such splits and the "
        "table reports the mean and standard deviation of the AUC. A comparison whose years hold "
        f"fewer than {MIN_CLASSIFIER_VECTORS} distinct vectors is not run and is shown as \"not "
        f"run\" ({n_skipped} of {len(c)} comparisons). The split is random inside the two-year "
        "sample on purpose: the question is whether the years differ, not whether a model "
        "generalises to later years, so the AUC is a descriptive measure of how separable two "
        "years are and is not a temporal-generalisation result.", "",
        "**What this analysis is not.** It is descriptive only. It was not used to choose the "
        f"train/test split ({_year_runs(train_years)} for training, {_year_runs(test_years)} for "
        "testing, fixed before this step), to select features, or to tune models. It shows that the "
        "**feature distributions** change over time (feature drift) and that years can be told "
        "apart. It does not by itself show concept drift (a change in the relationship between "
        "features and labels); that is discussed in Step 6 together with the change in predictive "
        "performance.", "",
        f"Limits to keep in mind: (1) sample size per year ({_size_phrase(stats)}) - the KS "
        "statistic is noisier (slightly larger by chance) for smaller samples, so mean KS values "
        "are best compared between pairs of similar sample size; (2) tied integer counts make KS "
        "p-values approximate; (3) repeated feature vectors inflate the effective sample size, and "
        "the table above shows how many distinct vectors each year actually holds.", "",
        "## Features that drift most", "",
        f"Of the {n_feat} features, {zero} have a mean KS of exactly 0 between consecutive years (they "
        f"are zero in every app of those years or distributed identically), and {big_prev} have KS >= "
        f"{LARGE_EFFECT_KS} between {first} and {last}. {emerging} features were almost unused in "
        f"{first} (prevalence below 1%) and are used by at least 10% of apps in {last}; {faded} went the "
        "other way.", "",
        f"**Features with the largest average year-to-year change** (mean KS over the "
        f"{len(pairs_consec)} consecutive year pairs, all apps):", "",
        t1.to_markdown(index=False), "",
        f"**Features with the largest change between {first} and {last}** (KS; all apps; adjusted p-values "
        "are BH within that year pair; several rows are different overloads of the same method):", "",
        t2.to_markdown(index=False), "",
        f"That ranking is led by {named} (the names are read from the table above, not from a "
        f"fixed list). {n_platform} of its {n_top} features are Android or Java platform calls "
        "(their signatures start with `Landroid/`, `Ljava/` or `Ljavax/`). "
        f"{n_rising} of the {n_top} grow in prevalence between {first} and {last} and "
        f"{n_top - n_rising} fall, so the change is mostly calls becoming common rather than "
        "disappearing, which fits changes in the Android platform and in how apps are built, not "
        "only changes in malware. The table does not prove the cause. The top features for every "
        "other year pair, and for benign and malware separately, are in "
        "`results/step2_top_features.csv`; the full ranking is in "
        "`results/step2_feature_summary.csv`.", "",
        "## Years that differ most", "",
        "Heatmap of the mean KS and mean JS distance over all years: `results/step2_drift_heatmap.png` "
        "(matrices in `results/step2_ks_matrix.csv` and `results/step2_js_matrix.csv`; per class in "
        "`results/step2_ks_matrix_benign.csv` and `results/step2_ks_matrix_malware.csv`).", "",
        "**Consecutive years (all apps; last two columns split the KS >= 0.2 count by class):**", "",
        t3.to_markdown(index=False), "",
        f"**Each year against {first}:**", "",
        t4.to_markdown(index=False), "",
        f"- **Drift accumulates with the time gap.** The mean KS is {near.mean_ks:.3f} for {first} vs "
        f"{years[1]} and {far.mean_ks:.3f} for {first} vs {last}; over all {len(allc)} year pairs the "
        f"mean KS and the gap in years correlate at {gap_corr:.2f}. The number of features with KS >= "
        f"{LARGE_EFFECT_KS} grows from {near.n_large_effect} to {far.n_large_effect}. Both classes "
        f"drift: between {first} and {last}, {far_b.n_large_effect} features reach KS >= "
        f"{LARGE_EFFECT_KS} for benign apps and {far_m.n_large_effect} for malware.",
        f"- **The most different pair of years.** Taking the largest mean KS over all {len(allc)} "
        f"pairs rather than the largest time gap, it is {int(max_pair.year_a)} vs "
        f"{int(max_pair.year_b)} at mean KS {max_pair.mean_ks:.3f}"
        + (", which is also the pair farthest apart in time."
           if max_is_first_last else
           f", not {first} vs {last} (mean KS {far.mean_ks:.3f}), which is the pair farthest apart "
           "in time."),
        f"- **Consecutive steps are mostly small and similar.** The mean KS between neighbouring years "
        f"ranges from {low_step.mean_ks:.3f} ({int(low_step.year_a)} to {int(low_step.year_b)}) to "
        f"{top_step.mean_ks:.3f} ({step_a} to {step_b}). The largest step is {step_a} to {step_b}, and "
        f"it is concentrated in {step_class} ({step_hi.n_large_effect} features with KS >= "
        f"{LARGE_EFFECT_KS} for {step_class} against {step_lo.n_large_effect} for the other class).",
        *([f"- **Duplicate sensitivity of that step.** In {step_a} and {step_b} the {step_class} "
           f"rows hold {dedup_n[0]} and {dedup_n[1]} distinct feature vectors, out of "
           f"{int(stats.loc[step_a, 'n_' + step_class]):,} and "
           f"{int(stats.loc[step_b, 'n_' + step_class]):,} rows. With identical vectors collapsed, "
           f"the {step_class} step {step_a} to {step_b} has {step_dedup.n_large_effect} features "
           f"with KS >= {LARGE_EFFECT_KS} and mean KS {step_dedup.mean_ks:.3f}, against "
           f"{step_hi.n_large_effect} and {step_hi.mean_ks:.3f} with all rows. "
           # R3/R8: say what the two versions show, whichever way they came out.
           + ("So a large share of this step reflects the weight of repeated feature vectors "
              "rather than many distinct vectors changing. Because identical vectors need not "
              "belong to the same app, that is a statement about the feature representation, not "
              "proof that the same apps recur. The collapsed samples are also small, so neither "
              "version is conclusive."
              if dedup_shrinks else
              "The two versions agree closely, so repeated feature vectors do not explain this "
              "step.")
           + " This should feed the student's decision on duplicates."]
          if has_dedup else []),
        f"- **The train/test boundary.** The step across the fixed boundary ({boundary_a} to "
        f"{boundary_b}) has mean KS {boundary.mean_ks:.3f}, which ranks {boundary_rank} of "
        f"{len(consec_all)} among the consecutive steps ({low_step.mean_ks:.3f} to "
        f"{top_step.mean_ks:.3f}), so it is not an unusual single-year step. The larger drift "
        "between training and test comes from the test years being several years away from the "
        "training years"
        + (f" (for example {boundary_a} vs {last}: mean KS "
           f"{row(summary, boundary_a, last).mean_ks:.3f})." if boundary_a != last else "."), "",
        *([f"- **Pooled counts can hide class-specific change.** The clearest consecutive pair is "
           f"{int(cz.year_a)} to {int(cz.year_b)}: {int(cz.n_in_a_class)} features reach KS >= "
           f"{LARGE_EFFECT_KS} in benign or malware apps, "
           f"{int(cz.n_in_a_class_opposite_direction)} of them with the benign and malware "
           "prevalence moving in strictly opposite directions (a change of exactly zero in either "
           f"class is not counted), but only {int(cz.n_pooled)} reach it in the pooled data (all "
           "apps). This is why the class-specific columns are shown next to the pooled count "
           "(`results/step2_class_cancellation.csv` has every consecutive pair, with the number of "
           "features excluded for a missing class).", ""]
          if cancel is not None else []),
        f"**Year classifier (held-out ROC AUC over {N_CLASSIFIER_SPLITS} group-aware splits; "
        "consecutive years and each year against the first year). The last three columns are the "
        "sizes this comparison sampled, not the data available per year (that is the table in the "
        "Method section):**", "",
        t5.to_markdown(index=False), "",
        f"- Neighbouring years are partly separable (mean AUC {auc_adj.min():.2f} to "
        f"{auc_adj.max():.2f} for all apps)"
        + (f", and years three or more apart are more separable still (mean AUC at least "
           f"{auc_far.min():.2f}), so the model can tell an old year from a recent one, which "
           "agrees with the KS results." if len(auc_far) else
           ", which agrees with the KS results.")
        + " These figures describe how separable two years are under one fixed seed and sampling "
        "scheme; they are not evidence that a model generalises to later years.",
        *([f"- Benign apps stand out for one consecutive pair, {jump_a} to {jump_b} (mean AUC "
           f"{jump.roc_auc_mean:.2f} +/- {jump.roc_auc_std:.2f}, against at most {jump_second:.2f} "
           f"for the other consecutive benign pairs). "
           # R8: check the claim against the KS statistic for the same pair.
           + (f"The KS measure ranks the same pair highest among the consecutive benign pairs "
              f"(mean KS {jump_ks:.3f}, against at most {jump_ks_other:.3f} for the others), so "
              "the two measures agree that this is the largest single-year change for benign apps, "
              "though the AUC gap is much wider than the KS gap."
              if jump_ks_is_top else
              f"The KS measure does not single out that pair (mean KS {jump_ks:.3f}, against up to "
              f"{jump_ks_other:.3f} for the other consecutive benign pairs), so the AUC gap may "
              "come from how the years were sampled; it is worth a look but nothing here explains "
              "it.")]
          if has_jump else
          [f"- No single consecutive benign pair stands out: the mean benign AUCs run from "
           f"{ben_ranked['roc_auc_mean'].min():.2f} to {ben_ranked['roc_auc_mean'].max():.2f} and "
           f"the two highest are within "
           f"{ben_ranked.iloc[0].roc_auc_mean - ben_ranked.iloc[1].roc_auc_mean:.2f} of each other."]
          if len(ben_ranked) >= 2 else []),
        f"- The malware comparisons rest on the fewest distinct feature vectors (the smallest "
        f"sampled {min_mal} per year), so their AUC values are the least reliable in the table"
        + (f", and {n_skipped} of the {len(c)} comparisons were not run at all for holding fewer "
           f"than {MIN_CLASSIFIER_VECTORS} distinct vectors in one of the two years."
           if n_skipped else
           f"; every one of the {len(c)} comparisons had at least "
           f"{MIN_CLASSIFIER_VECTORS} distinct vectors per year and was run.")
        + (f" At most {max_shared} identical feature vectors are shared by a compared pair of "
           "years; the group-aware split keeps each of them on one side, so they cannot raise the "
           "AUC." if max_shared else
           " No identical feature vector is shared by any compared pair of years, so the "
           "group-aware split and a plain random split would agree here."), "",
        "## What this means for the models", "",
        "- **Expect a model trained on the first years to meet different data later on.** Feature drift is "
        f"gradual and cumulative (mean KS {near.mean_ks:.3f} after one year, {far.mean_ks:.3f} after "
        f"{last - first} years), and years several steps apart can be separated almost perfectly from the "
        f"features alone. The test years ({_year_runs(test_years)}) are further from the training "
        f"years ({_year_runs(train_years)}) the later they are, so Step 6 should show whether "
        "accuracy, and recall in particular, falls year by year. This step only shows that the "
        "inputs change; it does not show that accuracy falls.",
        "- **Drift is not limited to one class.** Both benign and malware apps change, so it is not only "
        "a shift in the mix of apps. A shift in class balance is also ruled out as the cause here, "
        f"since {balance} (per-year counts in the table above).",
        f"- **New API calls appear in the later years.** {emerging} features that were almost never used in "
        f"{first} are common in {last}. A model fitted on the training years only sees these calls at "
        "very low counts, and a scaler fitted on those years will give them unusual values on test "
        "data. This is one reason to expect a static model to age, and it is a reason to compare static "
        "training with retraining or incremental updates in Step 7 (the Step 7 scaler stays fixed on the "
        "training years as planned).",
        "- **No change to the plan follows from this step.** The split stays "
        f"{_year_runs(train_years)} / {_year_runs(test_years)}, and no feature selection or tuning "
        "used these results. Whether the feature-label relationship itself changed (concept drift) "
        "is left to Step 6.",
        "- **Open decision for the student: duplicates.** Identical feature vectors are numerous "
        f"(the per-year table above), and they affect the drift numbers (see the {step_a} to "
        f"{step_b} {step_class} step). Nothing was dropped, and the two kinds of duplicate "
        "(identical vectors and repeated `apkname` values) are still reported separately.", ""]
    path.write_text("\n".join(L))
