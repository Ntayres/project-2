"""Step 1: summary tables and duplicate analysis for the combined dataset.

Two kinds of duplicates are counted separately (PLAN.md, leakage rule 4):
  (a) duplicate apkname identifiers, and
  (b) identical API-call feature vectors (which do not necessarily mean the same app).
Nothing is removed here; the counts are only reported.

Counting terms used in every table:
  * "duplicate rows"   = rows beyond the first occurrence of a key (a key seen 3 times
                         gives 2 duplicate rows).
  * "duplicated keys"  = number of distinct keys that occur more than once.
  * "rows involved"    = all rows whose key occurs under more than one year / class.
"""
import numpy as np
import pandas as pd

from src import config
from src.data_loading import feature_columns


def counts_per_year(df):
    """Return a table of files, rows, features, benign, malware, malware share per year."""
    table = df.groupby(config.YEAR_COLUMN)[config.LABEL_COLUMN].agg(
        rows="size", malware="sum")
    table["benign"] = table["rows"] - table["malware"]
    table["malware_share"] = (table["malware"] / table["rows"]).round(4)
    table["files"] = df.groupby(config.YEAR_COLUMN)["source_file"].nunique()
    table["feature_columns"] = len(feature_columns(df))
    return table[["files", "rows", "feature_columns", "benign", "malware", "malware_share"]]


def exact_vector_ids(df):
    """Return (ids, n_collisions): an integer id per row, equal only for identical vectors.

    Rows are first grouped by a hash of the feature vector (fast). Each hash group with
    more than one row is then compared with the real vectors, so a hash collision can
    never be reported as an identical vector. n_collisions counts hash groups that
    contained non-identical vectors (expected: 0).
    """
    values = df[feature_columns(df)].to_numpy()
    hashes = pd.util.hash_pandas_object(df[feature_columns(df)], index=False).to_numpy()
    ids = np.empty(len(df), dtype=np.int64)
    next_id, collisions = 0, 0
    order = pd.Series(np.arange(len(df))).groupby(hashes).apply(list)
    for rows in order:
        if len(rows) == 1:
            ids[rows[0]] = next_id
            next_id += 1
            continue
        # Compare real vectors inside the candidate group.
        exact = {}
        for r in rows:
            exact.setdefault(values[r].tobytes(), []).append(r)
        if len(exact) > 1:
            collisions += 1
        for members in exact.values():
            ids[members] = next_id
            next_id += 1
    return ids, collisions


def duplicate_summary(df):
    """Count duplicates within files and across years/classes, for ids and feature vectors.

    Returns (per_file, cross, totals):
      per_file: one row per source file (duplicate rows and duplicated keys, within file)
      cross:    one row per measure, with unique keys and rows involved
      totals:   overall unique counts and the number of hash collisions found
    """
    vector_ids, collisions = exact_vector_ids(df)
    work = pd.DataFrame({
        "apkname": df[config.ID_COLUMN].to_numpy(),
        "vector": vector_ids,
        "year": df[config.YEAR_COLUMN].to_numpy(),
        "label": df[config.LABEL_COLUMN].to_numpy(),
        "file": df["source_file"].to_numpy(),
    })

    # Within each file: duplicate rows (beyond first occurrence) and duplicated keys.
    rows = []
    for name, g in work.groupby("file"):
        id_counts = g["apkname"].value_counts()
        vec_counts = g["vector"].value_counts()
        rows.append({
            "file": name,
            "rows": len(g),
            "unique_apkname": int(id_counts.size),
            "duplicate_apkname_rows": len(g) - int(id_counts.size),
            "duplicated_apkname_keys": int((id_counts > 1).sum()),
            "unique_vectors": int(vec_counts.size),
            "duplicate_vector_rows": len(g) - int(vec_counts.size),
            "duplicated_vector_keys": int((vec_counts > 1).sum()),
        })
    per_file = pd.DataFrame(rows)

    def across(key, group, label):
        """Unique keys, and rows involved, for keys seen under more than one group value."""
        n_groups = work.groupby(key)[group].transform("nunique")
        keys = work.loc[n_groups > 1, key].nunique()
        return {"measure": label, "unique_keys": int(keys), "rows_involved": int((n_groups > 1).sum())}

    cross = pd.DataFrame([
        across("apkname", "year", "apkname appearing in more than one year"),
        across("apkname", "label", "apkname appearing in both classes"),
        across("vector", "year", "feature vector appearing in more than one year"),
        across("vector", "label", "feature vector appearing in both classes"),
    ])
    totals = {
        "total_rows": len(work),
        "unique_apkname": int(work["apkname"].nunique()),
        "unique_feature_vectors": int(work["vector"].nunique()),
        "hash_collisions_found": collisions,
    }
    return per_file, cross, totals


def write_report(df, per_year, per_file, cross, totals, path):
    """Write the Step 1 builder report (docs/step1_data_summary.md) from computed numbers.

    Sections follow the course template: Files loaded, Shape after dropping the first
    column, Class counts per year, Issues found.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    years = sorted(df[config.YEAR_COLUMN].unique())
    n_features = len(feature_columns(df))
    files = df.groupby(["source_file", config.YEAR_COLUMN, config.LABEL_COLUMN]).size()
    files = files.rename("rows").reset_index()
    files["class"] = files[config.LABEL_COLUMN].map({v: k for k, v in config.LABEL_MAP.items()})
    files["feature_columns"] = n_features

    lines = [
        "# Step 1 Report: Prepare the Data",
        "",
        "Written by the builder agent. Generated by `python -m src.main --step 1`; "
        "all numbers come from code.",
        "",
        "## Files loaded",
        "",
        f"{int(per_year['files'].sum())} files were loaded, covering {len(years)} years "
        f"({', '.join(str(y) for y in years)}), with one benign file and one malware file per year. "
        "The label comes from the file name (malware = 1, benign/goodware = 0).",
        "",
        files[["source_file", config.YEAR_COLUMN, "class"]].rename(
            columns={"source_file": "file"}).to_markdown(index=False),
        "",
        "## Shape after dropping the first column",
        "",
        f"Each raw file has {config.EXPECTED_COLS} columns; the number of rows differs by file "
        f"(see tables below). The first "
        f"column (`{config.ID_COLUMN}`) is not a feature; it is kept only as metadata for the "
        f"duplicate analysis. After removing it from the features, each file has "
        f"{n_features} feature columns.",
        "",
        "Per file:",
        "",
        files[["source_file", "rows", "feature_columns"]].rename(
            columns={"source_file": "file"}).to_markdown(index=False),
        "",
        "Per year (both classes combined):",
        "",
        per_year[["files", "rows", "feature_columns"]].reset_index().to_markdown(index=False),
        "",
        f"Combined dataset: {len(df)} rows x {n_features} features.",
        "",
        "## Class counts per year",
        "",
        per_year[["malware", "benign", "rows", "malware_share"]].rename(
            columns={"benign": "goodware", "rows": "total"}).reset_index().to_markdown(index=False),
        "",
        "## Issues found",
        "",
        "No missing values, non-integer values, negative values, repeated column names, or "
        "mismatched headers were found (the loader stops with an error if any occurs).",
        "",
        "### Duplicates (reported, not removed)",
        "",
        "Duplicate identifiers (`apkname`) and identical feature vectors are counted separately. "
        "Identical vectors do not necessarily mean the same app. Terms: *duplicate rows* = rows "
        "beyond the first occurrence of a key; *duplicated keys* = distinct keys that occur more "
        "than once; *rows involved* = all rows carrying such a key.",
        "",
        "Within each file:",
        "",
        per_file.to_markdown(index=False),
        "",
        "Across years and classes:",
        "",
        cross.to_markdown(index=False),
        "",
        "Totals:",
        "",
        "| Measure | Value |",
        "|---|---|",
    ]
    lines += [f"| {k} | {v} |" for k, v in totals.items()]
    lines += ["",
              "Identical feature vectors were confirmed by comparing the real vectors inside each "
              "hash group, not by the hash alone.",
              "",
              "No rows have been dropped. Whether to drop duplicates is the student's decision.",
              ""]
    path.write_text("\n".join(lines))
