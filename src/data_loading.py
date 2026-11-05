"""Step 1: load the per-year CSV files and build one combined dataset.

Each file holds one class for one year. The first column (apkname) is an app
identifier: it is NOT a feature, but we keep it as metadata so duplicates can be
analysed. The label and year come from the file name.
"""
import csv
from pathlib import Path

import numpy as np
import pandas as pd

from src import config


def parse_filename(path):
    """Return (year, label) from a file name such as sampled_2019_malware_api.csv.

    Raises ValueError if the name does not match the expected pattern, so an
    unexpected file is never guessed at.
    """
    name = Path(path).name
    match = config.FILE_PATTERN.match(name)
    if match is None:
        raise ValueError(f"File name does not match {config.FILE_PATTERN.pattern}: {name}")
    return int(match.group("year")), config.LABEL_MAP[match.group("cls")]


def load_file(path):
    """Load one CSV and return a DataFrame: apkname, year, label, then 1,848 features.

    Checks the raw file has 1,849 columns before separating apkname, and that the
    feature part has 1,848 columns (the row count varies by file), non-negative integer counts with no missing values.
    """
    year, label = parse_filename(path)
    raw = pd.read_csv(path)
    # Check the shape before dropping anything.
    if raw.shape[1] != config.EXPECTED_COLS or raw.shape[0] < config.MIN_ROWS:
        raise ValueError(f"{Path(path).name}: shape {raw.shape}, expected at least "
                         f"{config.MIN_ROWS} rows and {config.EXPECTED_COLS} columns")
    if raw.columns[0] != config.ID_COLUMN:
        raise ValueError(f"{Path(path).name}: first column is {raw.columns[0]!r}, "
                         f"expected {config.ID_COLUMN!r}")
    # Discard the first column from the features, keeping it only as metadata.
    apkname = raw.iloc[:, 0]
    features = raw.iloc[:, 1:]
    if features.shape != (raw.shape[0], config.EXPECTED_FEATURES):
        raise ValueError(f"{Path(path).name}: feature shape {features.shape}")
    # pandas silently renames repeated headers (a, a -> a, a.1), so check the raw header line.
    with open(path, newline="") as fh:
        header = next(csv.reader(fh))
    if len(set(header)) != len(header):
        raise ValueError(f"{Path(path).name}: duplicate column names in the header")
    if features.isna().any().any():
        raise ValueError(f"{Path(path).name}: missing values in features")
    # Every column must be read as an integer type (a float or text column is rejected,
    # so no value is silently truncated by the conversion below).
    if not all(pd.api.types.is_integer_dtype(t) for t in features.dtypes):
        raise ValueError(f"{Path(path).name}: non-integer feature values")
    if features.min().min() < 0:
        raise ValueError(f"{Path(path).name}: negative feature values")
    # Check the values fit in int32 before converting, so nothing can overflow.
    if features.max().max() > np.iinfo(np.int32).max:
        raise ValueError(f"{Path(path).name}: feature values exceed the int32 range")
    features = features.astype(np.int32)
    out = features.copy()
    out.insert(0, config.LABEL_COLUMN, np.int8(label))
    out.insert(0, config.YEAR_COLUMN, np.int16(year))
    out.insert(0, config.ID_COLUMN, apkname.to_numpy())
    out["source_file"] = Path(path).name
    return out


def build_dataset(input_dir=None):
    """Load every file in input_dir and combine them into one DataFrame.

    Files are loaded per year (benign + malware), then years are concatenated.
    Returns the combined DataFrame with columns: apkname, year, label, the 1,848
    API-call features, and source_file. Raises if a year lacks a class file, a file
    name is unrecognised, or the feature columns differ between files.
    """
    input_dir = Path(input_dir) if input_dir else config.INPUT_DIR
    paths = sorted(input_dir.glob("*.csv"))
    if not paths:
        raise FileNotFoundError(f"No CSV files found in {input_dir}")

    # Group by year, and require one file per class for each year.
    by_year = {}
    for path in paths:
        year, label = parse_filename(path)
        by_year.setdefault(year, {})
        if label in by_year[year]:
            raise ValueError(f"Two files for year {year}, label {label}")
        by_year[year][label] = path
    for year, files in by_year.items():
        if set(files) != set(config.LABEL_MAP.values()):
            raise ValueError(f"Year {year} is missing a class file: has labels {sorted(files)}")

    frames, reference_columns = [], None
    for year in sorted(by_year):
        year_frames = []
        for label in sorted(by_year[year]):
            df = load_file(by_year[year][label])
            cols = list(df.columns)
            if reference_columns is None:
                reference_columns = cols
            elif cols != reference_columns:
                raise ValueError(f"Column names/order differ in {by_year[year][label].name}")
            year_frames.append(df)
        year_df = pd.concat(year_frames, ignore_index=True)  # combine per year
        # Verify the loaded data (not just the file names) holds both classes for this year.
        if set(year_df[config.LABEL_COLUMN].unique()) != set(config.LABEL_MAP.values()):
            raise ValueError(f"Year {year}: loaded data does not contain both classes")
        frames.append(year_df)
    return pd.concat(frames, ignore_index=True)  # then across years


def feature_columns(df):
    """Return the API-call feature column names (everything except metadata)."""
    skip = set(config.METADATA_COLUMNS) | {"source_file"}
    return [c for c in df.columns if c not in skip]


def split_features(df):
    """Return (X, y, meta): the feature matrix, the labels, and the metadata table.

    apkname is never part of X.
    """
    X = df[feature_columns(df)]
    y = df[config.LABEL_COLUMN]
    meta = df[[config.ID_COLUMN, config.YEAR_COLUMN, "source_file"]]
    return X, y, meta
