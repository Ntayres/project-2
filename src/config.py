"""Project-wide settings for Project 2 (paths, file-name rules, expected shapes).

Everything that describes the dataset lives here so no other module hard-codes
years, file names, or label strings. Years are discovered from the files.
"""
import re
from pathlib import Path

# Paths are relative to the repository root (no absolute paths).
ROOT = Path(__file__).resolve().parent.parent
INPUT_DIR = ROOT / "input_data"
RESULTS_DIR = ROOT / "results"
DOCS_DIR = ROOT / "docs"
OUTPUTS_DIR = ROOT / "outputs"

RANDOM_STATE = 42

# File names look like sampled_2019_malware_api.csv (class: benign or malware).
FILE_PATTERN = re.compile(r"^sampled_(?P<year>\d{4})_(?P<cls>benign|malware)_api\.csv$")

# Malware is the positive class (label 1); benign (goodware) is 0.
LABEL_MAP = {"benign": 0, "malware": 1}

# Expected shape of every raw file. The column count is an assignment-specific
# expectation (1,849 columns per file); files with another column count are rejected
# on purpose. The row count is NOT fixed: the assignment says 1,500, but the real data
# has 9,000 rows per file for 2014-2018 and 1,500 per file for 2019-2021, so we only
# require at least one row and report the actual row counts.
MIN_ROWS = 1
EXPECTED_COLS = 1849
# The first column is an app identifier. It is metadata only and must NEVER enter the
# model feature matrix (use data_loading.feature_columns / split_features).
ID_COLUMN = "apkname"
EXPECTED_FEATURES = EXPECTED_COLS - 1  # 1,848 API-call features (identifier excluded)

# Names of the columns added to the combined table (not API-call features).
LABEL_COLUMN = "label"
YEAR_COLUMN = "year"
METADATA_COLUMNS = [ID_COLUMN, YEAR_COLUMN, LABEL_COLUMN]

# Train/test split policy from PLAN.md ("Year split policy"): the first half of the
# years available trains, the later half tests. It lives here so no module picks the
# boundary with its own positional index, and so the requirement is validated rather
# than assumed (Step 2 review R7).
MIN_SPLIT_YEARS = 4


def split_years(years):
    """Return (train_years, test_years): first half of the years trains, later half tests.

    Raises ValueError when there are too few years to form the boundary this project
    needs, instead of silently producing an empty or one-year training period.
    """
    ordered = sorted(int(y) for y in years)
    if len(ordered) != len(set(ordered)):
        raise ValueError(f"split_years: repeated years in {ordered}")
    if len(ordered) < MIN_SPLIT_YEARS:
        raise ValueError(f"split_years: need at least {MIN_SPLIT_YEARS} years for a "
                         f"train/test boundary, got {len(ordered)}: {ordered}")
    cut = len(ordered) // 2
    return ordered[:cut], ordered[cut:]
