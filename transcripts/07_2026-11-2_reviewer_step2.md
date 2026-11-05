Nt Ayres: we are now at step 2 here is the new updated code and the instructions the builder was givin&#x20;
Since this dataset spans 8 years, it likely has data drift (feature drift or concept drift).\
Here are some suggestions:\
• Split by Year: Train and test on different years to observe how the models handle data\
drift.\
• Monitoring Drift: Consider using statistical methods (e.g., Kolmogorov-Smirnov test) to\
measure the feature distribution changes over the years.
What are some changes that could be made
config.py&#x20;
"""Project-wide settings for Project 2 (paths, file-name rules, expected shapes).
Everything that describes the dataset lives here so no other module hard-codes
years, file names, or label strings. Years are discovered from the files.
"""
import re
from pathlib import Path
\# Paths are relative to the repository root (no absolute paths).
ROOT = Path(\_\_file\_\_).resolve().parent.parent
INPUT_DIR = ROOT / "input_data"
RESULTS_DIR = ROOT / "results"
DOCS_DIR = ROOT / "docs"
OUTPUTS_DIR = ROOT / "outputs"
RANDOM_STATE = 42
\# File names look like sampled_2019_malware_api.csv (class: benign or malware).
FILE_PATTERN = re.compile(r"^sampled\_(?P\<year>\d{4})\_(?P\<cls>benign|malware)\_ap&#x69;**\\.**&#x63;sv$")
\# Malware is the positive class (label 1); benign (goodware) is 0.
LABEL_MAP = {"benign": 0, "malware": 1}
\# Expected shape of every raw file. The column count is an assignment-specific
\# expectation (1,849 columns per file); files with another column count are rejected
\# on purpose. The row count is NOT fixed: the assignment says 1,500, but the real data
\# has 9,000 rows per file for 2014-2018 and 1,500 per file for 2019-2021, so we only
\# require at least one row and report the actual row counts.
MIN_ROWS = 1
EXPECTED_COLS = 1849
\# The first column is an app identifier. It is metadata only and must NEVER enter the
\# model feature matrix (use data_loading.feature_columns / split_features).
ID_COLUMN = "apkname"
EXPECTED_FEATURES = EXPECTED_COLS - 1  # 1,848 API-call features (identifier excluded)
\# Names of the columns added to the combined table (not API-call features).
LABEL_COLUMN = "label"
YEAR_COLUMN = "year"
METADATA_COLUMNS = [ID_COLUMN, YEAR_COLUMN, LABEL_COLUMN]
data_loading.py&#x20;
"""Step 1: load the per-year CSV files and build one combined dataset.\
\
Each file holds one class for one year. The first column (apkname) is an app\
identifier: it is NOT a feature, but we keep it as metadata so duplicates can be\
analysed. The label and year come from the file name.\
"""\
import csv\
from pathlib import Path\
\
import numpy as np\
import pandas as pd\
\
from src import config\
\
\
def parse_filename(path):\
&#x20;   """Return (year, label) from a file name such as sampled_2019_malware_api.csv.\
\
&#x20;   Raises ValueError if the name does not match the expected pattern, so an\
&#x20;   unexpected file is never guessed at.\
&#x20;   """\
&#x20;   name = Path(path).name\
&#x20;   match = config.FILE_PATTERN.match(name)\
&#x20;   if match is None:\
&#x20;       raise ValueError(f"File name does not match {config.FILE_PATTERN.pattern}: {name}")\
&#x20;   return int(match.group("year")), config.LABEL_MAP[match.group("cls")]\
\
\
def load_file(path):\
&#x20;   """Load one CSV and return a DataFrame: apkname, year, label, then 1,848 features.\
\
&#x20;   Checks the raw file has 1,849 columns before separating apkname, and that the\
&#x20;   feature part has 1,848 columns (the row count varies by file), non-negative integer counts with no missing values.\
&#x20;   """\
&#x20;   year, label = parse_filename(path)\
&#x20;   raw = pd.read_csv(path)\
&#x20;   \# Check the shape before dropping anything.\
&#x20;   if raw\.shape[1] != config.EXPECTED_COLS or raw\.shape[0] < config.MIN_ROWS:\
&#x20;       raise ValueError(f"{Path(path).name}: shape {raw\.shape}, expected at least "\
&#x20;                        f"{config.MIN_ROWS} rows and {config.EXPECTED_COLS} columns")\
&#x20;   if raw\.columns[0] != config.ID_COLUMN:\
&#x20;       raise ValueError(f"{Path(path).name}: first column is {raw\.columns[0]!r}, "\
&#x20;                        f"expected {config.ID_COLUMN!r}")\
&#x20;   \# Discard the first column from the features, keeping it only as metadata.\
&#x20;   apkname = raw\.iloc[:, 0]\
&#x20;   features = raw\.iloc[:, 1:]\
&#x20;   if features.shape != (raw\.shape[0], config.EXPECTED_FEATURES):\
&#x20;       raise ValueError(f"{Path(path).name}: feature shape {features.shape}")\
&#x20;   \# pandas silently renames repeated headers (a, a -> a, a.1), so check the raw header line.\
&#x20;   with open(path, newline="") as fh:\
&#x20;       header = next(csv.reader(fh))\
&#x20;   if len(set(header)) != len(header):\
&#x20;       raise ValueError(f"{Path(path).name}: duplicate column names in the header")\
&#x20;   if features.isna().any().any():\
&#x20;       raise ValueError(f"{Path(path).name}: missing values in features")\
&#x20;   \# Every column must be read as an integer type (a float or text column is rejected,\
&#x20;   \# so no value is silently truncated by the conversion below).\
&#x20;   if not all(pd.api.types.is_integer_dtype(t) for t in features.dtypes):\
&#x20;       raise ValueError(f"{Path(path).name}: non-integer feature values")\
&#x20;   if features.min().min() < 0:\
&#x20;       raise ValueError(f"{Path(path).name}: negative feature values")\
&#x20;   \# Check the values fit in int32 before converting, so nothing can overflow.\
&#x20;   if features.max().max() > np.iinfo(np.int32).max:\
&#x20;       raise ValueError(f"{Path(path).name}: feature values exceed the int32 range")\
&#x20;   features = features.astype(np.int32)\
&#x20;   out = features.copy()\
&#x20;   out.insert(0, config.LABEL_COLUMN, np.int8(label))\
&#x20;   out.insert(0, config.YEAR_COLUMN, np.int16(year))\
&#x20;   out.insert(0, config.ID_COLUMN, apkname.to_numpy())\
&#x20;   out["source_file"] = Path(path).name\
&#x20;   return out\
\
\
def build_dataset(input_dir=None):\
&#x20;   """Load every file in input_dir and combine them into one DataFrame.\
\
&#x20;   Files are loaded per year (benign + malware), then years are concatenated.\
&#x20;   Returns the combined DataFrame with columns: apkname, year, label, the 1,848\
&#x20;   API-call features, and source_file. Raises if a year lacks a class file, a file\
&#x20;   name is unrecognised, or the feature columns differ between files.\
&#x20;   """\
&#x20;   input_dir = Path(input_dir) if input_dir else config.INPUT_DIR\
&#x20;   paths = sorted(input_dir.glob("\*.csv"))\
&#x20;   if not paths:\
&#x20;       raise FileNotFoundError(f"No CSV files found in {input_dir}")\
\
&#x20;   \# Group by year, and require one file per class for each year.\
&#x20;   by_year = {}\
&#x20;   for path in paths:\
&#x20;       year, label = parse_filename(path)\
&#x20;       by_year.setdefault(year, {})\
&#x20;       if label in by_year[year]:\
&#x20;           raise ValueError(f"Two files for year {year}, label {label}")\
&#x20;       by_year[year][label] = path\
&#x20;   for year, files in by_year.items():\
&#x20;       if set(files) != set(config.LABEL_MAP.values()):\
&#x20;           raise ValueError(f"Year {year} is missing a class file: has labels {sorted(files)}")\
\
&#x20;   frames, reference_columns = [], None\
&#x20;   for year in sorted(by_year):\
&#x20;       year_frames = []\
&#x20;       for label in sorted(by_year[year]):\
&#x20;           df = load_file(by_year[year][label])\
&#x20;           cols = list(df.columns)\
&#x20;           if reference_columns is None:\
&#x20;               reference_columns = cols\
&#x20;           elif cols != reference_columns:\
&#x20;               raise ValueError(f"Column names/order differ in {by_year[year][label].name}")\
&#x20;           year_frames.append(df)\
&#x20;       year_df = pd.concat(year_frames, ignore_index=True)  # combine per year\
&#x20;       \# Verify the loaded data (not just the file names) holds both classes for this year.\
&#x20;       if set(year_df[config.LABEL_COLUMN].unique()) != set(config.LABEL_MAP.values()):\
&#x20;           raise ValueError(f"Year {year}: loaded data does not contain both classes")\
&#x20;       frames.append(year_df)\
&#x20;   return pd.concat(frames, ignore_index=True)  # then across years\
\
\
def feature_columns(df):\
&#x20;   """Return the API-call feature column names (everything except metadata)."""\
&#x20;   skip = set(config.METADATA_COLUMNS) | {"source_file"}\
&#x20;   return [c for c in df.columns if c not in skip]\
\
\
def split_features(df):\
&#x20;   """Return (X, y, meta): the feature matrix, the labels, and the metadata table.\
\
&#x20;   apkname is never part of X.\
&#x20;   """\
&#x20;   X = df[feature_columns(df)]\
&#x20;   y = df[config.LABEL_COLUMN]\
&#x20;   meta = df[[config.ID_COLUMN, config.YEAR_COLUMN, "source_file"]]\
&#x20;   return X, y, meta\
data_summary.py
"""Step 1: summary tables and duplicate analysis for the combined dataset.\
\
Two kinds of duplicates are counted separately (PLAN.md, leakage rule 4):\
&#x20; (a) duplicate apkname identifiers, and\
&#x20; (b) identical API-call feature vectors (which do not necessarily mean the same app).\
Nothing is removed here; the counts are only reported.\
\
Counting terms used in every table:\
&#x20; \* "duplicate rows"   = rows beyond the first occurrence of a key (a key seen 3 times\
&#x20;                        gives 2 duplicate rows).\
&#x20; \* "duplicated keys"  = number of distinct keys that occur more than once.\
&#x20; \* "rows involved"    = all rows whose key occurs under more than one year / class.\
"""\
import numpy as np\
import pandas as pd\
\
from src import config\
from src.data_loading import feature_columns\
\
\
def counts_per_year(df):\
&#x20;   """Return a table of files, rows, features, benign, malware, malware share per year."""\
&#x20;   table = df.groupby(config.YEAR_COLUMN)[config.LABEL_COLUMN].agg(\
&#x20;       rows="size", malware="sum")\
&#x20;   table["benign"] = table["rows"] - table["malware"]\
&#x20;   table["malware_share"] = (table["malware"] / table["rows"]).round(4)\
&#x20;   table["files"] = df.groupby(config.YEAR_COLUMN)["source_file"].nunique()\
&#x20;   table["feature_columns"] = len(feature_columns(df))\
&#x20;   return table[["files", "rows", "feature_columns", "benign", "malware", "malware_share"]]\
\
\
def exact_vector_ids(df):\
&#x20;   """Return (ids, n_collisions): an integer id per row, equal only for identical vectors.\
\
&#x20;   Rows are first grouped by a hash of the feature vector (fast). Each hash group with\
&#x20;   more than one row is then compared with the real vectors, so a hash collision can\
&#x20;   never be reported as an identical vector. n_collisions counts hash groups that\
&#x20;   contained non-identical vectors (expected: 0).\
&#x20;   """\
&#x20;   values = df[feature_columns(df)].to_numpy()\
&#x20;   hashes = pd.util.hash_pandas_object(df[feature_columns(df)], index=False).to_numpy()\
&#x20;   ids = np.empty(len(df), dtype=np.int64)\
&#x20;   next_id, collisions = 0, 0\
&#x20;   order = pd.Series(np.arange(len(df))).groupby(hashes).apply(list)\
&#x20;   for rows in order:\
&#x20;       if len(rows) == 1:\
&#x20;           ids[rows[0]] = next_id\
&#x20;           next_id += 1\
&#x20;           continue\
&#x20;       \# Compare real vectors inside the candidate group.\
&#x20;       exact = {}\
&#x20;       for r in rows:\
&#x20;           exact.setdefault(values[r].tobytes(), []).append(r)\
&#x20;       if len(exact) > 1:\
&#x20;           collisions += 1\
&#x20;       for members in exact.values():\
&#x20;           ids[members] = next_id\
&#x20;           next_id += 1\
&#x20;   return ids, collisions\
\
\
def duplicate_summary(df):\
&#x20;   """Count duplicates within files and across years/classes, for ids and feature vectors.\
\
&#x20;   Returns (per_file, cross, totals):\
&#x20;     per_file: one row per source file (duplicate rows and duplicated keys, within file)\
&#x20;     cross:    one row per measure, with unique keys and rows involved\
&#x20;     totals:   overall unique counts and the number of hash collisions found\
&#x20;   """\
&#x20;   vector_ids, collisions = exact_vector_ids(df)\
&#x20;   work = pd.DataFrame({\
&#x20;       "apkname": df[config.ID_COLUMN].to_numpy(),\
&#x20;       "vector": vector_ids,\
&#x20;       "year": df[config.YEAR_COLUMN].to_numpy(),\
&#x20;       "label": df[config.LABEL_COLUMN].to_numpy(),\
&#x20;       "file": df["source_file"].to_numpy(),\
&#x20;   })\
\
&#x20;   \# Within each file: duplicate rows (beyond first occurrence) and duplicated keys.\
&#x20;   rows = []\
&#x20;   for name, g in work.groupby("file"):\
&#x20;       id_counts = g["apkname"].value_counts()\
&#x20;       vec_counts = g["vector"].value_counts()\
&#x20;       rows.append({\
&#x20;           "file": name,\
&#x20;           "rows": len(g),\
&#x20;           "unique_apkname": int(id_counts.size),\
&#x20;           "duplicate_apkname_rows": len(g) - int(id_counts.size),\
&#x20;           "duplicated_apkname_keys": int((id_counts > 1).sum()),\
&#x20;           "unique_vectors": int(vec_counts.size),\
&#x20;           "duplicate_vector_rows": len(g) - int(vec_counts.size),\
&#x20;           "duplicated_vector_keys": int((vec_counts > 1).sum()),\
&#x20;       })\
&#x20;   per_file = pd.DataFrame(rows)\
\
&#x20;   def across(key, group, label):\
&#x20;       """Unique keys, and rows involved, for keys seen under more than one group value."""\
&#x20;       n_groups = work.groupby(key)[group].transform("nunique")\
&#x20;       keys = work.loc[n_groups > 1, key].nunique()\
&#x20;       return {"measure": label, "unique_keys": int(keys), "rows_involved": int((n_groups > 1).sum())}\
\
&#x20;   cross = pd.DataFrame([\
&#x20;       across("apkname", "year", "apkname appearing in more than one year"),\
&#x20;       across("apkname", "label", "apkname appearing in both classes"),\
&#x20;       across("vector", "year", "feature vector appearing in more than one year"),\
&#x20;       across("vector", "label", "feature vector appearing in both classes"),\
&#x20;   ])\
&#x20;   totals = {\
&#x20;       "total_rows": len(work),\
&#x20;       "unique_apkname": int(work["apkname"].nunique()),\
&#x20;       "unique_feature_vectors": int(work["vector"].nunique()),\
&#x20;       "hash_collisions_found": collisions,\
&#x20;   }\
&#x20;   return per_file, cross, totals\
\
\
def write_report(df, per_year, per_file, cross, totals, path):\
&#x20;   """Write the Step 1 builder report (docs/step1_data_summary.md) from computed numbers.\
\
&#x20;   Sections follow the course template: Files loaded, Shape after dropping the first\
&#x20;   column, Class counts per year, Issues found.\
&#x20;   """\
&#x20;   path.parent.mkdir(parents=True, exist_ok=True)\
&#x20;   years = sorted(df[config.YEAR_COLUMN].unique())\
&#x20;   n_features = len(feature_columns(df))\
&#x20;   files = df.groupby(["source_file", config.YEAR_COLUMN, config.LABEL_COLUMN]).size()\
&#x20;   files = files.rename("rows").reset_index()\
&#x20;   files["class"] = files[config.LABEL_COLUMN].map({v: k for k, v in config.LABEL_MAP.items()})\
&#x20;   files["feature_columns"] = n_features\
\
&#x20;   lines = [\
&#x20;       "# Step 1 Report: Prepare the Data",\
&#x20;       "",\
&#x20;       "Written by the builder agent. Generated by \`python -m src.main --step 1\`; "\
&#x20;       "all numbers come from code.",\
&#x20;       "",\
&#x20;       "## Files loaded",\
&#x20;       "",\
&#x20;       f"{int(per_year['files'].sum())} files were loaded, covering {len(years)} years "\
&#x20;       f"({', '.join(str(y) for y in years)}), with one benign file and one malware file per year. "\
&#x20;       "The label comes from the file name (malware = 1, benign/goodware = 0).",\
&#x20;       "",\
&#x20;       files[["source_file", config.YEAR_COLUMN, "class"]].rename(\
&#x20;           columns={"source_file": "file"}).to_markdown(index=False),\
&#x20;       "",\
&#x20;       "## Shape after dropping the first column",\
&#x20;       "",\
&#x20;       f"Each raw file has {config.EXPECTED_COLS} columns; the number of rows differs by file "\
&#x20;       f"(see tables below). The first "\
&#x20;       f"column (\`{config.ID_COLUMN}\`) is not a feature; it is kept only as metadata for the "\
&#x20;       f"duplicate analysis. After removing it from the features, each file has "\
&#x20;       f"{n_features} feature columns.",\
&#x20;       "",\
&#x20;       "Per file:",\
&#x20;       "",\
&#x20;       files[["source_file", "rows", "feature_columns"]].rename(\
&#x20;           columns={"source_file": "file"}).to_markdown(index=False),\
&#x20;       "",\
&#x20;       "Per year (both classes combined):",\
&#x20;       "",\
&#x20;       per_year[["files", "rows", "feature_columns"]].reset_index().to_markdown(index=False),\
&#x20;       "",\
&#x20;       f"Combined dataset: {len(df)} rows x {n_features} features.",\
&#x20;       "",\
&#x20;       "## Class counts per year",\
&#x20;       "",\
&#x20;       per_year[["malware", "benign", "rows", "malware_share"]].rename(\
&#x20;           columns={"benign": "goodware", "rows": "total"}).reset_index().to_markdown(index=False),\
&#x20;       "",\
&#x20;       "## Issues found",\
&#x20;       "",\
&#x20;       "No missing values, non-integer values, negative values, repeated column names, or "\
&#x20;       "mismatched headers were found (the loader stops with an error if any occurs).",\
&#x20;       "",\
&#x20;       "### Duplicates (reported, not removed)",\
&#x20;       "",\
&#x20;       "Duplicate identifiers (\`apkname\`) and identical feature vectors are counted separately. "\
&#x20;       "Identical vectors do not necessarily mean the same app. Terms: \*duplicate rows\* = rows "\
&#x20;       "beyond the first occurrence of a key; \*duplicated keys\* = distinct keys that occur more "\
&#x20;       "than once; \*rows involved\* = all rows carrying such a key.",\
&#x20;       "",\
&#x20;       "Within each file:",\
&#x20;       "",\
&#x20;       per_file.to_markdown(index=False),\
&#x20;       "",\
&#x20;       "Across years and classes:",\
&#x20;       "",\
&#x20;       cross.to_markdown(index=False),\
&#x20;       "",\
&#x20;       "Totals:",\
&#x20;       "",\
&#x20;       "| Measure | Value |",\
&#x20;       "|---|---|",\
&#x20;   ]\
&#x20;   lines += [f"| {k} | {v} |" for k, v in totals.items()]\
&#x20;   lines += ["",\
&#x20;             "Identical feature vectors were confirmed by comparing the real vectors inside each "\
&#x20;             "hash group, not by the hash alone.",\
&#x20;             "",\
&#x20;             "No rows have been dropped. Whether to drop duplicates is the student's decision.",\
&#x20;             ""]\
&#x20;   path.write_text("\n".join(lines))\
Drift.py """Step 2: measure feature drift across years (descriptive only).\
\
The features are non-negative integer API-call counts, so drift is measured with\
measures that suit counts. For every pair of years, and separately for all apps,\
benign apps and malware apps, each of the 1,848 features is compared with:\
&#x20; \* the two-sample Kolmogorov-Smirnov (KS) statistic and its asymptotic p-value\
&#x20;   (integer counts contain many ties, so the p-value is only approximate);\
&#x20; \* the change in prevalence (share of apps with a nonzero count of that call);\
&#x20; \* the Jensen-Shannon distance between the binned count distributions (base 2, so it\
&#x20;   lies in [0, 1]).\
Benjamini-Hochberg (BH) correction is applied across the 1,848 features, separately for\
each year pair and class. A "year classifier" shows how well a simple model can tell two\
years apart (temporal distinguishability; this is NOT evidence of concept drift).\
\
Nothing here is used to choose the train/test split, select features or tune models\
(PLAN.md leakage rule 2). Input data frames are never modified.\
"""\
from itertools import combinations\
\
import numpy as np\
import pandas as pd\
from scipy import stats\
from sklearn.linear_model import LogisticRegression\
from sklearn.metrics import roc_auc_score\
from sklearn.model_selection import train_test_split\
from sklearn.pipeline import make_pipeline\
from sklearn.preprocessing import StandardScaler\
\
from src import config\
from src.data_loading import feature_columns\
\
CLASSES = ["all", "benign", "malware"]\
LARGE_EFFECT_KS = 0.2   # descriptive convention for a "large" KS effect, not a tuned value\
ALPHA = 0.05            # BH false discovery rate level\
\
\
def bh_adjust(p):\
&#x20;   """Return Benjamini-Hochberg adjusted p-values (never smaller than the raw p-values)."""\
&#x20;   p = np.asarray(p, dtype=float)\
&#x20;   n = len(p)\
&#x20;   order = np.argsort(p)\
&#x20;   scaled = p[order] \* n / np.arange(1, n + 1)\
&#x20;   scaled = np.minimum.accumulate(scaled[::-1])[::-1]  # enforce monotonicity\
&#x20;   adjusted = np.empty(n)\
&#x20;   adjusted[order] = np.clip(scaled, 0.0, 1.0)\
&#x20;   return adjusted\
\
\
def count_bins(values):\
&#x20;   """Map counts to log2 bins: 0, 1, 2, 3-4, 5-8, 9-16, ... (returns an int8 array)."""\
&#x20;   v = np.asarray(values)\
&#x20;   idx = np.zeros(v.shape, dtype=np.int8)\
&#x20;   nz = v > 0\
&#x20;   idx[nz] = 1 + np.ceil(np.log2(v[nz])).astype(np.int8)\
&#x20;   return idx\
\
\
def bin_histogram(values, n_bins):\
&#x20;   """Return an (n_features, n_bins) matrix of how many apps fall in each count bin."""\
&#x20;   idx = count_bins(values)\
&#x20;   return np.stack([(idx == b).sum(axis=0) for b in range(n_bins)], axis=1)\
\
\
def js_distance(hist_a, hist_b):\
&#x20;   """Row-wise Jensen-Shannon distance (base 2) between two count-histogram matrices."""\
&#x20;   p = hist_a / hist_a.sum(axis=1, keepdims=True)\
&#x20;   q = hist_b / hist_b.sum(axis=1, keepdims=True)\
&#x20;   m = 0.5 \* (p + q)\
\
&#x20;   def kl(a, b):\
&#x20;       with np.errstate(divide="ignore", invalid="ignore"):\
&#x20;           term = np.where(a > 0, a \* np.log2(a / b), 0.0)\
&#x20;       return term.sum(axis=1)\
\
&#x20;   return np.sqrt(np.clip(0.5 \* kl(p, m) + 0.5 \* kl(q, m), 0.0, 1.0))\
\
\
def compare_years(a, b, hist_a, hist_b):\
&#x20;   """Compare two row blocks (rows x features) feature by feature.\
\
&#x20;   Returns arrays (ks, p, prevalence_a, prevalence_b, js). p is the asymptotic KS p-value.\
&#x20;   """\
&#x20;   n_features = a.shape[1]\
&#x20;   ks = np.empty(n_features)\
&#x20;   p = np.empty(n_features)\
&#x20;   for j in range(n_features):\
&#x20;       res = stats.ks_2samp(a[:, j], b[:, j], method="asymp")\
&#x20;       ks[j], p[j] = res.statistic, res.pvalue\
&#x20;   return ks, p, (a > 0).mean(axis=0), (b > 0).mean(axis=0), js_distance(hist_a, hist_b)\
\
\
def \_blocks(X, years, labels, keep=None):\
&#x20;   """Split the matrix into per (year, class) blocks; class 'all' joins both labels.\
\
&#x20;   keep is an optional boolean mask of rows to use (used for the de-duplicated check).\
&#x20;   Returns {(year, class): (matrix, histogram)}.\
&#x20;   """\
&#x20;   n_bins = int(np.ceil(np.log2(max(int(X.max()), 2)))) + 2\
&#x20;   out, hists = {}, {}\
&#x20;   for year in np.unique(years):\
&#x20;       for cls, lab in (("benign", 0), ("malware", 1)):\
&#x20;           mask = (years == year) & (labels == lab)\
&#x20;           if keep is not None:\
&#x20;               mask &= keep\
&#x20;           out[(year, cls)] = X[mask]\
&#x20;           hists[(year, cls)] = bin_histogram(out[(year, cls)], n_bins)\
&#x20;       out[(year, "all")] = np.vstack([out[(year, "benign")], out[(year, "malware")]])\
&#x20;       hists[(year, "all")] = hists[(year, "benign")] + hists[(year, "malware")]\
&#x20;   return out, hists\
\
\
def feature_drift(X, years, labels, names, pairs=None, keep=None):\
&#x20;   """Return one long table: one row per (year pair, class, feature).\
\
&#x20;   Columns: year_a, year_b, class, feature, ks, p_value, q_value (BH within the family of\
&#x20;   all features for that year pair and class), prevalence_a, prevalence_b,\
&#x20;   prevalence_change (b minus a), js_distance, n_a, n_b.\
&#x20;   """\
&#x20;   year_list = sorted(np.unique(years))\
&#x20;   pairs = pairs if pairs is not None else list(combinations(year_list, 2))\
&#x20;   blocks, hists = \_blocks(X, years, labels, keep)\
&#x20;   frames = []\
&#x20;   for ya, yb in pairs:\
&#x20;       for cls in CLASSES:\
&#x20;           a, b = blocks[(ya, cls)], blocks[(yb, cls)]\
&#x20;           ks, p, pa, pb, js = compare_years(a, b, hists[(ya, cls)], hists[(yb, cls)])\
&#x20;           frames.append(pd.DataFrame({\
&#x20;               "year_a": int(ya), "year_b": int(yb), "class": cls, "feature": names,\
&#x20;               "ks": ks, "p_value": p, "q_value": bh_adjust(p),\
&#x20;               "prevalence_a": pa, "prevalence_b": pb, "prevalence_change": pb - pa,\
&#x20;               "js_distance": js, "n_a": len(a), "n_b": len(b)}))\
&#x20;   return pd.concat(frames, ignore_index=True)\
\
\
def pair_summary(drift):\
&#x20;   """Summarise per (year pair, class): significant features, mean KS, mean JS, etc."""\
&#x20;   g = drift.groupby(["year_a", "year_b", "class"])\
&#x20;   out = g.agg(mean_ks=("ks", "mean"), mean_js=("js_distance", "mean"),\
&#x20;               mean_abs_prevalence_change=("prevalence_change", lambda s: s.abs().mean()),\
&#x20;               n_significant=("q_value", lambda s: int((s < ALPHA).sum())),\
&#x20;               n_large_effect=("ks", lambda s: int((s >= LARGE_EFFECT_KS).sum())),\
&#x20;               n_features=("ks", "size"), n_a=("n_a", "first"), n_b=("n_b", "first"))\
&#x20;   return out.reset_index()\
\
\
tHIS IS NOT THE FULL THING YET DONT DO ANYTHING, THE LAST FILE WAS CUT OFF
\

chatgpt: Understood. I'll wait until you send the rest of Drift.py and any remaining files or instructions. I won't review or suggest changes until you tell me you've sent everything.

Nt Aryres:def symmetric_matrix(summary, value, cls, years):
    """Turn the pair summary into a symmetric years x years matrix with a zero diagonal."""
    mat = pd.DataFrame(0.0, index=years, columns=years)
    sub = summary[summary["class"] == cls]
    for row in sub.itertuples():
        mat.loc[row.year_a, row.year_b] = getattr(row, value)
        mat.loc[row.year_b, row.year_a] = getattr(row, value)
    return mat


def year_classifier(X, years, labels, pairs, cls, per_year=2000, seed=config.RANDOM_STATE):
    """Temporal distinguishability: how well can a model tell year A from year B?

    Rows with an identical feature vector inside one year (and class) are collapsed first,
    and each year is sampled to the same size, so copies of one app cannot appear on both
    sides of the split and year sizes do not matter. The split here is a random split inside
    the two-year sample on purpose: the question is "are the years different", not "does a
    model generalise to later years". Reports held-out ROC AUC (0.5 = indistinguishable).
    """
    rng = np.random.RandomState(seed)
    rows = []
    for ya, yb in pairs:
        samples = []
        for y in (ya, yb):
            mask = (years == y) if cls == "all" else (years == y) & (labels == (cls == "malware"))
            idx = np.flatnonzero(mask)
            _, first = np.unique(X[idx], axis=0, return_index=True)   # unique vectors
            idx = idx[np.sort(first)]
            samples.append(idx)
        n = min(per_year, *(len(s) for s in samples))
        chosen = [rng.choice(s, n, replace=False) for s in samples]
        rows_idx = np.concatenate(chosen)
        target = np.repeat([0, 1], n)
        Xp = np.log1p(X[rows_idx].astype(float))
        Xtr, Xte, ytr, yte = train_test_split(Xp, target, test_size=0.3, random_state=seed,
                                              stratify=target)
        model = make_pipeline(StandardScaler(), LogisticRegression(
            C=0.1, max_iter=300, random_state=seed))
        model.fit(Xtr, ytr)
        auc = roc_auc_score(yte, model.predict_proba(Xte)[:, 1])
        rows.append({"year_a": int(ya), "year_b": int(yb), "class": cls,
                     "rows_per_year": n, "roc_auc": auc})
    return pd.DataFrame(rows)


def class_cancellation(table, pairs):
    """For each year pair: features with KS >= the large-effect cut-off in the pooled data,
    in at least one class, and in at least one class while the benign and malware prevalence
    move in opposite directions (so the pooled distribution can hide the change)."""
    rows = []
    for ya, yb in pairs:
        w = table[(table.year_a == ya) & (table.year_b == yb)].pivot(
            index="feature", columns="class", values=["ks", "prevalence_change"])
        any_class = (w["ks"]["benign"] >= LARGE_EFFECT_KS) | (w["ks"]["malware"] >= LARGE_EFFECT_KS)
        opposite = np.sign(w["prevalence_change"]["benign"]) * np.sign(
            w["prevalence_change"]["malware"]) < 0
        rows.append({"year_a": ya, "year_b": yb,
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


def _pct(x):
    """Format a share as a percentage with no decimals."""
    return f"{100 * x:.0f}%"


def write_report(summary, feat, top, clf, dedup, years, path, cancel=None):
    """Write docs/step2_drift_analysis.md using the course template headings.

    Every number in the text and tables is computed from the result tables passed in.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    first, last = years[0], years[-1]
    pairs_consec = list(zip(years[:-1], years[1:]))
    n_feat = int(summary["n_features"].iloc[0])

    def row(df, ya, yb, cls="all"):
        return df[(df.year_a == ya) & (df.year_b == yb) & (df["class"] == cls)].iloc[0]

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
        rows.append({"years": f"{ya} to {yb}", "rows (A / B)": f"{a.n_a} / {a.n_b}",
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

    # --- Table: year classifier ---------------------------------------------------------
    c = clf.copy()
    wide = c.pivot_table(index=["year_a", "year_b"], columns="class", values="roc_auc").round(2)
    size = c[c["class"] == "malware"].set_index(["year_a", "year_b"])["rows_per_year"]
    size_all = c[c["class"] == "all"].set_index(["year_a", "year_b"])["rows_per_year"]
    sel = [p for p in pairs_consec + [(first, y) for y in years[1:]]]
    sel = list(dict.fromkeys(sel))
    t5 = pd.DataFrame([{"years": f"{ya} vs {yb}",
                        "AUC all": wide.loc[(ya, yb), "all"],
                        "AUC benign": wide.loc[(ya, yb), "benign"],
                        "AUC malware": wide.loc[(ya, yb), "malware"],
                        "unique rows per year (all)": int(size_all.loc[(ya, yb)]),
                        "unique rows per year (malware)": int(size.loc[(ya, yb)])}
                       for ya, yb in sel])

    # --- Numbers quoted in the text -----------------------------------------------------
    allc = summary[summary["class"] == "all"]
    consec_all = allc[[(a, b) in set(pairs_consec) for a, b in zip(allc.year_a, allc.year_b)]]
    top_step = consec_all.sort_values("mean_ks", ascending=False).iloc[0]
    low_step = consec_all.sort_values("mean_ks").iloc[0]
    far = row(summary, first, last)
    near = row(summary, first, years[1])
    far_b, far_m = row(summary, first, last, "benign"), row(summary, first, last, "malware")
    boundary = row(summary, years[3], years[4])
    m_step = row(summary, years[-2], last, "malware")
    b_step = row(summary, years[-2], last, "benign")
    m_dedup = row(dedup, years[-2], last, "malware")
    m_dedup_n = int(m_dedup.n_a), int(m_dedup.n_b)
    zero = int((feat["mean_ks_consecutive"] == 0).sum())
    emerging = int(((feat["prevalence_first"] < 0.01) & (feat["prevalence_last"] >= 0.10)).sum())
    faded = int(((feat["prevalence_first"] >= 0.10) & (feat["prevalence_last"] < 0.01)).sum())
    big_prev = int((feat["ks_first_last"] >= 0.2).sum())
    auc_far = c[(c["class"] == "all") & (c.year_b - c.year_a >= 3)]["roc_auc"].min()
    auc_adj = c[(c["class"] == "all") & (c.year_b - c.year_a == 1)]["roc_auc"]
    b_jump = wide.loc[(years[-4], years[-3]), "benign"]
    b_other = wide[[idx[1] - idx[0] == 1 for idx in wide.index]]["benign"].drop(
        (years[-4], years[-3])).max()

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
        f"- **Effect sizes next to significance.** Because the samples are large (up to 18,000 rows per "
        "year), many small differences are statistically significant, so the tables also give the "
        f"number of features with KS >= {LARGE_EFFECT_KS}. That cut-off is a descriptive convention "
        "chosen in advance, not a tuned value.", "",
        "Two more checks:", "",
        "- **Duplicates.** The main numbers keep every row, because Step 1 found many repeated feature "
        "vectors and the student has not yet decided whether to drop them. As a sensitivity check the "
        "year-to-year comparison was repeated with identical feature vectors collapsed inside each "
        "year and class (`results/step2_dedup_sensitivity.csv`).",
        "- **Temporal distinguishability (year classifier).** For a pair of years, a logistic regression "
        "(on log(1 + count) features, standardised) was trained to tell the two years apart, and the "
        "held-out ROC AUC is reported (0.5 = the years look the same, 1.0 = perfectly separable). Within "
        "each year, identical vectors were collapsed first and both years were sampled to the same size "
        "(at most 2,000 rows each), so repeated apps cannot sit on both sides of the split. A random "
        "split inside the two-year sample is intended here; the question is whether the years differ, "
        "not whether a model generalises to later years.", "",
        "**What this analysis is not.** It is descriptive only. It was not used to choose the "
        f"train/test split ({years[0]}-{years[3]} for training, {years[4]}-{years[-1]} for testing, "
        "fixed before this step), to select features, or to tune models. It shows that the **feature "
        "distributions** change over time (feature drift) and that years can be told apart. It does not "
        "by itself show concept drift (a change in the relationship between features and labels); that "
        "is discussed in Step 6 together with the change in predictive performance.", "",
        "Limits to keep in mind: (1) the years have different sizes (18,000 rows per year for "
        f"{years[0]}-{years[4]} and 3,000 for {years[5]}-{years[-1]}), and the KS statistic is noisier "
        "(slightly larger by chance) for smaller samples, so mean KS values are best compared between "
        "pairs of similar sample size; (2) tied integer counts make KS p-values approximate; (3) repeated "
        "apps inflate the effective sample size.", "",
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
        "The top features are common platform calls (`ContentResolver.query`, `Process.myPid`, "
        "`PowerManager$WakeLock.acquire`, `WebView.evaluateJavascript`, `NotificationManager.notify`, "
        "`StringBuilder.append` overloads). Many of them rise from rare to common over the period, which "
        "fits changes in the Android platform and in how apps are built, not only changes in malware. The "
        "table does not prove the cause. The top features for every other year pair, and for benign and "
        "malware separately, are in `results/step2_top_features.csv`; the full ranking is in "
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
        f"{years[1]} and {far.mean_ks:.3f} for {first} vs {last}, which is the most different pair of "
        f"years. The number of features with KS >= {LARGE_EFFECT_KS} grows from {near.n_large_effect} to "
        f"{far.n_large_effect}. Both classes drift: between {first} and {last}, "
        f"{far_b.n_large_effect} features reach KS >= {LARGE_EFFECT_KS} for benign apps and "
        f"{far_m.n_large_effect} for malware.",
        f"- **Consecutive steps are mostly small and similar.** The mean KS between neighbouring years "
        f"ranges from {low_step.mean_ks:.3f} ({low_step.year_a} to {low_step.year_b}) to "
        f"{top_step.mean_ks:.3f} ({top_step.year_a} to {top_step.year_b}). The largest step is "
        f"{top_step.year_a} to {top_step.year_b}, and it is concentrated in malware "
        f"({m_step.n_large_effect} features with KS >= {LARGE_EFFECT_KS} for malware against "
        f"{b_step.n_large_effect} for benign).",
        f"- **Caution on that last step.** In {years[-2]} and {last} the malware files hold only "
        f"{m_dedup_n[0]} and {m_dedup_n[1]} distinct feature vectors out of 1,500 rows each (Step 1). With "
        f"identical vectors collapsed, the malware step {years[-2]} to {last} has {m_dedup.n_large_effect} "
        f"features with KS >= {LARGE_EFFECT_KS} and mean KS {m_dedup.mean_ks:.3f} (against "
        f"{m_step.n_large_effect} and {m_step.mean_ks:.3f} with all rows). So a large share of this jump "
        "reflects the weight of a few repeated vectors, not many different apps changing. The samples "
        "are also very small after collapsing, so neither version is conclusive. This should feed the "
        "student's decision on duplicates.",
        f"- **The train/test boundary.** The step across the fixed boundary ({years[3]} to {years[4]}) has "
        f"mean KS {boundary.mean_ks:.3f}, an ordinary single-year step. The larger drift between "
        f"training and test comes from the test years being several years away from the training years "
        f"(for example {years[3]} vs {last}: mean KS {row(summary, years[3], last).mean_ks:.3f}).", "",
        *([f"- **Pooled counts can hide class-specific change.** Between {years[0]} and {years[1]}, "
           f"{int(cancel.iloc[0].n_in_a_class)} features reach KS >= {LARGE_EFFECT_KS} in benign or malware "
           f"apps, {int(cancel.iloc[0].n_in_a_class_opposite_direction)} of them with the benign and "
           f"malware prevalence moving in opposite directions, but only {int(cancel.iloc[0].n_pooled)} do so "
           "in the pooled data (all apps). This is why the class-specific columns are shown next to the "
           "pooled count (`results/step2_class_cancellation.csv` has every consecutive pair).", ""]
          if cancel is not None else []),
        "**Year classifier (held-out ROC AUC; consecutive years and each year against the first year):**", "",
        t5.to_markdown(index=False), "",
        f"- Neighbouring years are partly separable (AUC {auc_adj.min():.2f} to {auc_adj.max():.2f} for all "
        f"apps), and years three or more apart are highly separable (AUC at least {auc_far:.2f}). "
        "The model can tell an old year from a recent one, which agrees with the KS results.",
        f"- Benign apps show a jump between {years[-4]} and {years[-3]} (AUC {b_jump:.2f}, against at most "
        f"{b_other:.2f} for the other consecutive benign pairs). The KS and JS measures do not show a "
        "matching jump, so this may come from how the two groups of years were sampled; it is worth a look "
        "but nothing here explains it.",
        "- The malware AUC values for the latest years rest on very few distinct vectors (see the last "
        "column) and are unreliable.", "",
        "## What this means for the models", "",
        "- **Expect a model trained on the first years to meet different data later on.** Feature drift is "
        f"gradual and cumulative (mean KS {near.mean_ks:.3f} after one year, {far.mean_ks:.3f} after "
        f"{last - first} years), and years several steps apart can be separated almost perfectly from the "
        f"features alone. The test years ({years[4]}-{last}) are further from the training years "
        f"({years[0]}-{years[3]}) the later they are, so Step 6 should show whether accuracy, and recall "
        "in particular, falls year by year. This step only shows that the inputs change; it does not "
        "show that accuracy falls.",
        "- **Drift is not limited to one class.** Both benign and malware apps change, so it is not only "
        "a shift in the mix of apps. A shift in class balance is also ruled out as the cause here, since "
        "every year has 50% malware (Step 1).",
        f"- **New API calls appear in the later years.** {emerging} features that were almost never used in "
        f"{first} are common in {last}. A model fitted on the training years only sees these calls at "
        "very low counts, and a scaler fitted on those years will give them unusual values on test "
        "data. This is one reason to expect a static model to age, and it is a reason to compare static "
        "training with retraining or incremental updates in Step 7 (the Step 7 scaler stays fixed on the "
        "training years as planned).",
        "- **No change to the plan follows from this step.** The split stays "
        f"{years[0]}-{years[3]} / {years[4]}-{last}, and no feature selection or tuning used these "
        "results. Whether the feature-label relationship itself changed (concept drift) is left to "
        "Step 6.",
        "- **Open decision for the student: duplicates.** Repeated vectors are numerous, especially in the "
        f"malware files, and they affect the drift numbers (see the {years[-2]} to {last} malware step). "
        "Nothing was dropped.", ""]
    path.write_text("\n".join(L))
main.py
"""Entry point: run project steps.

Usage:
    python -m src.main --step 1
    python -m src.main --step 2
    python -m src.main --all
    python -m src.main --step 1 --input-dir path/to/input_data
"""
import argparse

import numpy as np

from src import config, drift
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
    #    collapsed inside each (year, class), because Step 1 found many repeated vectors.
    keep = np.zeros(len(df), dtype=bool)
    for (y, lab), idx in pd.Series(np.arange(len(df))).groupby([years, labels]):
        idx = idx.to_numpy()
        _, first = np.unique(X[idx], axis=0, return_index=True)
        keep[idx[first]] = True
    dedup = drift.pair_summary(drift.feature_drift(X, years, labels, names, consecutive, keep))
    dedup.to_csv(config.RESULTS_DIR / "step2_dedup_sensitivity.csv", index=False)

    # 5. Temporal distinguishability (year classifier). Descriptive, not concept drift.
    clf = [drift.year_classifier(X, years, labels, all_pairs, "all")]
    for cls in ("benign", "malware"):
        clf.append(drift.year_classifier(X, years, labels, consecutive + vs_first, cls))
    pd.concat(clf, ignore_index=True).drop_duplicates(["year_a", "year_b", "class"]).to_csv(
        config.RESULTS_DIR / "step2_year_classifier.csv", index=False)

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
        ax.set_title(f"{title} over 1,848 features (all apps)")
        fig.colorbar(im, ax=ax, fraction=0.046)
    fig.tight_layout()
    fig.savefig(config.RESULTS_DIR / "step2_drift_heatmap.png", dpi=150)
    plt.close(fig)

    cancel = drift.class_cancellation(table, consecutive)
    cancel.to_csv(config.RESULTS_DIR / "step2_class_cancellation.csv", index=False)
    drift.write_report(summary, feat, top, pd.read_csv(config.RESULTS_DIR / "step2_year_classifier.csv"),
                       dedup, year_list, config.DOCS_DIR / "step2_drift_analysis.md", cancel)
    print(summary[summary["class"] == "all"].to_string())
    return table


def main():
    parser = argparse.ArgumentParser(description="Project 2 pipeline")
    parser.add_argument("--step", type=int, choices=[1, 2], help="run one step")
    parser.add_argument("--all", action="store_true", help="run all implemented steps (currently 1 and 2)")
    parser.add_argument("--input-dir", default=None, help="folder with the CSV files")
    args = parser.parse_args()
    if args.all or args.step == 1:
        run_step1(args.input_dir)
    if args.all or args.step == 2:
        run_step2(args.input_dir)
    if not args.all and args.step is None:
        parser.print_help()


if __name__ == "__main__":
    main()

chatgpt: I’ll review this as the Step 2 reviewer, focusing on whether the drift analysis is methodologically sound, whether the implementation matches the assignment and plan, and whether the report makes claims supported by the code.

One important limitation: I can review the code you pasted, but I haven't run it against the dataset or seen the generated CSVs and report. So I’ll distinguish definite code issues from things that need runtime verification.

# Step 2 code review

Changes recommended before accepting Step 2

## 1. High-priority issues

### R1. The year classifier does not use the same sampling procedure for all classes

In `year_classifier`, the code uses:

```
mask = (years == y) if cls == "all" else (years == y) & (labels == (cls == "malware"))
```

This works because `cls == "malware"` evaluates to `True` for malware and `False` for benign, matching labels 1 and 0. However, it is unnecessarily implicit and easy to misunderstand.

I'd change it to:

```
if cls == "all":    mask = years == yelse:    label = 1 if cls == "malware" else 0    mask = (years == y) & (labels == label)
```

This is a readability and maintainability improvement, rather than a current correctness bug.

### R2. The year classifier's duplicate handling is incomplete

The code collapses identical feature vectors within each year, which is good. But it does not collapse them across the two years being compared.

That means an identical feature vector appearing in both years can still appear on both sides of the train/test split. The model may learn to distinguish years based on other differences, but repeated vectors can affect the classifier's evaluation and its interpretation.

More importantly, the report says:

> repeated apps cannot sit on both sides of the split.

That is not necessarily true. The code collapses vectors separately within each year, not across the combined two-year sample.

Recommendation: Either:

- explicitly identify vectors shared across the two years and handle them consistently, or
- revise the report so it accurately describes the current procedure and its limitation.

Do not claim that the split prevents repeated apps from crossing between train and test unless the code actually enforces that.

### R3. `count_bins()` has a potential integer overflow

Current code:

```
idx[nz] = 1 + np.ceil(np.log2(v[nz])).astype(np.int8)
```

Using `int8` limits the bin index to 127. The current dataset may never approach that range, but this is an unnecessary constraint.

I'd use:

```
idx = np.zeros(v.shape, dtype=np.int32)
```

and:

```
idx[nz] = 1 + np.ceil(np.log2(v[nz])).astype(np.int32)
```

This is a small defensive improvement.

### R4. The report contains hard-coded assumptions about dataset sizes

The report says:

```
"18,000 rows per year for ...""3,000 for ..."
```

It also says the latest malware files have 1,500 rows each.

These statements may be accurate for the dataset currently being used, but they should be calculated from the actual data rather than hard-coded.

The report already calculates `n_a` and `n_b`, so use those values instead.

For example, replace the fixed-size explanation with something like:

> The year groups differ in sample size. This affects the precision of the statistical comparisons, so results should be interpreted alongside the sample counts reported in the tables.

This avoids a report that becomes inaccurate if the dataset changes.

### R5. The class cancellation analysis should handle missing values defensively

Current code:

```
w = table[(table.year_a == ya) & (table.year_b == yb)].pivot(    index="feature", columns="class", values=["ks", "prevalence_change"])
```

Then:

```
any_class = (w["ks"]["benign"] >= LARGE_EFFECT_KS) | (w["ks"]["malware"] >= LARGE_EFFECT_KS)
```

This assumes every feature has exactly one result for every class. That should be true if the earlier drift calculation completes correctly, but the code does not explicitly validate it.

I would add an assertion before calculating the cancellation statistics to ensure the pivot contains all three classes and all expected features.

For example:

```
required_classes = {"all", "benign", "malware"}if set(w["ks"].columns) != required_classes:    raise ValueError(f"Missing class results for {ya} vs {yb}")if len(w) != len(names):    raise ValueError(f"Incomplete feature results for {ya} vs {yb}")
```

The exact implementation may need to pass the expected feature count into the function.

## 2. Statistical methodology

### R6. KS p-values should not be treated as definitive evidence

The code correctly acknowledges that the KS test's asymptotic p-values are approximate because the features are discrete integer counts with many ties.

That is good.

However, the report should emphasize that KS effect sizes and JS distances are more useful for interpreting the magnitude of the changes than p-values alone.

The code already calculates:

- KS statistic
- BH-adjusted p-value
- Prevalence change
- Jensen-Shannon distance

I would keep all four.

### R7. The JS distance binning needs a little more explanation

The current bins are:

- 0
- 1
- 2
- 3–4
- 5–8
- 9–16
- and so on

This is reasonable for count data because it distinguishes zero from nonzero values while grouping increasingly large counts.

However, the report should explicitly say that JS distance is calculated on the binned distributions, not the raw count distributions.

It currently does this, but I'd make the distinction especially clear in the report's method section.

### R8. The year classifier is descriptive, not evidence of concept drift

This is one of the strongest parts of the implementation.

The code explicitly calls the year classifier a measure of temporal distinguishability and says it is not evidence of concept drift.

Keep that distinction.

A model distinguishing 2014 from 2021 only shows that the feature distributions differ. It does not demonstrate that the relationship between API calls and malware labels has changed.

Concept drift needs to be investigated through changes in predictive performance and the feature-label relationship, which should be addressed in Step 6.

## 3. Other implementation improvements

| Area                   | Suggested change                                                    | Priority |
| ---------------------- | ------------------------------------------------------------------- | -------- |
| `bh_adjust()`          | Validate that p-values are finite and within [0, 1].                | Medium   |
| `js_distance()`        | Validate that histogram rows have nonzero totals.                   | Medium   |
| `feature_drift()`      | Check that each year/class block has enough rows before running KS. | Medium   |
| `year_classifier()`    | Handle cases where a year has too few unique vectors.               | Medium   |
| `write_report()`       | Avoid hard-coded dataset-size claims.                               | High     |
| `count_bins()`         | Use a wider integer type.                                           | Low      |
| `class_cancellation()` | Assert expected classes and feature counts.                         | Medium   |

# 4. One particularly important issue: the year classifier's AUC interpretation

The report says:

> Neighbouring years are partly separable, and years three or more apart are highly separable.

This is a reasonable interpretation if the AUC values support it.

But the code currently samples at most 2,000 unique vectors per year, and the latest years may have very few unique vectors.

That means some AUC values may be based on substantially smaller samples than others.

The report should avoid treating every AUC as equally reliable. It already includes the number of unique rows per year, which is good. I'd also explicitly flag any AUC based on a very small sample.

For example:

> AUC estimates based on small numbers of unique feature vectors should be treated as exploratory rather than conclusive.

# 5. What I would keep unchanged

These are good decisions and should remain:

- Comparing every pair of years.
- Comparing all apps, benign apps, and malware apps separately.
- Using KS statistics as requested by the assignment.
- Applying Benjamini-Hochberg correction separately for each year-pair/class family.
- Reporting prevalence changes.
- Adding Jensen-Shannon distance for count distributions.
- Reporting effect sizes alongside statistical significance.
- Including a duplicate-vector sensitivity analysis.
- Using a year classifier to measure temporal distinguishability.
- Keeping Step 2 descriptive and preventing it from influencing the train/test split.
- Separating feature drift from concept drift.

# Overall reviewer verdict

Revise and rerun

The approach is strong and goes beyond simply running a KS test. The code measures several complementary aspects of feature drift and makes a useful distinction between feature drift and concept drift.

I would not ask the builder to redesign Step 2. I would ask for the following changes:

1. Fix the year-classifier explanation and duplicate-handling limitation.
2. Remove hard-coded sample-size claims from the report.
3. Make the bin index type more robust.
4. Add validation for the year-classifier sample sizes and class-cancellation inputs.
5. Make the limitations of small-sample AUC estimates explicit.

One thing I would not do: change the chronological train/test split based on the drift results. The split should stay fixed as planned.

If you send the remaining files (especially the tests, generated Step 2 report, and output tables), I can review whether the actual results support the report's conclusions and turn this into a formal `step2_review.md` file.

Nt Ayres: can you reformat this like the md review file **# Step 1 Code Review**

**\*\*Reviewer:\*\*** Reviewer agent\\
**\*\*Project:\*\*** IT348/448 Intro to ML --- Project 2\\
**\*\*Files reviewed:\*\*** \`Config.py\`, \`DataLoading.py\`, \`Data Summary.py\`, and
\`Main.py\`\\
**\*\*Review scope:\*\*** Step 1 code review only (correctness, assignment
requirements, and recommended changes before approval). The project was
not run against the dataset.

**## Overall assessment**

The implementation is a good start. It validates raw file shapes,
preserves \`apkname\` as metadata, derives labels from filenames, and
reports duplicates without removing rows. The design does not need to be
reworked. However, several targeted changes are recommended before Step
1 is considered complete. The most important issues are:

1\.  The file names do not match the module names used in the imports and
    in the run command.
2\.  The \`--all\` flag claims to run all steps but currently runs only
    Step 1.
3\.  Duplicate counts are ambiguous: row counts, rows beyond the first
    occurrence, and unique duplicated keys are not clearly separated.
4\.  A few defensive dataset validations are missing (unique feature
    headers, explicit class presence, integer value checks).
5\.  Output handling (directory creation, CSV index) should be explicit.

**------------------------------------------------------------------------**

**## Review comments**

**### R1. Rename files to match the imports and run command**

**\*\*Priority:\*\*** High\\
**\*\*Location:\*\*** \`Config.py\`, \`DataLoading.py\`, \`Data Summary.py\`,
\`Main.py\`.

The other files import \`from src import config\`,
\`from src.data_loading import build_dataset\`, and
\`from src.data_summary import counts_per_year, duplicate_summary, write_report\`.
The usage line is \`python -m src.main --step 1\`. The supplied file
names (\`Config.py\`, \`DataLoading.py\`, \`Data Summary.py\`, \`Main.py\`) do
not match these. Python imports are case-sensitive in many environments,
and a space in a module file name is not compatible with this import
style, so the imports can fail.

**\*\*Recommendation:\*\*** Rename the files to \`config.py\`, \`data_loading.py\`,
\`data_summary.py\`, and \`main.py\`, and use these module names
consistently everywhere.

**### R2. Make the \`--all\` flag truthful**

**\*\*Priority:\*\*** High\\
**\*\*Location:\*\*** \`Main.py\`, argument parser and step dispatch.

The parser describes \`--all\` as "run all steps", but the code
(\`if args.all or args.step == 1: run_step1(args.input_dir)\`) runs only
Step 1. The CLI currently promises more than it delivers. This is the
most important issue in \`main.py\`.

**\*\*Recommendation:\*\*** Either change the help text to say that \`--all\`
currently runs the available Step 1 pipeline, or implement the later
steps before claiming that \`--all\` runs all project steps.

**### R3. Make duplicate counts unambiguous**

**\*\*Priority:\*\*** High\\
**\*\*Location:\*\*** \`Data Summary.py\`, \`duplicate_summary()\` and
\`rows_shared()\`.

Several counts can be misread:

\-   \`rows - unique_apkname\` measures duplicate rows beyond the first
    occurrence, not the number of duplicated identifier groups. For
    example, an identifier that appears three times counts as two
    duplicate rows.
\-   \`rows_shared\` counts rows whose identifier or vector appears across
    multiple years or classes. These are row counts, not counts of
    unique identifiers or vectors, so readers may interpret them as
    unique apps.

**\*\*Recommendation:\*\*** Label each count clearly and add unique
duplicated-key counts alongside the row totals. The final report should
separately state:

\-   Duplicate identifier rows within a file.
\-   Duplicate feature-vector rows within a file.
\-   Identifiers appearing across years.
\-   Identifiers appearing across classes.
\-   Feature vectors appearing across years.
\-   Feature vectors appearing across classes.

Most of these are already calculated. Keeping the identifier and
feature-vector distinction (as in the docstring) is correct.

**### R4. Check that feature headers are unique**

**\*\*Priority:\*\*** Medium\\
**\*\*Location:\*\*** \`DataLoading.py\`, column consistency check
(\`elif cols != reference_columns\`).

The existing check of column names and order across files is a strong
validation and should be kept. Duplicate headers, however, can make
DataFrame selection and downstream feature handling ambiguous.

**\*\*Recommendation:\*\*** Explicitly check that the feature column names are
unique.

**### R5. Add explicit class-presence validation after loading**

**\*\*Priority:\*\*** Medium\\
**\*\*Location:\*\*** \`DataLoading.py\`, \`set(files) != set(config.LABEL_MAP.values())\`.

The current check correctly requires one benign and one malware file for
each year. The filename grouping should guarantee both labels, but a
direct check would make the loader's output contract clearer.

**\*\*Recommendation:\*\*** After loading both class files, verify that each
year contains both labels.

**### R6. Validate integer values and \`int32\` range before conversion**

**\*\*Priority:\*\*** Medium\\
**\*\*Location:\*\*** \`DataLoading.py\`, feature dtype check and
\`features.astype(np.int32)\`.

The integer dtype check is consistent with the assignment's count-valued
features, but it does not verify that every original value was an
integer before conversion. Separately, values above the \`int32\` range
would overflow. Neither is an observed problem with the current data.

**\*\*Recommendation:\*\***

\-   Keep the integer dtype check, and load feature columns with a strict
    numeric conversion or explicitly check that values are
    integer-valued.
\-   Verify that values fit in \`int32\` before converting, or keep the
    original integer dtype.

**### R7. Make CSV index handling explicit**

**\*\*Priority:\*\*** Medium\\
**\*\*Location:\*\*** \`Main.py\`, \`per_year.to_csv(...)\` and \`per_file.to_csv(...)\`.

Pandas writes the DataFrame index by default. The output is valid, but
the choice should be explicit so downstream readers know how to
interpret the CSV.

**\*\*Recommendation:\*\*** Use \`index=True\` intentionally if the year/file
index is part of the output, or reset the index and use \`index=False\`.

**### R8. Make output directory creation robust**

**\*\*Priority:\*\*** Medium\\
**\*\*Location:\*\*** \`Main.py\` (\`mkdir\` calls) and \`Data Summary.py\`
(\`write_report()\`).

\`mkdir(exist_ok=True)\` does not create missing parent directories, and
\`path.write_text(...)\` assumes the parent directory exists. The code
works when run through the entry point, since \`main.py\` creates the
directories first.

**\*\*Recommendation:\*\***

\-   Use \`config.RESULTS_DIR.mkdir(parents=True, exist_ok=True)\` and
    \`config.DOCS_DIR.mkdir(parents=True, exist_ok=True)\`.
\-   Either create the parent directory inside \`write_report()\` or
    document that the caller must create it.

**### R9. Compare exact vectors after hash grouping**

**\*\*Priority:\*\*** Low\\
**\*\*Location:\*\*** \`Data Summary.py\`,
\`pd.util.hash_pandas_object(df[feature_columns(df)], index=False)\`.

Hashing is an efficient way to identify matching feature vectors, but a
hash is not the original vector and collisions are theoretically
possible (unlikely in practice).

**\*\*Recommendation:\*\*** Use hashes to identify candidate groups, then
compare the original feature vectors within those groups before
declaring them identical, so the report identifies exact matches.

**### R10. Documentation and consistency notes**

**\*\*Priority:\*\*** Low\\
**\*\*Location:\*\*** \`Config.py\`, \`DataLoading.py\`.

\-   **\*\*Shape constants:\*\*** \`EXPECTED_ROWS = 1500\` and \`EXPECTED_COLS = 1849\`
    are useful, but they will reject files if the dataset changes. Keep
    them for this assignment and document that they are
    assignment-specific expectations, not general CSV requirements.
\-   **\*\*Feature count:\*\*** \`EXPECTED_FEATURES = EXPECTED_COLS - 1\` is correct
    for one identifier column plus 1,848 features. Add a comment that
    the identifier is metadata and must never enter the model feature
    matrix.
\-   **\*\*Directory scan:\*\*** \`input_dir.glob("\*.csv")\` with every file passed
    to \`parse_filename\` is good fail-fast behavior; keep it. Consider
    adding the discovered years and file names to the report so the
    processed files can be verified.
\-   **\*\*\`source_file\` column:\*\*** Keep it for duplicate reporting, and make
    sure it is excluded from model features everywhere features are
    constructed (it is already excluded in \`feature_columns()\`).

**------------------------------------------------------------------------**

**## Reviewer priority summary**

\| Priority | Change                                                                                           |
\|----------|--------------------------------------------------------------------------------------------------|
\| High     | Rename files to match the imports: \`config.py\`, \`data_loading.py\`, \`data_summary.py\`, \`main.py\`. |
\| High     | Fix the \`--all\` description or implement all steps.                                              |
\| High     | Make duplicate reporting distinguish row counts from unique duplicate groups.                    |
\| Medium   | Check feature headers are unique.                                                                |
\| Medium   | Add explicit class-presence validation after loading.                                            |
\| Medium   | Verify integer values fit the chosen dtype before conversion.                                    |
\| Medium   | Make CSV index handling explicit.                                                                |
\| Medium   | Make output directory creation robust.                                                           |
\| Low      | Add exact vector comparison after hash grouping.                                                 |
\| Low      | Add documentation and consistency notes.                                                         |

**## What should remain unchanged**

The following parts of the implementation are sound and should generally
be retained:

\-   Preserving \`apkname\` as metadata and excluding it from model
    features.
\-   Deriving labels from filenames.
\-   Fail-fast validation of raw file shapes and unexpected CSV files.
\-   Column name and order consistency checks across files.
\-   Reporting duplicates without silently dropping rows.
\-   Separating duplicate identifiers from duplicate feature vectors.

**## Reviewer verdict**

**\*\*Verdict: Request revisions before approval.\*\***

The overall Step 1 design is approved. The loader and summary code are
well organized and follow the intended approach of retaining identifiers
as metadata, validating the dataset, and reporting duplicates without
silently dropping rows. The main changes requested are:

1\.  Fix the module file name and import mismatch.
2\.  Make the CLI's \`--all\` behavior truthful.
3\.  Make duplicate counts unambiguous.
4\.  Add a few defensive dataset validations.

These are code-review comments based on the four files provided. The
project was not run against the dataset, so this review makes no claim
that the current implementation passes runtime or assignment tests.

**------------------------------------------------------------------------**

**## Student decision log**

\| Review ID                              | Decision (Accepted / Rejected) | Student's reason |
\|----------------------------------------|--------------------------------|------------------|
\| R1 --- File names and imports          |            Reject              |     Unneeded     |
\| R2 --- \`--all\` flag behavior           |            Accept              |      Valid       |
\| R3 --- Duplicate count clarity         |            Accept              |      Valid       |
\| R4 --- Unique feature headers          |            Accept              |      Valid       |
\| R5 --- Class-presence validation       |            Accept              |      Valid       |
\| R6 --- Integer value and dtype checks  |            Accept              |      Valid       |
\| R7 --- CSV index handling              |            Accept              |      Valid       |
\| R8 --- Output directory creation       |            Accept              |      Valid       |
\| R9 --- Exact vector comparison         |            Accept              |      Valid       |
\| R10 --- Documentation and consistency  |            Accept              |      Valid       |

chatgpt: Absolutely. I’ve formatted the Step 2 review to match your Step 1 review's structure, including:
- Overall assessment
- Numbered review comments with priority, location, and recommendation
- Reviewer priority summary
- What should remain unchanged
- Reviewer verdict
- Student decision log
The Markdown content is prepared, but I couldn't create the downloadable .md file. Data analysis should become available again in approximately 22 hours, though that is only an estimate.

I can still provide the complete Markdown text here for you to copy into a file named step2_code_review.md.

Nt Ayres: Give me the info that was on the file.
chatgpt: Copy everything below into a file named `step2_code_review.md`.

````
# Step 2 Code Review

## Overall Assessment

Step 2 is organized well and covers the main descriptive drift analyses required by the project. The implementation includes feature-level drift metrics, year-pair comparisons, duplicate-vector sensitivity analysis, a year classifier, class cancellation analysis, and report generation.

The overall approach is appropriate for a descriptive drift analysis. The main concerns are with how the year classifier handles identical feature vectors, how the report describes sampled versus available data, and several hardcoded assumptions about the dataset's year structure.

The report also makes some claims that are not directly established by the calculations currently shown. These should be made data-driven or qualified before the report is considered final.

**Recommendation: Revise before proceeding to Step 3.** The overall structure can remain, but the issues below should be addressed to make the analysis more reliable and the generated report accurate.

---

## Review Comments

### R1 — Prevent identical feature vectors from appearing in both classifier splits

**Priority:** High  
**Location:** `year_classifier(...)`

The classifier removes duplicate feature vectors within each `(year, class)` group:

```python
np.unique(X[idx], axis=0, return_index=True)
````

However, identical feature vectors can still exist in both years being compared. After the two year samples are concatenated, the code performs a random train/test split.

This means an identical feature vector can appear in both the training and testing sets, potentially inflating the classifier's ROC AUC.

This is especially relevant because the project already includes duplicate sensitivity analysis.

Recommendation:

Ensure identical feature vectors are kept together during the classifier split.

Possible approaches:

- Group identical feature vectors across both compared years and use a group-aware split.
- Remove cross-year duplicate vectors from the classifier evaluation and report how many were removed.
- Explicitly report this limitation if the classifier is intended only as a descriptive demonstration.

Do not describe the resulting AUC as evidence of generalization to unseen applications unless the split actually supports that interpretation.

### R2 — Distinguish identical feature vectors from duplicate applications

Priority: Medium\
Location: `year_classifier(...)`, duplicate sensitivity analysis, report wording

The code collapses identical feature vectors, but identical feature vectors do not necessarily represent the same application.

Different applications may have the same feature representation.

The current analysis can establish that feature vectors are identical, but it cannot establish that the applications themselves are duplicates unless application identity is checked separately.

Recommendation:

Use terminology such as:

- Unique feature vectors
- Identical feature vectors
- Feature-vector deduplication

Avoid describing the operation as removing duplicate applications unless the application identifiers establish that relationship.

Also distinguish:

1. Duplicate application identifiers.
2. Identical feature vectors.
3. Identical vectors appearing across different years.

These are different situations and should not be treated as interchangeable.

### R3 — Make dataset-specific report claims data-driven

Priority: High\
Location: `write_report(...)`

The report contains several dataset-specific assumptions, including:

- The expected number of years.
- The number of rows in particular year ranges.
- The class balance.
- Specific year comparisons.
- Named features associated with particular observations.
- Statements about which year pair is most different.

Some of these may accurately describe the current dataset, but the report should establish them from the actual analysis results rather than relying on fixed assumptions.

For example, a statement that the first and last years are the most different should not be made solely because those years are farthest apart chronologically.

The pair with the greatest mean KS statistic may be a different pair.

Recommendation:

Calculate dataset-specific statements from the actual results.

At minimum:

- Calculate the number of available rows per year.
- Calculate class counts and proportions per year.
- Identify the year pair with the largest mean KS statistic.
- Identify the largest consecutive-year change dynamically.
- Verify that named features actually correspond to the reported values.

If a statement is intended specifically for the current dataset, make that clear and validate the assumption before generating the report.

### R4 — Correct the interpretation of classifier sample sizes

Priority: Medium\
Location: `year_classifier(...)`, `write_report(...)`

The year classifier uses:

```
per_year = 2000
```

and selects an equal number of vectors from each compared year, subject to availability.

The report's `rows_per_year` values represent the number selected for the classifier, not necessarily the total number of unique feature vectors available in that year.

Consequently, a report table describing these values as the number of unique rows available per year could be misleading.

Recommendation:

Separate the following quantities:

- Original rows available per year.
- Unique feature vectors available per year.
- Number of vectors selected for the classifier.
- Number of vectors remaining after any cross-year duplicate handling.

Use explicit column names such as:

```
available_rows
available_unique_vectors
classifier_sample_size
```

Do not use the sampled classifier size as a substitute for the dataset's actual yearly size.

### R5 — Add validation for missing classes and insufficient samples

Priority: Medium\
Location: `year_classifier(...)`

The classifier assumes that each compared year and class has enough unique feature vectors to sample and perform a train/test split.

If a year/class combination contains zero unique vectors, or too few vectors for the split, the current workflow may fail during sampling or classifier training.

This is particularly important when the analysis is performed separately for benign and malware samples.

Recommendation:

Before sampling, validate that each required year/class combination has enough unique feature vectors.

If a group is insufficient:

- Raise a clear error, or
- Skip the classifier comparison and record why it was skipped.

Do not silently produce an incomplete classifier table.

Also verify that the resulting train/test split contains both classes.

### R6 — Add uncertainty or qualify the year-classifier results

Priority: Medium\
Location: `year_classifier(...)`, `write_report(...)`

The year classifier uses a random train/test split and a fixed random seed.

This makes the analysis reproducible, but a single split can produce a noisy ROC AUC estimate.

The issue becomes more important when the available sample is small or when duplicate-vector handling removes many observations.

The classifier is useful for measuring whether the feature representation distinguishes two years, but its AUC should not be interpreted as a definitive temporal-generalization result.

Recommendation:

Either:

- Repeat the evaluation using multiple stratified splits and report the average and variability, or
- Keep the current single split and explicitly describe the AUC as a descriptive result from one fixed random split.

If repeated splits are added, ensure identical feature vectors remain grouped together across splits.

The classifier should remain descriptive and should not be used to tune the test-year analysis.

### R7 — Make the report robust to different numbers of years

Priority: Medium\
Location: `write_report(...)`

The report contains positional year references such as:

```
years[3]years[4]years[-4]years[-3]
```

These assume a particular number and ordering of years.

If the dataset contains fewer years, the report may fail with an index error.

If the dataset contains more years or the year range changes, the selected comparisons may no longer represent the intended analysis.

Recommendation:

Validate the year structure before generating the report.

For example:

- Verify that enough years exist for the required train/test boundary comparison.
- Verify that the expected years are present.
- Use explicit year values when the comparison is part of the project design.
- Use dynamically selected year pairs when the comparison is intended to describe the dataset generally.

If the assignment specifically requires a five-year training period followed by a test year, encode that requirement explicitly and validate it rather than relying on positional indexing.

### R8 — Verify that the report's selected year pairs match its claims

Priority: Medium\
Location: `write_report(...)`

The report includes comparisons involving the first and last years, consecutive years, and particular year positions.

Some statements are based on specific year positions rather than selecting the relevant pair from the computed results.

For example, a selected pair may be described as a particularly large change even though the code has not established that it is the largest change.

Recommendation:

For each claim about a year pair:

1. Identify the actual pair used.
2. Retrieve its corresponding computed statistics.
3. Verify that the statistic supports the statement.
4. Use dynamic selection if the claim refers to the largest or smallest result.

Avoid using positional year comparisons as evidence for a general claim about the dataset.

### R9 — Validate completeness before pivoting class cancellation results

Priority: Medium\
Location: `class_cancellation(...)`

The class cancellation analysis pivots the results using class-specific KS statistics and prevalence changes.

The later calculations assume that each feature has the required benign, malware, and all-class entries.

If one of these entries is missing, the resulting pivot may not contain the expected columns, or the analysis may fail.

Recommendation:

Validate that the input contains the expected class entries for each feature/year comparison.

If a class is missing:

- Raise a clear error, or
- Exclude the incomplete comparison and report the reason.

The current opposite-direction calculation is appropriate for identifying changes with opposing signs, but the report should make clear that a zero change is not counted as an opposite-direction change.

### R10 — Validate year-pair summary tables before accessing rows

Priority: Medium\
Location: `symmetric_matrix(...)`, `write_report(...)`

The report retrieves summary rows using year-pair and class filters and then accesses the first matching row.

This assumes the requested comparison exists and appears exactly once.

If a year pair or class is missing, the report may fail when accessing `.iloc[0]`.

If duplicate summary rows exist, the report may silently select the wrong row.

Recommendation:

Add validation for:

- Required year pairs.
- Required classes.
- Duplicate year-pair/class entries.
- Missing feature summaries.

The symmetric matrix helper should also validate that the year values match the expected years and that the input does not contain duplicate entries for the same pair.

Fail with a descriptive message rather than allowing an obscure indexing error.

### R11 — Keep preprocessing inside the classifier pipeline

Priority: Low\
Location: `year_classifier(...)`

The classifier uses `log1p`, `StandardScaler`, and logistic regression.

The scaler is included in the pipeline, which is good because it allows scaling to be fitted on the training split rather than the full dataset.

Recommendation:

Keep the preprocessing within the pipeline.

Do not fit the scaler on the combined training and testing data before splitting.

Also verify that the input features are nonnegative before applying `log1p`, and that the resulting feature matrix contains no invalid values.

This is a good part of the current implementation and should remain unchanged unless there is a specific reason to modify it.

### R12 — Add focused tests for the analysis helpers

Priority: Medium\
Location: Step 2 analysis functions

The supplied code includes several helper functions whose behavior is important to the final report:

- `symmetric_matrix(...)`
- `year_classifier(...)`
- `class_cancellation(...)`
- `top_features(...)`
- `write_report(...)`

The review snippet does not show tests for these functions.

Recommendation:

Add focused tests using small synthetic datasets.

At minimum, test:

- Symmetric year-pair matrix construction.
- Missing year-pair handling.
- Duplicate summary row handling.
- Identical feature-vector handling in the classifier.
- Insufficient samples for classifier training.
- Missing class entries in cancellation analysis.
- Feature ranking and top-feature selection.
- Report generation with the expected number of years.
- Report generation when required years or classes are missing.

These tests would help prevent report-generation failures and catch incorrect interpretations of the computed statistics.

### R13 — Remove unused helper code

Priority: Low\
Location: `_pct(...)`

The `_pct` helper appears unused in the supplied report-generation code.

Recommendation:

Remove it if it is not needed elsewhere, or use it consistently if percentage formatting is intended.

Keeping unused helpers makes the report code harder to maintain.

## Reviewer Priority Summary

| Priority | Comments        | Action                                                            |
| -------- | --------------- | ----------------------------------------------------------------- |
| High     | R1, R3          | Fix classifier duplicate handling and validate report claims      |
| Medium   | R2, R4–R10, R12 | Improve terminology, sample-size reporting, validation, and tests |
| Low      | R11, R13        | Preserve the pipeline approach and clean up unused code           |

## What Should Remain Unchanged

The following parts of Step 2 are useful and should remain in the implementation:

1. Feature-level drift analysis\
   The analysis of KS statistics, prevalence changes, JS divergence, and related feature-level summaries provides useful information about how the dataset changes over time.
2. Year-pair comparisons\
   Comparing multiple year pairs is useful for distinguishing gradual changes from larger changes between distant years.
3. Duplicate sensitivity analysis\
   Including a comparison with and without duplicate handling is valuable. The analysis should clearly distinguish application duplicates from identical feature vectors.
4. Year classifier as a descriptive analysis\
   The classifier can provide a useful measure of how distinguishable two years are using the feature representation. It should remain descriptive rather than becoming a tuning mechanism.
5. Class cancellation analysis\
   Comparing benign and malware prevalence changes can help identify features whose overall drift may hide class-specific changes.
6. Automated report generation\
   Generating the report from the analysis results is a good approach. The report should remain automated, with the dataset-specific claims calculated from the actual results.
7. Pipeline-based preprocessing\
   Keeping scaling inside the classifier pipeline is appropriate and should be preserved.

## Reviewer Verdict

Revise before proceeding to Step 3.

The overall Step 2 structure is appropriate, and the analysis covers useful forms of temporal drift.

The most important correction is to ensure that the year classifier does not allow identical feature vectors to appear in both training and testing data. The report should also distinguish the number of available unique vectors from the number selected for classifier evaluation.

The remaining changes are primarily validation, report accuracy, and terminology improvements.

No major redesign is needed.

## Student Decision Log

| Review ID | Decision | Reason / Notes |
| --------- | -------- | -------------- |
| R1        |          |                |
| R2        |          |                |
| R3        |          |                |
| R4        |          |                |
| R5        |          |                |
| R6        |          |                |
| R7        |          |                |
| R8        |          |                |
| R9        |          |                |
| R10       |          |                |
| R11       |          |                |
| R12       |          |                |
| R13       |          |                |

```

**Note:** This review is based on the Step 2 code excerpt you provided earlier, not a fresh inspection of the full repository.
```