# Step 1 Code Review

**Reviewer:** Reviewer agent\
**Project:** IT348/448 Intro to ML --- Project 2\
**Files reviewed:** `Config.py`, `DataLoading.py`, `Data Summary.py`, and
`Main.py`\
**Review scope:** Step 1 code review only (correctness, assignment
requirements, and recommended changes before approval). The project was
not run against the dataset.

## Overall assessment

The implementation is a good start. It validates raw file shapes,
preserves `apkname` as metadata, derives labels from filenames, and
reports duplicates without removing rows. The design does not need to be
reworked. However, several targeted changes are recommended before Step
1 is considered complete. The most important issues are:

1.  The file names do not match the module names used in the imports and
    in the run command.
2.  The `--all` flag claims to run all steps but currently runs only
    Step 1.
3.  Duplicate counts are ambiguous: row counts, rows beyond the first
    occurrence, and unique duplicated keys are not clearly separated.
4.  A few defensive dataset validations are missing (unique feature
    headers, explicit class presence, integer value checks).
5.  Output handling (directory creation, CSV index) should be explicit.

------------------------------------------------------------------------

## Review comments

### R1. Rename files to match the imports and run command

**Priority:** High\
**Location:** `Config.py`, `DataLoading.py`, `Data Summary.py`,
`Main.py`.

The other files import `from src import config`,
`from src.data_loading import build_dataset`, and
`from src.data_summary import counts_per_year, duplicate_summary, write_report`.
The usage line is `python -m src.main --step 1`. The supplied file
names (`Config.py`, `DataLoading.py`, `Data Summary.py`, `Main.py`) do
not match these. Python imports are case-sensitive in many environments,
and a space in a module file name is not compatible with this import
style, so the imports can fail.

**Recommendation:** Rename the files to `config.py`, `data_loading.py`,
`data_summary.py`, and `main.py`, and use these module names
consistently everywhere.

### R2. Make the `--all` flag truthful

**Priority:** High\
**Location:** `Main.py`, argument parser and step dispatch.

The parser describes `--all` as "run all steps", but the code
(`if args.all or args.step == 1: run_step1(args.input_dir)`) runs only
Step 1. The CLI currently promises more than it delivers. This is the
most important issue in `main.py`.

**Recommendation:** Either change the help text to say that `--all`
currently runs the available Step 1 pipeline, or implement the later
steps before claiming that `--all` runs all project steps.

### R3. Make duplicate counts unambiguous

**Priority:** High\
**Location:** `Data Summary.py`, `duplicate_summary()` and
`rows_shared()`.

Several counts can be misread:

-   `rows - unique_apkname` measures duplicate rows beyond the first
    occurrence, not the number of duplicated identifier groups. For
    example, an identifier that appears three times counts as two
    duplicate rows.
-   `rows_shared` counts rows whose identifier or vector appears across
    multiple years or classes. These are row counts, not counts of
    unique identifiers or vectors, so readers may interpret them as
    unique apps.

**Recommendation:** Label each count clearly and add unique
duplicated-key counts alongside the row totals. The final report should
separately state:

-   Duplicate identifier rows within a file.
-   Duplicate feature-vector rows within a file.
-   Identifiers appearing across years.
-   Identifiers appearing across classes.
-   Feature vectors appearing across years.
-   Feature vectors appearing across classes.

Most of these are already calculated. Keeping the identifier and
feature-vector distinction (as in the docstring) is correct.

### R4. Check that feature headers are unique

**Priority:** Medium\
**Location:** `DataLoading.py`, column consistency check
(`elif cols != reference_columns`).

The existing check of column names and order across files is a strong
validation and should be kept. Duplicate headers, however, can make
DataFrame selection and downstream feature handling ambiguous.

**Recommendation:** Explicitly check that the feature column names are
unique.

### R5. Add explicit class-presence validation after loading

**Priority:** Medium\
**Location:** `DataLoading.py`, `set(files) != set(config.LABEL_MAP.values())`.

The current check correctly requires one benign and one malware file for
each year. The filename grouping should guarantee both labels, but a
direct check would make the loader's output contract clearer.

**Recommendation:** After loading both class files, verify that each
year contains both labels.

### R6. Validate integer values and `int32` range before conversion

**Priority:** Medium\
**Location:** `DataLoading.py`, feature dtype check and
`features.astype(np.int32)`.

The integer dtype check is consistent with the assignment's count-valued
features, but it does not verify that every original value was an
integer before conversion. Separately, values above the `int32` range
would overflow. Neither is an observed problem with the current data.

**Recommendation:**

-   Keep the integer dtype check, and load feature columns with a strict
    numeric conversion or explicitly check that values are
    integer-valued.
-   Verify that values fit in `int32` before converting, or keep the
    original integer dtype.

### R7. Make CSV index handling explicit

**Priority:** Medium\
**Location:** `Main.py`, `per_year.to_csv(...)` and `per_file.to_csv(...)`.

Pandas writes the DataFrame index by default. The output is valid, but
the choice should be explicit so downstream readers know how to
interpret the CSV.

**Recommendation:** Use `index=True` intentionally if the year/file
index is part of the output, or reset the index and use `index=False`.

### R8. Make output directory creation robust

**Priority:** Medium\
**Location:** `Main.py` (`mkdir` calls) and `Data Summary.py`
(`write_report()`).

`mkdir(exist_ok=True)` does not create missing parent directories, and
`path.write_text(...)` assumes the parent directory exists. The code
works when run through the entry point, since `main.py` creates the
directories first.

**Recommendation:**

-   Use `config.RESULTS_DIR.mkdir(parents=True, exist_ok=True)` and
    `config.DOCS_DIR.mkdir(parents=True, exist_ok=True)`.
-   Either create the parent directory inside `write_report()` or
    document that the caller must create it.

### R9. Compare exact vectors after hash grouping

**Priority:** Low\
**Location:** `Data Summary.py`,
`pd.util.hash_pandas_object(df[feature_columns(df)], index=False)`.

Hashing is an efficient way to identify matching feature vectors, but a
hash is not the original vector and collisions are theoretically
possible (unlikely in practice).

**Recommendation:** Use hashes to identify candidate groups, then
compare the original feature vectors within those groups before
declaring them identical, so the report identifies exact matches.

### R10. Documentation and consistency notes

**Priority:** Low\
**Location:** `Config.py`, `DataLoading.py`.

-   **Shape constants:** `EXPECTED_ROWS = 1500` and `EXPECTED_COLS = 1849`
    are useful, but they will reject files if the dataset changes. Keep
    them for this assignment and document that they are
    assignment-specific expectations, not general CSV requirements.
-   **Feature count:** `EXPECTED_FEATURES = EXPECTED_COLS - 1` is correct
    for one identifier column plus 1,848 features. Add a comment that
    the identifier is metadata and must never enter the model feature
    matrix.
-   **Directory scan:** `input_dir.glob("*.csv")` with every file passed
    to `parse_filename` is good fail-fast behavior; keep it. Consider
    adding the discovered years and file names to the report so the
    processed files can be verified.
-   **`source_file` column:** Keep it for duplicate reporting, and make
    sure it is excluded from model features everywhere features are
    constructed (it is already excluded in `feature_columns()`).

------------------------------------------------------------------------

## Reviewer priority summary

| Priority | Change                                                                                           |
|----------|--------------------------------------------------------------------------------------------------|
| High     | Rename files to match the imports: `config.py`, `data_loading.py`, `data_summary.py`, `main.py`. |
| High     | Fix the `--all` description or implement all steps.                                              |
| High     | Make duplicate reporting distinguish row counts from unique duplicate groups.                    |
| Medium   | Check feature headers are unique.                                                                |
| Medium   | Add explicit class-presence validation after loading.                                            |
| Medium   | Verify integer values fit the chosen dtype before conversion.                                    |
| Medium   | Make CSV index handling explicit.                                                                |
| Medium   | Make output directory creation robust.                                                           |
| Low      | Add exact vector comparison after hash grouping.                                                 |
| Low      | Add documentation and consistency notes.                                                         |

## What should remain unchanged

The following parts of the implementation are sound and should generally
be retained:

-   Preserving `apkname` as metadata and excluding it from model
    features.
-   Deriving labels from filenames.
-   Fail-fast validation of raw file shapes and unexpected CSV files.
-   Column name and order consistency checks across files.
-   Reporting duplicates without silently dropping rows.
-   Separating duplicate identifiers from duplicate feature vectors.

## Reviewer verdict

**Verdict: Request revisions before approval.**

The overall Step 1 design is approved. The loader and summary code are
well organized and follow the intended approach of retaining identifiers
as metadata, validating the dataset, and reporting duplicates without
silently dropping rows. The main changes requested are:

1.  Fix the module file name and import mismatch.
2.  Make the CLI's `--all` behavior truthful.
3.  Make duplicate counts unambiguous.
4.  Add a few defensive dataset validations.

These are code-review comments based on the four files provided. The
project was not run against the dataset, so this review makes no claim
that the current implementation passes runtime or assignment tests.

------------------------------------------------------------------------

## Student decision log

| Review ID                              | Decision (Accepted / Rejected) | Student's reason |
|----------------------------------------|--------------------------------|------------------|
| R1 --- File names and imports          |            Reject              |     Unneeded     |
| R2 --- `--all` flag behavior           |            Accept              |      Valid       |
| R3 --- Duplicate count clarity         |            Accept              |      Valid       |
| R4 --- Unique feature headers          |            Accept              |      Valid       |
| R5 --- Class-presence validation       |            Accept              |      Valid       |
| R6 --- Integer value and dtype checks  |            Accept              |      Valid       |
| R7 --- CSV index handling              |            Accept              |      Valid       |
| R8 --- Output directory creation       |            Accept              |      Valid       |
| R9 --- Exact vector comparison         |            Accept              |      Valid       |
| R10 --- Documentation and consistency  |            Accept              |      Valid       |
