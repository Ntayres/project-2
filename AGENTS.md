# AGENTS.md: Project 2 context for all agents

This is the shared project context for the builder, reviewer, and tester. Each agent
must read this file before coding, reviewing, or writing tests. Keep it synchronized
with the repository layout and the commands that actually work.

## Set up before coding

- Work from the repository root. This project is expected to be a Git repository; if
  starting from an uninitialized project directory, run `git init` before making
  changes.
- Confirm that `AGENTS.md` is present and use it as the shared context file for all
  three agents. `CLAUDE.md` may point to or summarize this file, but agent-specific
  instructions must not contradict it.
- Create or use a Python 3 virtual environment, then install the dependencies from
  `requirements.txt`.
- Do not commit the dataset, generated outputs, caches, virtual environments, or
  secrets. Check `.gitignore` before adding files.
- Before coding, inspect `PLAN.md` and the current step's source, tests, and review
  decision log. Preserve unrelated work already present in the working tree.

## Project goal

Classify Android apps as malware or goodware from their API-call features, and find a sustainable approach that stays accurate as the data drifts over 8 years. The full assignment is in ASSIGNMENT.md. The plan is in PLAN.md.

## Dataset

- Location: input_data/ (not committed to git).
- File names follow the pattern `sampled_<YEAR>_<CLASS>_api.csv`, for example `sampled_2019_benign_api.csv` and `sampled_2019_malware_api.csv`. `<YEAR>` is a four-digit year; `<CLASS>` is `benign` or `malware`. Each year has one benign file and one malware file. Parse the year and class from the file name with a strict regex and raise an error on any name that does not match. Do not hard-code the list of years; take them from the files.
- Each file is a comma-separated CSV with a header row and 1,849 columns. **Row counts differ by year**: 9,000 rows per file for 2014-2018 and 1,500 rows per file for 2019-2021 (the assignment text says 1,500). The first column is `apkname` (app name plus an MD5-style hash). It is an identifier, not a feature, and must be dropped. The other 1,848 columns are API-call features; their header is the API method signature (long strings such as `Landroid/accounts/AccountManager;.addAccount:(...)...`). The header is identical in every file.
- Label: not in the file contents; it comes from the file name. `malware` = 1, `benign` (goodware) = 0.
- Feature values are non-negative integer counts of API calls (not binary). Values go up to at least several hundred, so the scale differs by feature. There are no missing values and no non-numeric values in the files checked.
- Duplicates: the files contain repeated apps. In the six files checked (2019-2021), `apkname` has about 1,309-1,437 unique values per 1,500 rows, so about 63-191 rows per file repeat an earlier row. Step 1 must count duplicates within a file and across years, and report them. Whether to drop them is decided with the student; do not drop silently.
- Years: 2014-2021 (8 years), 16 files: one benign and one malware file per year.

## Folder layout

- src/: project code (builder only).
- tests/: pytest tests test_stepN_*.py and test logs stepN_test_log.md (tester only; the student adds failure decisions).
- input_data/: source CSV dataset (not committed).
- docs/: one builder report per step.
- reviews/: one review report per step (reviewer writes comments; student writes the decision log).
- transcripts/: saved agent sessions (saved by the harness; agents never edit these).
- outputs/: intermediate data and models (not committed). results/: small result tables and figures (committed).
- PLAN.md: the plan for Steps 1-7. REPORT.md and REFLECTION.md: written by the student.

## Roles

- Builder: writes all code in src/ and the reports in docs/. Does not edit tests/ or reviews/.
- Reviewer: read-only on code. Writes only reviews/stepN_review.md. Checks for bugs, leakage between years, wrong metrics, unclear code, and mismatch with PLAN.md.
- Tester: writes only tests/. Derives tests from the assignment, this file, and PLAN.md. Does not read the builder's reports in docs/.
- Student: lead engineer. Writes decision logs, REPORT.md, and REFLECTION.md, and makes final decisions.

## Coding conventions

- Python 3, pandas, numpy, scikit-learn, scipy (matplotlib for figures). No other libraries without asking.
- Functions have docstrings. Each module starts with a short comment on its role. Comment each major step.
- `random_state=42` everywhere. No hard-coded absolute paths; paths are relative to the repo root.
- Fail loudly: assert shapes (1,849 columns before dropping the first column, 1,848 after; row counts vary by file), identical headers across files, and file-name matches.
- Train/test split is by year, never a random shuffle across years. The default is the first half of the years for training and the later half for testing, as set in PLAN.md.
- Anything fit to data (scalers, feature selection, thresholds, hyperparameters, models) is fit on training years only. Drift statistics (Step 2) are descriptive and must not be used to choose features or tune models using test years.
- Model selection inside the training years uses time-ordered validation (train on earlier years, validate on a later training year), not random k-fold.
- Preprocessing is built per model, never loaded pre-fitted. Use `preprocessing.preprocessor_for(model_name)` to get a fresh pipeline, put the classifier in the same `Pipeline`, and pass `preprocessing.fold_indices(...)` as `cv`, so every fold refits the dropped-column set and the scaler on that fold's training years only. The fitted pipelines in `outputs/step3_preprocessor_*.joblib` are diagnostic artifacts for inspection; loading one and transforming the data before model selection would leak each fold's validation year into its own preprocessing. The final pipeline is fitted on the whole training half once, after the model is chosen.
- Malware is the positive class (label 1) for precision, recall, and F1.

## How to run

- From the repository root, run one pipeline step with `python -m src.main --step N`
  and run the complete pipeline with `python -m src.main --all`.
- Run the tests with `pytest -q`. The harness also supports
  `python harness.py test N` for one step and `python harness.py test all` for the
  complete test suite.
- Keep test logs in `tests/stepN_test_log.md` and record the exact command, output,
  failures, and whether each failure came from the source or the test fixture.

## Commit rules

Every commit message starts with a role tag: [builder], [reviewer], [tester], or [student]. A git hook rejects other messages. Make small commits, one per task. Never rewrite or squash history.
