# Step 5 Code Review

Files reviewed

- `evaluations.py` — classification metrics, loading fitted models, test-set evaluation, and report generation.
- `main.py` — Step 5 integration and command-line execution.&#x20;

  Pasted text(3).txt



Review scope: The review is based on the supplied `evaluations.py` code and updated `main.py` excerpt. I have not run the project or inspected the full repository, including `config.py`, `preprocessing.py`, `models.py`, `AGENTS.md`, or `PLAN.md`.

## Overall Assessment

Step 5 is straightforward and follows the intended evaluation workflow:

- Rebuilds the dataset and uses the same year-based split.
- Loads the fitted Step 4 model pipelines.
- Evaluates each model on the held-out test matrix.
- Computes accuracy, precision, recall, and F1 with malware as the positive class.
- Sorts the models by test accuracy.
- Writes a machine-readable CSV and a Markdown report.

The most important remaining issues are input validation, consistency between the model artifact and test feature matrix, report assumptions, and output handling.

Verdict: Revise before treating Step 5 as complete. The evaluation approach is sound, and the fixes are targeted rather than requiring a redesign.

## Review Comments

### R1 — High: Step 5 does not validate that the loaded model matches the expected Step 4 artifact

Location: `evaluations.py`, `evaluate_models`, lines 24–31

The code checks that the expected `.joblib` file exists and then loads it:

```
estimator = joblib.load(path)predictions = estimator.predict(X_test)
```

This assumes the file is a valid fitted pipeline for the current dataset and feature contract.

Potential problems:

- A stale model from a previous run may have a different feature set.
- A file may contain an unfitted estimator or an object that does not implement the expected prediction interface.
- A model may be trained on a different year split or different label mapping.
- A malformed artifact may fail with an unclear exception during prediction.

The project has already established a feature contract in Step 3, but Step 5 does not explicitly validate the loaded artifacts against it.

Recommendation:\
Before prediction, validate that each loaded model:

- Has the expected prediction interface.
- Is fitted.
- Accepts the current test feature columns in the same order.
- Produces one prediction per test row.
- Produces only labels allowed by the configured label map.

If the model is a scikit-learn Pipeline, check its preprocessing and final estimator rather than assuming every `.joblib` file is a valid Step 4 model.

### R2 — High: `classification_metrics` does not reject empty or non-binary evaluation inputs explicitly

Location: `evaluations.py`, lines 14–36

The function validates matching lengths and that labels belong to `config.LABEL_MAP.values()`.

However, it does not explicitly reject:

- Empty `y_true` and `y_pred`.
- A test set containing only one class.
- Labels whose values are technically allowed but whose types differ from the configured mapping.
- Non-1D arrays.

An empty test set will fail later in scikit-learn, while a single-class test set can produce metrics that look valid but do not evaluate the detector's behavior on both classes.

The assignment may permit a test set with one class, but the report should not imply that both malware and benign performance were evaluated if one class is absent.

Recommendation:\
Validate:

- Both arrays are one-dimensional.
- They are nonempty.
- Their lengths match.
- Their labels match the configured classes.
- The test labels contain both classes, if that is a project requirement.

If single-class evaluation is allowed, record the class coverage in the report and explain the limitation.

### R3 — Medium: Step 5 validates feature columns but does not validate test-row alignment with labels

Location: `main.py`, `run_step5`, and `evaluations.py`, `run`

The updated main function delegates Step 5 to:

```
return evaluation.run(input_dir)
```

The evaluation workflow rebuilds the dataset, creates `X_test` and `y_test`, and checks the feature contract.

That is a good start. But the excerpt does not call the Step 3 label/year validation helper or otherwise confirm that:

- The test labels correspond to the same rows as `X_test`.
- The test rows contain only the expected test years.
- The test split has the expected row count.
- The feature matrix and label series preserve the same row order.

Since metrics depend on exact row alignment, the feature contract alone is not enough.

Recommendation:\
Reuse the existing Step 3 validation helpers where appropriate:

- Validate the split and year lists.
- Validate `X_test` against the expected feature contract.
- Validate the test labels and years against the test frame.
- Check that the prediction count equals the test-row count.

This should be done before scoring.

### R4 — Medium: `write_report` assumes `metrics.iloc[0]` is the winner

Location: `evaluations.py`, `write_report`, lines 48–75

The report chooses:

```
winner = metrics.iloc[0]
```

This is correct when `metrics` comes directly from `evaluate_models`, because that function sorts the results by accuracy.

However, `write_report` is a separate function and does not verify that the table is sorted. If it receives an unsorted table, the report can name the wrong classifier as having the highest accuracy.

The function also checks classifier membership but does not verify that each classifier occurs exactly once.

Recommendation:\
Make the report determine the winner explicitly:

```
winner = metrics.sort_values(    "accuracy", ascending=False, kind="stable").iloc[0]
```

Also validate:

- No duplicate classifier names.
- Exactly one row per expected classifier.
- Finite metric values.
- A nonempty table.

That way, the report is correct even if it is called independently of `evaluate_models`.

### R5 — Medium: `run` accepts custom output directories but does not create or consistently validate them

Location: `evaluations.py`, `run`

The function accepts:

```
results_dir=Nonedocs_dir=Nonemodels_dir=None
```

But it only creates the configured default directories:

```
for directory in (config.RESULTS_DIR, config.DOCS_DIR):    directory.mkdir(parents=True, exist_ok=True)
```

When a caller supplies a custom `results_dir`, the code writes the CSV there without ensuring that the directory exists. The report function creates its parent directory, but the CSV write occurs before that.

This makes the custom directory parameters inconsistent.

Recommendation:\
Resolve the directories at the beginning:

```
results_path = Path(results_dir or config.RESULTS_DIR)docs_path = Path(docs_dir or config.DOCS_DIR)results_path.mkdir(parents=True, exist_ok=True)docs_path.mkdir(parents=True, exist_ok=True)
```

Then use those resolved paths for all outputs.

Also validate that `models_dir` points to the expected model artifact directory before starting evaluation.

### R6 — Medium: CSV output is written before report validation

Location: `evaluations.py`, `run`

The code writes:

```
metrics.to_csv(...)
```

before calling:

```
write_report(...)
```

If report validation or report generation fails, the CSV may be updated while the Markdown report remains stale.

That leaves the Step 5 outputs inconsistent.

Recommendation:\
Validate all report inputs before writing either artifact. For stronger consistency, write the CSV and report to temporary files, then replace the final outputs only after both are successfully generated.

At minimum, ensure report validation happens before the CSV is written.

### R7 — Medium: The report's claim that the test set is never used for fitting is not independently verified

Location: `evaluations.py`, `write_report`, lines 67–75

The report says:

> “No test-year rows were used to fit, tune, or select these models.”

The current Step 5 code only loads models and predicts, so this statement is consistent with the code shown.

However, the function itself does not establish that the loaded models were trained only on training years. That depends on Step 4's artifact creation and whether the loaded files are the current artifacts.

This is especially important because stale model files could be loaded from a previous run.

Recommendation:

- Keep the statement, but qualify it as describing the intended Step 4/Step 5 workflow.
- Verify artifact provenance if the project has a saved training-year or split metadata record.
- Avoid claiming the artifact's training history was independently verified unless that check exists.

The code shown supports “Step 5 does not fit or tune models,” but not independently proving the full history of every loaded artifact.

### R8 — Low: `METRICS` excludes test class counts and the report does not show class coverage

Location: `evaluations.py`, `METRICS`, `evaluate_models`, and `write_report`

The report shows:

- Test row count.
- Accuracy.
- Precision.
- Recall.
- F1-score.

This matches the required metrics in the excerpt.

However, the report does not show how many test rows are benign and how many are malware. Since the report itself emphasizes that malware is the positive class, class counts would help interpret the precision and recall results.

This is not a requirement violation based on the supplied code, but it would improve the report's context.

Recommendation:\
Optionally add:

- `n_benign_test`
- `n_malware_test`

to the results table, or include a separate test class balance summary.

Do not add class counts to `METRICS`, because they are not performance metrics.

### R9 — Low: The report is hardcoded to four metrics and a specific metric interpretation

Location: `evaluations.py`, `write_report`

The report manually lists accuracy, precision, recall, and F1. This matches the current `METRICS` tuple, so it is correct today.

But the wording and table labels are maintained separately from `METRICS`. If the metric list changes, the report could become inconsistent.

Recommendation:\
Keep the current four required metrics, but derive report wording and validation from `METRICS` where practical. This is a maintainability improvement, not a blocker.

## Reviewer Priority Summary

| Priority | Finding                                                                            | Recommended action                                                     |
| -------- | ---------------------------------------------------------------------------------- | ---------------------------------------------------------------------- |
| High     | R1 — Loaded model artifacts are not validated against the current feature contract | Validate fitted state, feature order, prediction interface, and output |
| High     | R2 — Empty or single-class evaluation inputs are not explicitly handled            | Validate input shape, nonempty arrays, and class coverage              |
| Medium   | R3 — Test labels and feature rows are not independently checked for alignment      | Reuse Step 3 split/label validation                                    |
| Medium   | R4 — Report assumes first row is the winner                                        | Sort explicitly and validate one row per classifier                    |
| Medium   | R5 — Custom output directories are not consistently created                        | Resolve and create all requested directories                           |
| Medium   | R6 — CSV can be written before report validation fails                             | Validate first and write outputs consistently                          |
| Medium   | R7 — Report claims artifact training history that Step 5 does not verify           | Qualify the claim or validate artifact provenance                      |
| Low      | R8 — Test class coverage is not reported                                           | Add optional class counts for context                                  |
| Low      | R9 — Metric report wording is maintained separately from `METRICS`                 | Derive report content dynamically                                      |

## What Should Remain Unchanged

These parts of Step 5 are good and should be preserved:

- Rebuilding the dataset and using the same time-based split.
- Using `expected_columns=X_train.columns` for the test feature matrix.
- Calling `check_feature_contract` before evaluation.
- Loading the Step 4 fitted pipelines instead of fitting new models.
- Evaluating each classifier on the same test matrix.
- Computing accuracy, precision, recall, and F1.
- Treating malware label `1` as the positive class.
- Explicit `zero_division=0` for precision, recall, and F1.
- Sorting the evaluation table by accuracy.
- Keeping test evaluation separate from model selection.
- Saving machine-readable test metrics and a human-readable report.

Pasted text(3).txt

## Reviewer Verdict

Revise before marking Step 5 complete.

The overall evaluation workflow is appropriate and keeps model fitting and hyperparameter selection separate from held-out test evaluation.

The most important fixes are:

1. Validate the loaded models against the current Step 3 feature contract.
2. Validate test labels, row alignment, and class coverage before calculating metrics.
3. Make `write_report` independently determine the highest-accuracy model.
4. Resolve and create custom output directories consistently.
5. Validate report inputs before writing output files.

No major redesign is needed. The existing evaluation structure and metric definitions should remain.

## Student Decision Log

| Review ID | Decision (Accept / Reject / Defer) | Notes |
| --------- | ---------------------------------- | ----- |
| R1        |              Accept                | Valid |
| R2        |              Accept                | Valid |
| R3        |              Accept                | Valid |
| R4        |              Accept                | Valid |
| R5        |              Accept                | Valid |
| R6        |              Accept                | Valid |
| R7        |              Accept                | Valid |
| R8        |                Deny                |Unneeded|
| R9        |              Accept                | Valid |

Review is based on the supplied `evaluations.py` code and updated `main.py` excerpt. The full repository and runtime results were not inspected.