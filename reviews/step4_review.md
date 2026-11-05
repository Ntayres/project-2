# Step 4 Code Review

Files reviewed

- `models.py` — classifier specifications, preprocessing pipelines, forward-chaining checks, hyperparameter search, reproducibility checks, and report generation.&#x20;

  Pasted text(2).txt

- `main.py` — updated Step 4 workflow and integration with Steps 1–3.&#x20;

  Pasted text (2)(1).txt



Review scope: This review is based on the supplied excerpts. I have not run the project or inspected the full repository, including `config.py`, `preprocessing.py`, `PLAN.md`, and `AGENTS.md`.

## Overall Assessment

Step 4 is structured well and addresses several important concerns from the Step 3 review:

- Each model gets a fresh preprocessing pipeline.
- Forward-chaining folds are used instead of random cross-validation.
- The search refits the winning pipeline on all training years.
- Model selection uses training data rather than scoring test years.
- The code records multiple metrics, selected parameters, fit time, and a reproducibility check.

The main remaining concerns are test-data handling, validation of fold indices and model-search inputs, report accuracy, and whether the reported reproducibility check is strong enough to support its wording.

Verdict: Revise before proceeding to Step 5. The core approach is appropriate; the findings below are targeted fixes rather than a redesign.

## Review Comments

### R1 — High: Step 4 loads and preprocesses test-year data despite claiming not to touch it

Location: `main.py`, lines 267–275

Step 4 calls `features_labels` on both `split.train` and `split.test`, then runs `check_feature_contract` and `check_labels_years` before deleting `X_test`, `y_test`, and `years_test`.

The code does not fit or score against test rows, which is good. However, the docstring and comments claim Step 4 does not load or touch test years. That is not literally true: it constructs the test feature matrix and uses test labels and years for validation.

This is a distinction worth preserving because the project’s leakage policy should be precise.

Recommendation:

- Reuse Step 3’s validated split and feature contract without constructing test matrices in Step 4.
- If rebuilding the dataset is necessary, validate only the training portion in this step.
- Update the report and comments to distinguish “test data is not used for model selection” from “test data is never loaded.”

The current code appears to avoid model-selection leakage, but the stronger claim is unsupported.

### R2 — High: `check_folds` does not validate fold indices before indexing the year array

Location: `models.py`, lines 104–129

`check_folds` directly evaluates:

```
years[train_idx]years[val_idx]
```

It checks overlap, empty halves, allowed years, and chronological ordering, but it does not first verify that the indices are valid for the dataset.

Malformed indices can produce an `IndexError`, and negative indices can silently refer to rows at the end of the array. Duplicate indices can also make a fold's row membership differ from what the caller intended.

The function's current checks do not prove that each fold is a valid partition of the intended rows.

Recommendation:\
Validate before indexing:

- Indices are integer indices.
- Indices are within `[0, len(years))`.
- Each half contains unique row indices.
- The training and validation halves are disjoint.
- Each fold's indices correspond to the intended years and rows.

If `preprocessing.fold_indices` already guarantees some of these properties, document that contract and still make `check_folds` robust to malformed input.

### R3 — Medium: Fold validation does not confirm that each validation fold is the intended next year

Location: `models.py`, lines 104–129

The chronological check requires:

```
max(fold_train) < min(fold_val)
```

That prevents training on a year equal to or later than the validation year. However, it does not verify that validation contains exactly the expected next year from the generated forward-chaining fold plan.

For example, a fold that validates on a later year while skipping the intended validation year could pass the current check.

Recommendation:\
Compare each fold against the supplied `folds` plan or explicitly validate:

- The validation year is the expected next year.
- The training years are exactly the intended prefix.
- Validation contains only the intended year.
- Fold ordering matches the plan.

This makes the check verify the forward-chaining policy, not merely a weaker chronological property.

### R4 — Medium: `reproducibility_check` can pass without testing the same intended training conditions

Location: `models.py`, lines 132–147

The reproducibility check selects a random subset of training rows and fits the cloned pipeline twice. It requires at least two classes, then compares predictions.

This is a useful lightweight determinism check, but it has limitations:

- It checks only the sampled rows.
- It compares predictions on the same rows used to fit each clone.
- It does not compare fitted parameters or preprocessing state.
- It does not establish that the full training fit is reproducible.
- It may not exercise rare classes or rare feature values.

The report currently describes identical predictions as evidence that the seeds fix the selected configuration. That wording is stronger than the check supports.

Recommendation:\
Keep the check, but describe it as a sampled prediction-determinism check. If stronger evidence is required, compare:

- Predictions on a fixed holdout subset of training data.
- The fitted preprocessing feature names and scaler statistics.
- Model parameters or predictions from repeated fits.

Avoid claiming that the check proves full-fit reproducibility.

### R5 — Medium: `best_params_row` assumes every metric and fold score is finite

Location: `models.py`, lines 166–182

The function converts all mean and standard deviation scores to `float` and records per-fold accuracy. The grid search uses `error_score="raise"`, which is helpful for estimator failures, but it does not guarantee every metric is finite.

For example, ROC AUC can be undefined if a validation fold contains only one class. In that case, the report may contain `NaN` values or fail to provide a meaningful comparison.

Recommendation:\
Before building the summary row:

- Check that the expected metric keys exist.
- Validate that each fold's validation data contains both classes if ROC AUC is required.
- Check score values for `NaN` or infinity.
- Fail with a clear message or explicitly record an unavailable metric.

Do not silently turn undefined fold metrics into ordinary report values.

### R6 — Medium: The report's class-weight decision is based on training-year shares but does not validate the summary input

Location: `models.py`, lines 367–376

The code uses:

```
shares = summary[summary["split"] == "train"]["malware_share"]
```

Then it compares the minimum and maximum shares. This is a reasonable way to describe training-year class balance, but it assumes:

- The summary contains training rows.
- `malware_share` is numeric and finite.
- Every training year is represented.
- The summary is consistent with `split.train_years`.

If the summary is incomplete or malformed, the decision note can be misleading or fail.

Recommendation:\
Validate the summary against `split.train_years` before deriving the note. Confirm that every training year appears once and that its malware share is finite and within `[0, 1]`.

Also ensure the note reflects the actual rule in `PLAN.md`; the excerpt only states that the plan does not treat the observed range as a large imbalance.

### R7 — Medium: The report hardcodes the number of validation folds

Location: `models.py`, lines 278–280

The report says:

> “Mean over the three validation folds”

But the report is generated from `folds` and `search.n_splits_`. If the fold plan changes, this sentence can become stale while the numerical tables remain dynamically generated.

Recommendation:\
Generate the wording from the actual fold count, for example:

```
f"Mean over {len(folds)} validation folds..."
```

Also validate that `len(folds)` agrees with the fold count recorded by the fitted search.

### R8 — Medium: The report's “real classifiers beat the baseline” claim uses a strict accuracy comparison without checking score validity

Location: `models.py`, lines 206–210 and 284–294

The report counts real classifiers whose mean CV accuracy exceeds the DummyClassifier mean. This is a clear comparison, but the logic assumes:

- Exactly one DummyClassifier row exists.
- All expected real classifiers are present.
- Scores are finite.
- The rows correspond to the model specifications.

If the baseline is missing, duplicated, or has an invalid score, the report can omit or misstate the comparison.

Recommendation:\
Before generating the report, validate that:

- Every model in `MODEL_SPECS` has exactly one summary row.
- Exactly one DummyClassifier summary exists.
- The compared CV accuracy values are finite.

The report should also keep the interpretation narrow: exceeding the baseline on CV accuracy does not by itself establish that a model learned useful malware-detection behavior across all classes.

### R9 — Medium: The saved-model/report writes are not fully transactional

Location: `models.py`, lines 378–385

The code writes the CSV files, then saves each model, then writes the report. If a later model save or report generation fails, the output directory may contain a mixture of newly generated and stale Step 4 artifacts.

The docstring says validation first, then fits, then writes, which is good. But the final writing phase can still fail partway through.

Recommendation:\
For stronger artifact consistency:

- Write outputs to a temporary Step 4 directory.
- Verify the expected model and result files were produced.
- Replace the final artifacts only after all writes succeed.

At minimum, ensure a failed run cannot leave a partial set that looks like a complete Step 4 result.

### R10 — Low: `write_report` receives `cv_table` but does not use it

Location: `models.py`, line 202 and report construction

The report function accepts:

```
write_report(best, cv_table, folds, ...)
```

but the excerpt builds the report tables from `best` and does not use `cv_table`.

This is not a correctness bug by itself, but it makes the function signature suggest that the report includes or validates the full candidate table when it does not.

Recommendation:\
Either:

- Remove `cv_table` from the report function if it is intentionally unused, or
- Use it to validate that each model's candidate results are present and consistent with the selected summary.

### R11 — Low: The report describes a fixed set of metrics and model types

Location: `models.py`, lines 243–340

The report text explicitly says there are four estimators, three real classifiers, three validation folds, and a fixed set of metrics.

Those are correct for the current model specifications, but several statements are manually maintained:

- “Four scikit-learn estimators”
- “Three of them are real classifiers”
- “Mean over the three validation folds”
- “all five metrics”

Recommendation:\
Derive model count, real-classifier count, fold count, and metric names from `MODEL_SPECS`, `folds`, and `SCORING`. This will reduce the chance that future Step 4 changes make the report inaccurate.

## Reviewer Priority Summary

| Priority | Finding                                                                    | Recommended action                                     |
| -------- | -------------------------------------------------------------------------- | ------------------------------------------------------ |
| High     | R1 — Test data is loaded in Step 4 despite the stronger no-touch claim     | Remove test matrix construction or correct the claim   |
| High     | R2 — Fold indices are not fully validated before indexing                  | Validate index type, range, uniqueness, and membership |
| Medium   | R3 — Fold validation does not enforce the exact forward-chaining plan      | Validate expected train/validation years               |
| Medium   | R4 — Reproducibility check supports a narrower claim than the report makes | Qualify report wording or strengthen check             |
| Medium   | R5 — Undefined/nonfinite fold metrics are not handled explicitly           | Validate metric availability and finiteness            |
| Medium   | R6 — Class-balance report input is assumed valid                           | Validate training-year coverage and shares             |
| Medium   | R7 — Fold count is hardcoded in report prose                               | Derive from actual folds                               |
| Medium   | R8 — Baseline comparison assumes complete valid model rows                 | Validate model rows and scores                         |
| Medium   | R9 — Writes can leave partial/stale artifacts                              | Write to temporary outputs and commit after success    |
| Low      | R10 — Unused `cv_table` argument                                           | Remove or use for consistency checks                   |
| Low      | R11 — Report has manually maintained counts/metric wording                 | Derive dynamic values                                  |

## What Should Remain Unchanged

These parts are good and should be preserved:

- The use of `preprocessing.preprocessor_for(name)` to create a fresh model-specific pipeline.
- The use of forward-chaining validation inside the training years.
- The use of `GridSearchCV(refit="accuracy")` to select by the assignment's primary metric and refit on the full training half.
- Keeping the test years out of hyperparameter selection and model scoring.
- Recording accuracy, precision, recall, F1, and ROC AUC.
- Explicit `zero_division=0` for precision, recall, and F1.
- Fixed random states in the model constructors.
- Saving the preprocessing and classifier together in the fitted pipeline.
- The intent to run validations before fitting or writing artifacts.



Pasted text(2).txt

Pasted text (2)(1).txt



## Reviewer Verdict

Revise before Step 5.

The model-selection design is sound, and the Step 3 model-specific preprocessing concern is addressed. The highest-priority fixes are to make Step 4's test-data handling match its stated policy and to strengthen fold-index validation. Then tighten report validation and wording so the generated documentation cannot overstate what the checks establish.

The next step does not require changing the classifier choices or replacing forward-chaining validation. It mainly requires making the guards and claims as precise as the pipeline design already is.

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
| R8        |              Accept                | Valid |
| R9        |              Accept                | Valid |
| R10       |               Deny                 |Unneeded|
| R11       |              Accept                | Valid |

Review is based on the supplied `models.py` and updated `main.py` excerpts; the full repository and runtime results were not inspected.