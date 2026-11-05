# Step 3 Code Review

Files reviewed: `preprocessing.py` and `main.py` Step 3 changes.

## Overall Assessment

Step 3 is structured well and follows the project's central requirements: a chronological year-based split, metadata exclusion, preprocessing fitted on training data, and forward-chaining validation folds.

The new preprocessing module is a good separation of concerns. It provides reusable functions for splitting, feature extraction, pipeline construction, fold generation, and reporting. The main entry point also follows a logical sequence: split → summarize → fit preprocessing → verify → save → report.

The most important remaining concerns are:

- The Step 3 saved preprocessing pipeline is always fitted with scaling, despite the model-specific scaling policy.
- The split validation does not fully verify that the declared year lists correspond to the original dataset's year partition.
- Some edge cases can cause preprocessing or report generation to fail without a clear explanation.
- Several report statements describe checks more strongly than the code actually verifies.
- Step 3's feature matrix and saved artifacts should be validated more explicitly before Step 4 consumes them.

Recommendation: Revise before proceeding to Step 4. The overall design is appropriate; the requested changes are primarily validation, consistency, and preventing misleading report claims.

## Review Comments

### R1 — Use the model-specific preprocessing policy when fitting the saved pipeline

Priority: High\
Location: `main.py` — `run_step3()`, lines 173–175

The preprocessing module defines:

```
SCALING_BY_MODEL = {    "LogisticRegression": (True, ...),    "LinearSVC": (True, ...),    "SGDClassifier": (True, ...),    "RandomForestClassifier": (False, ...),}
```

It also defines:

```
def preprocessor_for(model_name):    ...
```

However, Step 3 currently fits:

```
pipeline = preprocessing.make_preprocessor(scale=True)pipeline.fit(X_train)
```

This means the saved Step 3 pipeline always contains a scaler, even though the model policy explicitly says Random Forest should not use one.

The report later states that Random Forest uses the same dropped-column set without scaling. That may be true in Step 4 if Step 4 correctly uses `preprocessor_for("RandomForestClassifier")`, but the saved Step 3 artifact itself does not represent that model-specific choice.

Recommendation:

Clarify the role of the Step 3 saved pipeline.

Two reasonable approaches:

1. Save a general preprocessing artifact: Fit the constant-column removal step once, but document that Step 4 constructs each model's pipeline using `preprocessor_for(model_name)`. Do not describe the general saved artifact as the preprocessing pipeline for every model.
2. Save model-specific artifacts: Fit a pipeline for each model using `preprocessor_for(model_name)` and save the resulting artifacts separately.

For the current project structure, the first approach may be simpler. The important point is that Step 3's report and artifacts should not imply that all models use the same scaler configuration.

### R2 — Verify that the split years exactly partition the original dataset

Priority: Medium\
Location: `preprocessing.py` — `split_by_year()` and `check_split()`, lines 43–83

The split is correctly created using the year column:

```
split = Split(    train=df[year_col.isin(train_years)].copy(),    test=df[year_col.isin(test_years)].copy(),    train_years=train_years,    test_years=test_years)
```

The validation checks year overlap, chronological ordering, and total row count.

However, `check_split()` does not receive the original dataset or its complete year set. It verifies that the frames contain the years declared in the split, but it does not directly verify that the declared training and testing year lists together contain every year from the original dataset.

The total row count is useful, but by itself it does not establish that the year partition is correct.

Recommendation:

Pass the original year set into the validation or validate the partition directly inside `split_by_year()`.

Check that:

- Every original year belongs to exactly one split.
- No original year is omitted.
- No year outside the original dataset is declared.
- Training and testing years are disjoint.
- Every test year is later than every training year.

This would make the split check more directly establish the project's chronological partition policy.

### R3 — Handle datasets where every training feature is constant

Priority: Medium\
Location: `preprocessing.py` — `make_preprocessor()` and `kept_features()`, lines 102–126

The pipeline uses:

```
VarianceThreshold(threshold=0.0)
```

This is appropriate for removing features that never vary in the training data.

However, if every feature is constant in the training years, `VarianceThreshold` cannot produce a valid nonempty feature matrix. The failure will occur during fitting or when the retained feature names are retrieved.

The current code does not explicitly validate that at least one feature survives.

Recommendation:

After fitting the pipeline, verify that at least one feature remains.

If no features survive, raise a descriptive error that identifies the training years and explains that every feature was constant.

This is preferable to allowing a later classifier or report step to fail with an unrelated shape error.

### R4 — Verify that the feature matrix contains only the intended feature columns

Priority: Medium\
Location: `preprocessing.py` — `features_labels()`, lines 86–99

The function correctly obtains the feature matrix through:

```
X = df[feature_columns(df)]
```

It also checks for metadata columns and verifies the expected feature count.

This is good defensive behavior.

However, the metadata check occurs after `feature_columns(df)` has already selected the columns. If `feature_columns()` itself incorrectly includes a metadata column, the check catches it. If the feature-selection helper silently excludes or omits a legitimate feature, the expected count check catches a count mismatch, but not necessarily a wrong feature identity or order.

Recommendation:

Keep the existing checks, but strengthen the feature contract.

Verify that:

- The selected feature names match the expected feature list.
- Metadata columns are excluded.
- The number of features matches the expected count.
- Feature names are unique.
- Feature values are numeric and finite.
- The feature order remains consistent between training and testing.

The feature order is especially important because Step 4 will reuse the preprocessing artifact.

### R5 — Make the scaler verification robust to an empty retained feature set

Priority: Medium\
Location: `preprocessing.py` — `verify_scaler()`, lines 183–200

The scaler verification correctly recomputes the mean and population standard deviation from the retained training columns.

It also accounts for `StandardScaler` storing a scale of `1.0` for a zero-variance feature.

However, this calculation assumes at least one feature remains:

```
np.abs(scaler.mean_ - expected_mean).max()
```

If the retained feature list is empty, the maximum of an empty array is undefined.

Recommendation:

Validate the retained feature count before computing the differences.

If the pipeline has no retained features, raise a descriptive error rather than attempting to calculate the maximum difference.

Also consider checking that the scaler's `mean_` and `scale_` arrays have the same length as the retained feature list.

### R6 — Ensure the report accurately describes the scaler check

Priority: Medium\
Location: `preprocessing.py` — `verify_scaler()` and `write_report()`, lines 183–200 and 303–315

The scaler check compares the fitted scaler's statistics with recomputed training-set statistics.

That is a useful verification.

However, the report should distinguish between:

1. Verifying that the fitted scaler stores the expected training statistics.
2. Verifying that transforming the training data produces standardized features.
3. Verifying that test data did not contribute to fitting.

The current implementation checks the first property directly. It also computes transformed training statistics in `run_step3()`.

Those checks are useful, but the report should avoid treating them as independent proof of every leakage-prevention claim.

Recommendation:

Keep the numerical checks, but describe exactly what each establishes.

For example:

- The scaler-statistics check verifies that the fitted scaler's stored statistics match the training data.
- The transformed-training summary describes the resulting training feature distribution.
- The chronological split and training-only fit establish the intended separation from test years.

The existing checks should remain, but the report should not imply that a mean/std check alone proves the entire pipeline is leakage-free.

### R7 — Avoid making the report depend on a fixed number of training years

Priority: Medium\
Location: `preprocessing.py` — `forward_chaining_folds()` and `write_report()`, lines 135–146 and 203–332

The fold generator correctly requires at least two training years:

```
if len(train_years) < 2:    raise ValueError(...)
```

This is necessary because forward-chaining validation needs an earlier year for training and a later year for validation.

The report also references:

```
train_years[-1]
```

and assumes that the final fold trains on the earlier years and validates on the last training year.

That assumption is consistent with the current fold generator.

However, the split policy and report structure should be validated together so the report does not fail unexpectedly if the year range changes.

Recommendation:

Validate the minimum number of training years before the pipeline is fitted and before artifacts are written.

If the project requires a fixed training/test year structure, check that structure explicitly.

If the year range is allowed to change, make the report dynamically describe the resulting fold structure.

### R8 — Make the report's training/test totals robust to missing split labels

Priority: Medium\
Location: `preprocessing.py` — `write_report()`, lines 213–240

The report calculates:

```
totals = summary.groupby("split")[["rows", "benign", "malware"]].sum()
```

It then accesses:

```
totals.loc["train", ...]totals.loc["test", ...]
```

This assumes the summary contains both split labels.

The split validation normally guarantees nonempty year lists, but the report itself does not validate that both groups appear in the summary.

If the summary is incomplete or inconsistent, the report can fail with a `KeyError`.

Recommendation:

Before accessing the totals, validate that:

- Both `train` and `test` are present.
- The row counts are positive.
- The sum of summary rows matches the original dataset size.
- The benign and malware counts sum to the total rows for each year.

Raise a clear error if the report inputs are inconsistent.

### R9 — Validate class labels before computing class balance

Priority: Medium\
Location: `preprocessing.py` — `split_summary()`, lines 167–180

The summary calculates benign and malware counts using:

```
labels == config.LABEL_MAP["benign"]labels == config.LABEL_MAP["malware"]
```

This is appropriate if the dataset contains only those two known labels.

However, if an unexpected label appears, it will not be counted as either benign or malware.

The report could then display class counts that do not add up to the total number of rows.

Recommendation:

Validate that the label column contains only the expected label values before computing the summary.

Alternatively, include an explicit count of unknown labels and fail if it is nonzero.

This would make the class-balance report more reliable and prevent silently misrepresenting the dataset.

### R10 — Validate that dropped features are recorded consistently

Priority: Medium\
Location: `main.py` — `run_step3()`, lines 173–204

The code correctly obtains the retained features from the fitted pipeline:

```
kept = preprocessing.kept_features(pipeline)
```

and derives the dropped list from the training columns.

It then records whether each dropped feature varies in the test years.

This is a useful diagnostic because it identifies features that were constant in training but varied later.

However, the artifact-writing code assumes the dropped list and test feature columns are consistent.

Recommendation:

Before writing the dropped-feature CSV, verify that:

- Every dropped feature existed in the original training matrix.
- Every dropped feature exists in the test matrix.
- Retained and dropped features are disjoint.
- Retained and dropped features together account for the original feature set.
- The order of retained features matches the fitted pipeline.

This would make the generated feature lists safer for Step 4 to consume.

### R11 — Clarify the interpretation of dropped features that vary in the test years

Priority: Low\
Location: `preprocessing.py` — `write_report()`, lines 209–271

The report correctly notes that a feature may be constant during training but vary during testing.

That is a useful observation about temporal changes in the dataset.

However, the report describes this as a concrete example of drift measured in Step 2.

A feature varying in the test years does establish that the feature is not constant in the test period, but it does not necessarily establish a distributional drift of a particular magnitude.

Recommendation:

Keep this analysis, but use precise wording.

For example:

> These features were constant in the training years but varied in the test years. This indicates a change in feature variability across the split and is consistent with the temporal changes examined in Step 2.

Avoid implying that this check independently establishes the magnitude or statistical significance of drift.

### R12 — Make the transformed-training summary's interpretation explicit

Priority: Low\
Location: `main.py` — `run_step3()`, lines 187–195

The code computes:

```
sample_stats = {    "n_rows": len(X_train),    "mean": float(transformed.mean(axis=0).mean()),    "std": float(transformed.std(axis=0).mean())}
```

This calculates:

- The average of the retained features' transformed means.
- The average of the retained features' transformed standard deviations.

It does not calculate the standard deviation of the entire transformed matrix.

The report currently says:

> Across all training rows the transformed features have a mean ... and a standard deviation ... on average across features.

That is close, but could be more precise.

Recommendation:

Use wording such as:

> The average feature mean across the transformed training set is ..., and the average feature standard deviation is ....

This avoids confusion between a global matrix standard deviation and an average of per-feature standard deviations.

### R13 — Validate input before saving artifacts

Priority: Medium\
Location: `main.py` — `run_step3()`, lines 197–211

The pipeline and feature lists are saved before the report is generated:

```
joblib.dump(...)
```

and then the report is written.

If a later fold check or report-generation step fails, Step 3 may leave behind saved artifacts even though the step did not complete successfully.

This can make it unclear whether the outputs represent a fully successful Step 3 run.

Recommendation:

Perform all validation that can fail before writing the final artifacts.

At minimum:

- Validate the split.
- Validate the feature matrices.
- Validate that the pipeline has retained features.
- Validate scaler statistics.
- Validate fold generation and fold indices.
- Validate report inputs.

Then write the artifacts and report.

Alternatively, clearly mark outputs as incomplete if report generation fails.

This would reduce the risk that Step 4 uses stale or partially updated artifacts.

### R14 — Ensure Step 4 uses the intended preprocessing interface

Priority: High\
Location: `preprocessing.py` — `preprocessor_for()` and `main.py` — `run_step3()`

Step 3 provides several interfaces for later use:

```
preprocessor_for(model_name)forward_chaining_folds(...)fold_indices(...)features_labels(...)
```

The report explicitly states that Step 4 should wrap each classifier in the corresponding pipeline and refit preprocessing during each fold.

That is the correct direction.

However, the current Step 3 implementation also saves a single fitted pipeline. Step 4 must not simply load that fitted pipeline and transform all data before model selection.

Doing so would violate the intended fold-specific preprocessing behavior.

Recommendation:

Make the Step 4 contract explicit:

- Step 4 uses `preprocessor_for(model_name)` to create a fresh pipeline for each model.
- The pipeline is fitted inside each forward-chaining training fold.
- Validation data is transformed by the fold-fitted pipeline.
- Test years are not used during model selection.
- The saved Step 3 pipeline is a diagnostic artifact, not a substitute for fold-specific fitting.

This is an important boundary to preserve when implementing Step 4.

### R15 — Avoid unused variables in `run_step3()`

Priority: Low\
Location: `main.py` — `run_step3()`, lines 169–172

The function obtains:

```
X_train, y_train, years_train = ...X_test, y_test, years_test = ...
```

but the supplied Step 3 implementation does not use `y_train`, `y_test`, or `years_test`.

The training year array is used for fold-index validation.

Recommendation:

Either remove the unused variables from the assignments or use them for explicit checks.

For example, verify that the labels and years returned by `features_labels()` match the corresponding split frames.

This would make the feature-extraction step more self-checking and reduce unused variables.

## Reviewer Priority Summary

| Priority | Comments           | Action                                                                            |
| -------- | ------------------ | --------------------------------------------------------------------------------- |
| High     | R1, R14            | Clarify model-specific scaling and preserve fold-specific preprocessing in Step 4 |
| Medium   | R2–R5, R7–R10, R13 | Strengthen split, feature, label, report, and artifact validation                 |
| Low      | R6, R11, R12, R15  | Improve report precision and clean up unused variables                            |

## What Should Remain Unchanged

The following parts of Step 3 are good and should remain in the implementation:

1. Chronological year-based split\
   The split is based on years rather than randomly shuffled rows. This preserves the project's intended temporal evaluation.
2. Pipeline-based preprocessing\
   `VarianceThreshold` and `StandardScaler` are placed inside a scikit-learn pipeline. This supports refitting preprocessing on each fold's training data.
3. Metadata exclusion\
   The feature extraction function explicitly checks that metadata does not reach the model.
4. Forward-chaining validation\
   The fold generator trains on earlier years and validates on the next year. This is appropriate for time-ordered model selection.
5. Training-only scaler verification\
   Recomputing scaler statistics from the training data is a useful numerical check.
6. Retained and dropped feature artifacts\
   Saving the feature lists is useful for reproducibility and later model interpretation.
7. Per-year class-balance summary\
   The summary provides context for the chronological split and can help interpret Step 4 and Step 6 results.
8. Automated report generation\
   Generating the report from the actual split and pipeline objects is a good approach. The report should remain data-driven.

## Reviewer Verdict

Revise before proceeding to Step 4.

Step 3 has a sound structure and implements the most important project requirements: chronological splitting, metadata exclusion, training-only preprocessing, and forward-chaining validation.

The highest-priority issue is the relationship between the model-specific scaling policy and the single always-scaled pipeline fitted in Step 3. Step 4 must use fresh model-specific pipelines within each fold rather than relying on the saved Step 3 pipeline.

The remaining changes are primarily validation and reporting improvements:

- Validate the year partition against the original dataset.
- Handle empty retained feature sets.
- Validate labels and feature identities.
- Ensure report inputs are complete.
- Make artifact writing occur only after successful validation.
- Clarify what the scaler and transformed-training checks actually establish.

No major redesign is needed. The existing structure can be retained with these changes.

## Student Decision Log

| Review ID | Decision | Reason / Notes |
| --------- | -------- | -------------- |
| R1        |  Accept  |     Valid      |
| R2        |  Accept  |     Valid      |
| R3        |  Accept  |     Valid      |
| R4        |  Accept  |     Valid      |
| R5        |  Accept  |     Valid      |
| R6        |  Accept  |     Valid      |
| R7        |  Accept  |     Valid      |
| R8        |  Accept  |     Valid      |
| R9        |  Accept  |     Valid      |
| R10       |  Accept  |     Valid      |
| R11       |  Accept  |     Valid      |
| R12       |  Accept  |     Valid      |
| R13       |  Accept  |     Valid      |
| R14       |  Accept  |     Valid      |
| R15       |  Accept  |     Valid      |

Review based on the supplied Step 3 excerpts from `preprocessing.py` and `main.py`. The full repository, `config.py`, `data_loading.py`, and the current Step 4 implementation were not included in this review, so comments about their behavior are limited to the interfaces shown here.