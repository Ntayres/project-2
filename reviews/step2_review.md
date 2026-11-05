Copy everything below into a file named `step2_code_review.md`.

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
| R12       |  Reject  |    Unneeded    |
| R13       |  Accept  |     Valid      |
