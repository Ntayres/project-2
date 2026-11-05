# Step 0 Plan Review

**Reviewer:** Reviewer agent\
**Project:** IT348/448 Intro to ML --- Project 2\
**Files reviewed:** `AGENTS.md`, `PLAN.md` (v1), and
`Project_2_updated.docx`\
**Review scope:** Step 0 plan review only. No project code was reviewed
or changed.

## Overall assessment

The plan is well organized and covers all seven required project steps.
It demonstrates good awareness of temporal data leakage, time-ordered
validation, model evaluation, and the need to assess performance as the
dataset changes over time.

The plan does not need to be redesigned. However, several targeted
changes are recommended before Step 1 begins. The most important issues
are:

1.  Section 0 of `PLAN.md` contains dataset unknowns that are already
    answered in `AGENTS.md`.
2.  Duplicate-app analysis should preserve the application identifier
    long enough to check duplicates.
3.  The year split should not be changed based on test-year drift
    observations.
4.  Step 2 should reflect that the features are integer API-call counts,
    not binary features.
5.  Step 3 must ensure preprocessing is fitted separately inside each
    time-ordered validation fold.
6.  Step 7 should make the comparison between model-update strategies as
    fair as possible.
7.  The plan should explicitly track all final submission deliverables.

------------------------------------------------------------------------

## Review comments

### R1. Update the outdated dataset information

**Priority:** High\
**Location:** `PLAN.md`, Section 0, especially lines 14--19.

The plan lists the file naming pattern, file format, feature types, row
counts, and class-per-file structure as unknown. These facts are already
documented in `AGENTS.md`:

-   Files follow `sampled_<YEAR>_<CLASS>_api.csv`.
-   Files are CSVs with a header row.
-   Each file has 1,500 rows and 1,849 columns.
-   The first column is `apkname`; the remaining 1,848 columns are
    API-call features.
-   Feature values are non-negative integer counts.
-   The class is encoded in the filename: malware = 1 and benign = 0.

The exact year range is still unknown and should remain an open item
until the input directory is inspected.

**Recommendation:** Revise Section 0 to distinguish known facts from
genuinely unresolved facts. Step 1 should validate the documented
assumptions and discover the exact year range rather than rediscovering
the file format and feature types.

### R2. Preserve application identifiers for duplicate analysis

**Priority:** High\
**Location:** `PLAN.md`, Step 1 and the duplicate-related leakage rule.

The dataset contains repeated application identifiers in the files
already inspected. The current plan says to drop the first column during
loading and also investigate duplicate applications. If `apkname` is
discarded immediately, the project loses the identifier needed to count
repeated application names across files and years.

There are two different kinds of duplicates to investigate:

1.  **Duplicate identifiers:** the same `apkname` occurs more than once.
2.  **Duplicate feature vectors:** two rows have identical API-call
    feature values, even if their application identifiers differ.

These should be counted and reported separately. Identical feature
vectors do not necessarily prove that the rows represent the same
application.

**Recommendation:** Preserve `apkname` as metadata during data
inspection and duplicate analysis, but exclude it from the model feature
matrix. Report duplicate identifiers within each file and across years,
as well as identical feature vectors within and across years. Do not
remove duplicates silently; document the counts and obtain the student's
decision before dropping any.

### R3. Keep the chronological split independent of test-year drift

**Priority:** High\
**Location:** `PLAN.md`, Year split policy, lines 30--46.

The proposed split of Y1--Y4 for training and Y5--Y8 for testing is a
reasonable default. However, the plan says the split may change if Step
2 reveals a better cut, such as a sharp drift point.

Step 2 analyzes all eight years. If test-year feature distributions are
used to choose the training/testing boundary, the test data influences
an experimental design decision. Even without using test labels, this
weakens the independence of the final evaluation.

**Recommendation:** Fix the Y1--Y4 / Y5--Y8 split before examining
test-year drift results. Step 2 may describe drift across all years, but
test-year drift should not be used to choose the split boundary, select
features, or tune model parameters. If the split must change for a
compelling reason, the reason and decision should be documented before
test results are used to guide it.

### R4. Adjust Step 2 to the actual count-valued features

**Priority:** High\
**Location:** `PLAN.md`, Step 2, lines 71--78.

The plan says the API-call features are probably binary or sparse.
`AGENTS.md` states that the feature values are non-negative integer
counts and can reach several hundred. The analysis should therefore be
designed and described as count-valued feature drift, not assume binary
data.

The assignment specifically suggests the Kolmogorov--Smirnov (KS) test,
so retaining it is reasonable. However, the conventional KS p-value
calibration assumes continuous distributions; integer counts contain
ties. The report should acknowledge this limitation rather than treating
the p-values as unproblematic.

**Recommendation:**

-   Retain the KS statistic as an assignment-aligned distribution
    comparison, with a note about its discrete-data limitation.
-   Retain change in prevalence (the share of apps with a nonzero count)
    because it is easy to interpret.
-   Include a count-sensitive distribution distance such as
    Jensen--Shannon distance. Wasserstein distance could be considered
    if time permits.
-   Apply Benjamini--Hochberg correction consistently and state the
    correction family clearly (for example, separately for each year
    pair and class).
-   Report effect sizes or distance values alongside statistical
    significance.

The plan should also distinguish feature drift from concept drift.
Comparing feature distributions separately for malware and benign apps
is useful class-conditional analysis, but it does not by itself
establish that the relationship between features and labels has changed.
A year classifier shows temporal distinguishability, not concept drift
by itself. Consider naming this subsection "Class-conditional drift and
temporal distinguishability," and discuss possible concept drift
alongside changes in predictive performance in Step 6.

### R5. Prevent preprocessing leakage during time-ordered validation

**Priority:** High\
**Location:** `PLAN.md`, Step 3, lines 93--98.

The plan correctly says that scaling and feature selection must be
fitted on training data only, and it proposes putting scaling in a
scikit-learn Pipeline. However, the implementation must ensure that
constant-feature removal and scaling are both inside the pipeline used
for each validation fold.

For example, if a scaler is fitted once on Y1--Y4 before a fold trains
on Y1--Y3 and validates on Y4, information from the validation year has
influenced preprocessing. That is leakage during model selection, even
though Y4 belongs to the overall training period.

**Recommendation:**

-   Put every learned preprocessing operation, including
    constant-feature removal and scaling, inside the pipeline used by
    forward-chaining validation.
-   Fit each fold's preprocessing only on that fold's training years.
-   After model selection, fit the final pipeline on all of Y1--Y4.
-   Apply that final fitted pipeline unchanged to Y5--Y8.
-   Preserve feature ordering and apply the same retained-feature set to
    training and test data.

Also consider that Random Forest generally does not require feature
scaling, while Logistic Regression and SVM commonly benefit from it. The
plan need not force every model to use an identical scaling method if
that is not appropriate.

### R6. Make Step 7's update-strategy comparison fair

**Priority:** High\
**Location:** `PLAN.md`, Step 7, lines 165--180.

The proposed strategies compare static training, expanding-window
retraining, sliding-window retraining, and incremental learning. The
incremental strategy may use `SGDClassifier`, while the other strategies
may use a different classifier selected in Step 4.

If the classifier changes as well as the update strategy, performance
differences cannot be attributed clearly to the update strategy.

**Recommendation:** Consider comparing all update strategies using the
same underlying classifier that supports the required training modes.
For example, use an `SGDClassifier` configuration for static,
expanding-window, sliding-window, and incremental strategies. The best
Step 4 static model can still be included as an additional reference,
but it should be clearly distinguished from the controlled strategy
comparison.

There is also a scaling concern. If a scaler is updated over time while
the classifier retains weights learned under the old feature
transformation, the model's feature representation changes underneath
those weights. A simpler initial design is to fit the scaler on the
initial training years and keep it fixed while incrementally updating
the classifier. If the scaler is updated, the plan should explain how
consistency of the classifier's learned representation is maintained.

The test-then-train order is appropriate: evaluate year Yt using only
earlier data, then make Yt's labels available for training the model
used on Yt+1. The assumed label delay should remain explicit in the
report.

### R7. Clarify the role of training metrics

**Priority:** Medium\
**Location:** `PLAN.md`, Step 5, lines 131--134.

Reporting training metrics can help show the difference between
in-sample performance and held-out performance. However, these metrics
should not be confused with validation or test performance.

**Recommendation:** Label training metrics explicitly as in-sample
results. Do not use them as the primary basis for model selection or as
evidence of future performance. Keep the held-out test metrics separate
and use accuracy as the primary metric, with precision, recall, and F1
as additional metrics as the assignment requests.

### R8. Add an explicit final-submission checklist

**Priority:** Medium\
**Location:** `PLAN.md`, review/test workflow and Step 0 checklist.

The plan describes the seven builder reports, reviews, tests, and
transcripts. The assignment also requires final artifacts that are not
all explicitly tracked in the plan's final checklist:

-   `README.md`, identifying the tool and model used for each agent role
    and explaining how to run the code.
-   A one-page report explaining the sustainable malware-detection
    approach.
-   A 1--2 page process reflection.
-   Full agent-session transcripts.
-   Full role-tagged git history.
-   At least two independently verified results, with the verification
    method recorded.

Some of these are mentioned in `AGENTS.md`, but the plan should
explicitly include them in its final completion checklist.

**Recommendation:** Add a final deliverables checklist that names each
artifact, identifies who is responsible for it, and confirms that it is
complete before submission.

------------------------------------------------------------------------

## Additional lower-priority recommendations

1.  **Step 1:** Assert the original shape `(1500, 1849)` before dropping
    `apkname`, then assert `(1500, 1848)` afterward.
2.  **Step 1:** Validate that every discovered year has both malware and
    benign files.
3.  **Step 2:** State whether multiple-testing correction is performed
    separately for each year pair and class.
4.  **Step 2:** Report effect sizes or distribution distances, not only
    statistical significance.
5.  **Step 3:** Preserve feature names and ordering after
    constant-feature removal.
6.  **Step 4:** Define the exact forward-chaining folds available with
    four training years and keep the hyperparameter search appropriately
    small.
7.  **Step 6:** Use a fixed label order for confusion matrices so a year
    containing only one class does not change their interpretation.
8.  **Step 7:** Record the exact training years used for every test-year
    prediction.

## What should remain unchanged

The following parts of the plan are sound and should generally be
retained:

-   The chronological Y1--Y4 / Y5--Y8 split as the initial fixed policy.
-   Forward-chaining validation within the training years.
-   Malware as the positive class.
-   Logistic Regression and Random Forest as initial classifiers.
-   Accuracy as the primary evaluation metric, with precision, recall,
    and F1 as additional metrics.
-   Per-year test performance reporting.
-   Comparison of static, expanding-window, sliding-window, and
    incremental strategies.
-   Test-then-train ordering in Step 7, with the label-delay assumption
    documented.
-   The role-separated build, review, test, and decision-log workflow.

## Reviewer verdict

**Verdict: Revise the plan before beginning Step 1.**

The plan is a strong foundation and does not require a complete
redesign. The high-priority changes concern dataset consistency,
duplicate analysis, temporal leakage, count-valued drift analysis, and
fair comparison of sustainable learning strategies. The medium-priority
changes improve clarity and help ensure that all assignment deliverables
are completed.

The reviewer recommends that the builder prepare PLAN.md v2 after the
student has considered the comments below. PLAN.md v1 should remain
unchanged in git history, and the revised plan should be committed
separately as required by the assignment.

------------------------------------------------------------------------

## Student decision log

  Review ID                        Decision (Accepted / Rejected)   Student's reason
  -------------------------------- -------------------------------- ------------------
  R1 --- Dataset information                  Accept                      Valid
  R2 --- Duplicate analysis                   Accept                      Valid
  R3 --- Year split independence              Accept                      Valid
  R4 --- Drift methodology                    Accept                      Valid
  R5 --- Preprocessing leakage                Accept                      Valid
  R6 --- Step 7 comparison                    Accept                      Valid
  R7 --- Training metrics                     Accept                      Needed
  R8 --- Submission checklist                 Rejected                    Unnedded 
  Additional recommendations                  Reject                      Unneeded

