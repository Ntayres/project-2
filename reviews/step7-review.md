Step 7 Code Review
Files reviewed
- sustainable.py — full code provided.
- main.py — updated entry point, including Step 7 integration.
Review scope: This review covers the supplied code and its integration. I have not executed the project or inspected the supporting implementations in config.py, evaluation.py, preprocessing.py, or models.py.
Overall Assessment
Step 7 has a clear experimental purpose: compare three ways of maintaining a malware classifier as data changes over time.
1. Frozen baseline: Keep the original Step 4 model unchanged.
2. Periodic retraining: Retrain using all data available before each test year.
3. Incremental adaptation: Predict a test year, then update the model using that year's labeled data.
The temporal ordering is generally handled thoughtfully. In particular, periodic retraining excludes the current test year from training, and incremental learning predicts before calling partial_fit().

The main areas I would investigate are preprocessing consistency, the assumptions behind the frozen baseline, validation of the three strategy outputs, and whether the comparison is fair and clearly described.
Review Comments
R1 — High Priority: Periodic retraining needs stronger feature-contract validation
Location: sustainable.py, _periodic_retraining()

The function creates a preprocessor for each test year using all prior data:
preprocessor = preprocessing.fit_preprocessor(    preprocessing.preprocessor_for(MODEL_NAME),    X_prior,    sorted(prior[config.YEAR_COLUMN].unique()),)



It then transforms the current year's data using that preprocessor. This is the correct general approach for avoiding future-year preprocessing leakage.

However, the code does not explicitly verify that the current year's feature columns match the prior training feature columns before transforming them.

Recommendation: Validate the feature contract between X_prior and X_current before fitting and transforming. Reuse the existing preprocessing validation helpers where appropriate.
R2 — High Priority: Frozen baseline compatibility is not checked against the current feature contract
Location: sustainable.py, _frozen_predictions()

The frozen baseline loads the Step 4 SGDClassifier and checks that it is fitted:
estimator = joblib.load(path)check_is_fitted(estimator)



However, unlike the Step 6 implementation, this function does not check whether the loaded model's expected feature names and order match the current test data.

A stale or incompatible model artifact could therefore produce an error or invalid results.

Recommendation: Add the same feature compatibility checks used in Step 6, including a feature-count check when feature names are unavailable.
R3 — High Priority: Incremental adaptation assumes the preprocessing representation remains valid
Location: sustainable.py, _incremental_update()

The incremental model fits its preprocessor once on the original training years:
preprocessor = preprocessing.fit_preprocessor(    preprocessing.preprocessor_for(MODEL_NAME),    X_train,    split.train_years,)



It then keeps that preprocessor fixed while updating the classifier with later years.

This is a defensible experimental choice because it isolates classifier adaptation from preprocessing changes. However, it also means that later-year feature distributions may be transformed using old scaling statistics.

The report explains this choice, which is good. The important point is to make sure the project intends to compare classifier updates under a fixed representation rather than a fully adaptive pipeline.

Recommendation: Keep the fixed-preprocessor approach if that is the intended experiment, but make the limitation explicit in the report's interpretation of incremental adaptation.
R4 — Medium Priority: Test-year ordering should be explicitly validated
Location: sustainable.py, _incremental_update() and _periodic_retraining()

Both strategies rely on the order of split.test_years.

Periodic retraining selects prior data using:
df[df[config.YEAR_COLUMN] < year]



Incremental adaptation processes test years in the order provided by split.test_years.

The periodic strategy is protected against future-year leakage by its filter. Incremental adaptation, however, assumes that the years are processed chronologically.

Recommendation: Explicitly verify that split.train_years and split.test_years are sorted and that every test year is later than every training year. This will make the temporal assumptions visible and easier to maintain.
R5 — Medium Priority: The frozen model's prediction rows are not validated for every expected year
Location: sustainable.py, _frozen_predictions()

The function checks that the total number of predictions matches the number of labels, but it does not independently verify that every expected test year has at least one row.

If an unexpected empty year or invalid year structure reaches this function, the resulting output may be incomplete.

Recommendation: Validate that the test-year list is nonempty and that every expected test year has rows before calculating per-year metrics.
R6 — Medium Priority: _validate_results() does not verify the expected number of rows
Location: sustainable.py, _validate_results()

The function checks:
- All three strategies appear.
- All expected years appear.
- There are no duplicate strategy/year pairs.
- Metrics are finite.
These are useful checks. However, it does not explicitly verify that every strategy has one row for every test year.

For example, a malformed DataFrame could contain all three strategies and all test years while still missing one strategy/year combination.

Recommendation: Validate the complete Cartesian product of expected strategies and test years. Also check that the row count equals:
len(expected_strategies) * len(test_years)



R7 — Medium Priority: Results should be validated before being written to disk
Location: sustainable.py, run()

The code validates the results before writing the CSV, which is good:
_validate_results(results, split.test_years)results.to_csv(...)write_report(...)



However, write_report() performs additional validation and can still fail after the CSV has been written.

This can leave a CSV without its corresponding report.

Recommendation: Complete all report-input validation before writing either output, or use temporary files and finalize the outputs only after both writes succeed.
R8 — Medium Priority: Periodic retraining does not explicitly validate model predictions
Location: sustainable.py, _periodic_retraining()

The function predicts using:
predictions = model.predict(preprocessor.transform(X_current))



Unlike _frozen_predictions(), it does not explicitly verify that the prediction count matches the current year's labels.

The metrics helper may catch some mismatches, but an explicit check would make the contract consistent across strategies.

Recommendation: Validate prediction length before passing predictions to _score_predictions().
R9 — Medium Priority: Incremental updates should validate the prediction/update sequence
Location: sustainable.py, _incremental_update()

The ordering is correct:
predictions = model.predict(transformed)rows.append(_score_predictions(...))model.partial_fit(transformed, y_current)



This ensures the current year's labels are not used before that year's predictions are generated.

However, the code assumes the update succeeds and does not explicitly validate the prediction length or the current year's label values before updating.

Recommendation: Preserve the predict-before-update order. Add validation for prediction count and labels before calling partial_fit().
R10 — Low Priority: Mean metrics across years are not weighted by test-year size
Location: sustainable.py, write_report()

The summary uses:
results.groupby("strategy", as_index=False)[list(evaluation.METRICS)].mean()



This gives every test year equal weight, regardless of how many test rows that year contains.

That is not necessarily wrong. It represents average performance across years rather than overall performance across all test rows.

However, it should be clearly distinguished from a pooled metric across all test examples.

Recommendation: Keep the current equal-year average if that is the intended summary, but label it explicitly as the unweighted mean across test years. Consider including n_test_rows in the per-year table so readers can see the difference in year sizes.
R11 — Low Priority: The report's conclusion makes a recommendation that is not directly established by the results
Location: sustainable.py, write_report()

The report concludes:
For a production malware detector, periodic retraining is the safer default when labeled data can be collected in batches; incremental updates are useful when updates must be frequent and computationally inexpensive.

This is a reasonable general recommendation, but the code does not measure operational safety, update cost, label quality, or deployment reliability.

The experiment measures classification metrics, not the full production trade-offs.

Recommendation: Qualify this conclusion as a general consideration rather than a finding demonstrated by the experiment. If the project requires a results-based conclusion, derive it from the measured results.
R12 — Low Priority: The model name and hyperparameters are hardcoded
Location: sustainable.py, MODEL_NAME, ALPHA, and _sgd()

The experiment currently uses one classifier:
MODEL_NAME = "SGDClassifier"ALPHA = 1e-6



This is understandable because SGDClassifier supports partial_fit(), which is necessary for the incremental strategy.

However, the choice and hyperparameters are embedded directly in the module.

Recommendation: Keep the single-model comparison if required by the assignment, but document why this classifier was selected and confirm that its hyperparameters match the intended Step 4 baseline.
R13 — Low Priority: main.py usage documentation and help text should be checked for consistency
Location: main.py, module docstring and --all argument

The module docstring now includes:
python -m src.main --step 7


The parser also accepts Step 7, and the --all help text correctly says it runs Steps 1 through 7.

The integration looks correct. The only thing to check is that the project instructions permit --all to run every step sequentially, since later steps depend on artifacts generated by earlier steps.

Recommendation: Keep the Step 7 CLI integration. Confirm that the all-steps workflow is intended and that the required input files and output directories are available.

Additional important comparison issue
R14 — High Priority: The three strategies may not use equivalent model configurations
Location: sustainable.py, _sgd(), _frozen_predictions(), _periodic_retraining(), and _incremental_update()

The frozen baseline loads the SGD model selected and fitted by Step 4:
estimator = joblib.load(path)



However, periodic retraining and incremental adaptation create new classifiers using:
model = _sgd()



That function uses a hardcoded alpha=1e-6 and other fixed settings.

If Step 4 selected different hyperparameters for its SGD model, the three strategies are not comparing only different update approaches. They are also comparing potentially different model configurations.

Recommendation: Check the Step 4 SGD hyperparameters and make sure the periodic and incremental models use the same intended configuration. If the experiment deliberately uses different settings, document that limitation clearly.

Reviewer Priority Summary
Priority	Review items	Main concern
High	R1, R2, R14	Feature compatibility and fair model comparison
Medium	R3–R9	Preprocessing assumptions, temporal validation, completeness, and output reliability
Low	R10–R13	Reporting interpretation, configuration, and CLI documentation

What Should Remain Unchanged
These are good design decisions in the current implementation:
1. Three-strategy comparison: Keep frozen, periodic retraining, and incremental adaptation as separate strategies.
2. Frozen baseline: Use the Step 4 model without updating it.
3. Temporal ordering: Periodic retraining must only use data from years before the current test year.
4. Predict before updating: Incremental adaptation must score each year before using that year's labels.
5. Fixed incremental preprocessor: Keep it fixed if the goal is to isolate classifier adaptation from preprocessing changes.
6. SGDClassifier: It is an appropriate choice for this experiment because it supports partial_fit().
7. Per-year metrics: Keep accuracy, precision, recall, and F1 for every strategy and test year.
8. Mean performance summary: Keep the summary across years, but make clear that it is an unweighted mean.
9. Drift-aware interpretation: Do not claim that feature drift alone caused a performance change.
10. Separate Step 7 outputs: Keep the CSV and Markdown report.
11. Step 7 CLI integration: Keep run_step7() and the --step 7 option.
Reviewer Verdict
Revise before completion
Step 7 has a good experimental structure. The temporal sequencing is particularly important, and the code generally respects it.

The biggest issue I would resolve is R14. If the frozen baseline uses Step 4's tuned hyperparameters while the other strategies use a separately hardcoded configuration, the comparison may not be measuring what the report says it measures.

I would address R1, R2, and R14 first, then work through the medium-priority validation improvements. No major redesign is necessary.
Student Decision Log
Record whether you accept, reject, or modify each recommendation.
ID	Decision (Accept / Reject / Modify)	Notes
R1		   Accept    Valid
R2		   Accept    Valid
R3		   Accept    Valid
R4		   Accept    Valid
R5		   Accept    Valid
R6		   Accept    Valid
R7		   Accept    Valid
R8		   Accept    Valid
R9		   Accept    Valid
R10		   Accept    Valid
R11		   Accept    Valid
R12		   Accept    Valid
R13		   Accept    Valid
R14		   Accept    Valid


Final scope note: This review is based on the supplied sustainable.py and updated main.py. I have not executed the project or inspected the Step 7 assignment specification, so hyperparameter equivalence and required report contents should be checked against the supporting files and project instructions.

Send the revised files when you're ready, and I'll check each change against this review.