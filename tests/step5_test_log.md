# Step 5 Test Log

## Review-fix validation

The accepted Step 5 review comments were implemented:

- Validate that every loaded Step 4 artifact is fitted, exposes `predict`, matches the
  current feature-name order, and returns one allowed prediction per test row.
- Reject empty, non-one-dimensional, or single-class evaluation inputs.
- Validate test labels and years against the validated Step 3 split.
- Select the report winner by sorting accuracy instead of relying on input row order.
- Create and validate custom output directories before writing.
- Validate the report before writing the CSV, so the two outputs are not intentionally
  written in different stages.
- Qualify the report's statement about artifact provenance.

The reviewer comment requesting test class counts (R8) was rejected in the decision log as
unneeded for the assignment's required output.

## Tests

Command:

```text
pytest -q tests/test_step5_evaluation.py tests/test_step4_models.py
```

Result:

```text
17 passed in 1.18s
```

The Step 5 command was also rerun against the project dataset:

```text
python -m src.main --step 5
```

It completed successfully and regenerated `docs/step5_evaluation.md` and
`results/step5_test_metrics.csv`. The held-out test set contains 27,000 rows, and
`RandomForestClassifier` remains the highest-accuracy classifier at `0.818741`.

## Additional evaluation tests

The evaluation tests were expanded to cover:

- Shape, prediction-length, and invalid-label rejection
- Missing and unfitted model artifacts
- Report validation for missing columns, duplicate classifiers, non-finite
  metrics, and out-of-range metrics

The full project suite was then run:

```text
pytest -q
.............................................                            [100%]
45 passed, 1 warning in 1.71s
```

The warning is an existing SciPy `RuntimeWarning` from the small synthetic
Step 2 KS comparison and does not fail the suite.

The only failure during this test expansion was in the test double for the
prediction-length case: it was not recognized by scikit-learn as fitted. The
test double was corrected to provide an instance-level fitted attribute and a
`fit()` method. No `src` files were edited.
