# Step 4 Test Log

## Test command

```text
pytest -q
```

## Final result

```text
.............................                                            [100%]
29 passed, 1 warning in 1.52s
```

The warning is an existing SciPy `RuntimeWarning` from the small synthetic
Step 2 KS comparison. It does not fail the suite.

## Step 4 tests written

`tests/test_step4_models.py` covers:

- The four configured estimators and their model-specific preprocessing
- Random-state configuration
- Forward-chaining fold acceptance
- Invalid, duplicate, out-of-range, overlapping, and misplanned fold rejection
- Single-class validation-fold rejection
- Sampled reproducibility checks for predictions and fitted state
- A small real `DummyClassifier` grid-search smoke test
- Finite search-score validation and summary-row creation
- Model-summary and candidate-table guard behavior
- Class-balance input validation

## Failure found and fixed

The first Step 4 run had one test expectation mismatch. The implementation
correctly reported that a validation half was “not exactly” the planned year,
while the test expected a different phrase. The test was aligned with the
source's error contract.

The full suite then exposed one stale Step 3 assertion: the new Step 4 model
configuration correctly places `DummyClassifier` in the unscaled artifact
group alongside `RandomForestClassifier`. The Step 3 test expectation was
updated. No `src` files were edited.
