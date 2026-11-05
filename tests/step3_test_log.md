# Step 3 Test Log

## Test command

```text
pytest -q
```

## Final result

```text
..................                                                       [100%]
============================== warnings summary ===============================
tests/test_step2_drift.py::test_pair_summary_and_symmetric_matrix_require_complete_unique_pairs
  RuntimeWarning: divide by zero encountered in divide

18 passed, 1 warning in 1.61s
```

The warning is an existing SciPy warning from the small synthetic Step 2 KS
comparison. It does not fail the suite. No `src` files were edited.

## Step 3 tests written

`tests/test_step3_preprocessing.py` covers:

- Year-based train/test splitting and row preservation
- Rejection of overlapping, missing, or incorrectly ordered years
- Removal of metadata from feature matrices
- Feature-count and feature-order contracts
- Rejection of negative, non-numeric, and non-finite features
- Training-only constant-feature removal
- Training-only scaler statistics
- Scaled and unscaled preprocessing configurations
- Model-specific scaler decisions
- Forward-chaining validation folds and fold indices
- Split summaries and report-input validation
- Loud failure when every training feature is constant

## Failure found and fixed

The first Step 3 run had one test assertion using the wrong expected error
phrase for an invalid label. The source correctly reports `label value(s)`;
the test expectation was updated to match the actual error contract. A pandas
fixture warning was also removed by copying the DataFrame before mutation.
