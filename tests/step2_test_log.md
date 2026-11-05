# Step 2 Test Log

## Test command

```text
pytest -q
```

## Result

```text
.........                                                                [100%]
9 passed, 1 warning in 0.99s
```

The warning is emitted by SciPy during a tiny synthetic KS comparison:

```text
RuntimeWarning: divide by zero encountered in divide
```

It does not fail the tests and does not change the returned assertions.

## Tests written

- Count binning and Jensen-Shannon distance bounds
- Synthetic feature drift detection
- Input dataframe immutability
- BH-adjusted p-values and p-value bounds
- Pair summaries and symmetric matrices
- Duplicate and missing pair validation
- Top-feature ranking
- Class-cancellation analysis
- Feature-vector deduplication versus `apkname` uniqueness
- Per-year class statistics
- Small-block and negative-count safeguards in the year classifier
- Four-year train/test split validation
- API signature shortening

## Initial failure and resolution

The first synthetic drift test expected the shifted feature's prevalence change
to be zero, but the fixture changed the feature from all zero values to all
positive values. The test data was corrected so the feature's count distribution
shifts while its nonzero prevalence remains constant. No source file was edited.
