# Step 1 Test Log

## Test command

```text
pytest -vv tests\test_step1_data_pipeline.py
```

## Final result

```text
============================= test session starts =============================
platform win32 -- Python 3.11.9, pytest-9.2.0
collecting ... collected 6 items

tests/test_step1_data_pipeline.py::test_parse_filename_and_rejects_unknown_name PASSED
tests/test_step1_data_pipeline.py::test_load_file_keeps_identifier_as_metadata_and_labels_from_filename PASSED
tests/test_step1_data_pipeline.py::test_build_dataset_requires_both_classes_and_matching_columns PASSED
tests/test_step1_data_pipeline.py::test_loader_rejects_missing_negative_float_and_overflow_features PASSED
tests/test_step1_data_pipeline.py::test_duplicate_identifier_and_vector_counts_are_separate PASSED
tests/test_step1_data_pipeline.py::test_counts_per_year_and_step1_entry_point PASSED

============================== 6 passed in 1.08s ==============================
```

## Failures found and fixes

1. Initial collection failed because the standalone test workspace was not on
   Python's import path. Added `pytest.ini` with `pythonpath = .`.
2. The invalid-data fixtures used class names that intentionally violated the
   filename regex, so filename validation ran before the intended data-quality
   checks. Reused a valid `benign` filename for those fixtures.
3. The duplicate-vector fixture contained three identical vectors in one file.
   The implementation correctly reported two duplicate rows beyond the first;
   the assertion was corrected from 1 to 2.
4. `src/main.py` unpacked two values from `duplicate_summary`, which returns
   `(per_file, cross, totals)`. Updated `run_step1` to unpack and pass `totals`
   to `write_report`.
5. Report generation requires pandas' optional `tabulate` dependency for
   `DataFrame.to_markdown`. Installed `tabulate` in the test environment.
