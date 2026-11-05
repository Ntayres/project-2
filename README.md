# Project 2: Malware Classification Under Data Drift

This project classifies Android applications as malware or benign software from API-call
count features while measuring how those features and model performance change over time.
It implements data loading, drift analysis, leakage-safe preprocessing, model selection,
evaluation, year-by-year predictions, and three sustainable learning strategies. The
completed test suite contains 66 passing tests.

This repository was built with the Project 2 harness. See HARNESS.md for how the builder, reviewer, and tester agents were run.

## Agents used

| Role | Tool | Model |
| --- | --- | --- |
| Builder | Claude | Claude Opus |
| Reviewer | ChatGPT | Think |
| Tester | GitHub Copilot | Copilot Fusion |

## How to run

1. Install packages: `pip install -r requirements.txt`
2. Put the dataset files in `input_data/`.
3. Run Step 5 evaluation: `python -m src.main --step 5`.
   The Markdown report is written to `docs/step5_evaluation.md` and the CSV results are
   written to `results/step5_test_metrics.csv`. Run earlier steps first if the Step 4 model
   artifacts in `outputs/models/` are not already present.
4. Run Step 6 prediction analysis: `python -m src.main --step 6`.
   The per-year report is written to `docs/step6_test_results.md`, with CSV outputs in
   `results/step6_test_predictions_by_year.csv`, `results/step6_test_prediction_overall.csv`,
   and `results/step6_drift_comparison.csv`.
5. Run Step 7 sustainability comparison: `python -m src.main --step 7`.
   The report is written to `docs/step7_sustainable_approach.md`, and the results are written
   to `results/step7_sustainable_results.csv`.
6. Run all tests: `python harness.py test all`

## Results

Step 5 evaluates the saved classifiers on later held-out years using accuracy as the primary
metric and also reports precision, recall, and F1-score. See
`docs/step5_evaluation.md` and `results/step5_test_metrics.csv`.

## Hand-written code

No project source code was written or edited manually; the builder, reviewer, and tester
agents produced and validated the implementation. The student supplied project direction,
review decisions, and repository integration.
