"""Project 2 harness: shared settings (steps, roles, file names)."""
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

ROLES = ("builder", "reviewer", "tester")
COMMIT_TAGS = ("builder", "reviewer", "tester", "student", "harness")

STEP_TITLES = {
    0: "Plan the Project",
    1: "Prepare the Data",
    2: "Handle Data Drift",
    3: "Preprocess Data for Model Training",
    4: "Apply Machine Learning Classifiers",
    5: "Evaluate Classifier Performance",
    6: "Test Data Prediction",
    7: "Sustainable ML Approach",
}

DOC_REPORTS = {
    1: "docs/step1_data_summary.md",
    2: "docs/step2_drift_analysis.md",
    3: "docs/step3_preprocessing.md",
    4: "docs/step4_models.md",
    5: "docs/step5_evaluation.md",
    6: "docs/step6_test_results.md",
    7: "docs/step7_sustainable_approach.md",
}

TOP_LEVEL_FILES = ["README.md", "AGENTS.md", "PLAN.md", "REPORT.md", "REFLECTION.md"]

# Directories left out of the diff the reviewer sees (builder explanations, other reviews, transcripts).
DIFF_EXCLUDES = ["transcripts", "reviews", "docs", ".harness", "tests",
                 "AGENTS.md", "CLAUDE.md", "PLAN.md", "README.md", "REPORT.md", "REFLECTION.md"]

TRANSCRIPT_RE = re.compile(
    r"^(?P<num>\d{2,3})_(?P<date>\d{4}-\d{2}-\d{2})_(?P<role>builder|reviewer|tester)_step(?P<step>[0-7])\.(?P<ext>md|jsonl|txt)$"
)

TODO_RE = re.compile(r"\[TODO[^\]]*\]")

REVIEWER_FOCUS = {
    0: [
        "Does PLAN.md cover every task in Steps 1-7 of ASSIGNMENT.md?",
        "Is the train/test split defined by year, with later years used for testing?",
        "Are the scaler and any feature selection fit on training years only?",
        "Are metrics computed on held-out (validation or test) data, not on training data?",
        "Does each step say which files it creates and how it will be tested?",
    ],
    1: [
        "Is the first column dropped from every file before anything else uses the data?",
        "Are labels parsed from file names correctly for both malware and goodware files?",
        "Are all files in input_data/ loaded, and grouped by year correctly?",
        "Do row and column counts match the assignment (1,500 rows, 1,849 columns before the drop)?",
    ],
    2: [
        "Is the drift test (e.g., Kolmogorov-Smirnov) applied per feature between years, correctly?",
        "Are constant or all-zero features handled so the test does not fail or mislead?",
        "Does the code report which features and years drift most?",
    ],
    3: [
        "Is the scaler fit on training years only, then applied to test years?",
        "Do training and test sets share no years (no leakage)?",
        "Are the same feature columns, in the same order, used for train and test?",
    ],
    4: [
        "Are at least two scikit-learn classifiers implemented?",
        "Is random_state set so results are reproducible?",
        "Are hyperparameters and class imbalance handled sensibly?",
    ],
    5: [
        "Are accuracy, precision, recall, and F1 computed on held-out data?",
        "Is malware treated as the positive class consistently?",
        "Are the metric calls (averaging, labels) correct?",
    ],
    6: [
        "Are results reported per test year and per classifier?",
        "Is any model trained on a year before that year is predicted?",
    ],
    7: [
        "Is evaluation time-forward (train on years up to t, test on year t+1)?",
        "Is there a never-updated baseline model for comparison?",
        "Is the comparison fair (same test years, same metrics)?",
    ],
}

TESTER_FOCUS = {
    1: [
        "The first column is not present after loading.",
        "Labels contain only the two classes and match each file name.",
        "Each loaded file has 1,500 rows; combined data per year has the expected size.",
        "There are no missing values after loading.",
    ],
    2: [
        "The drift statistic is between 0 and 1.",
        "Identical distributions produce no drift; a clearly shifted synthetic feature is detected.",
    ],
    3: [
        "The scaler is fit on training years only (training columns have mean near 0; test data is transformed, not refit).",
        "Training and test sets share no years.",
        "Train and test have identical columns in the same order.",
    ],
    4: [
        "Each classifier fits and predicts on a small synthetic dataset.",
        "Predictions use only the two class labels.",
        "Results are identical across two runs with the same random_state.",
    ],
    5: [
        "Metric values match scikit-learn on a small hand-checked example.",
        "Evaluation uses the held-out split, not the training split.",
    ],
    6: [
        "Results include one entry per test year per classifier.",
        "All metric values are between 0 and 1.",
    ],
    7: [
        "No model is ever trained on a year at or after the year it is tested on.",
        "The updated model actually uses the newer data (retraining or partial_fit).",
        "The baseline model is never updated.",
    ],
}


def review_path(step: int) -> str:
    return "reviews/step0_plan_review.md" if step == 0 else f"reviews/step{step}_review.md"


def test_log_path(step: int) -> str:
    return f"tests/step{step}_test_log.md"


def roles_for_step(step: int):
    return ("builder", "reviewer") if step == 0 else ROLES
