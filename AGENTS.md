# AGENTS.md: Project 2 context for all agents

Every agent (builder, reviewer, tester) reads this file first. Fill in the [TODO] parts before Step 0. Keep it short and specific.

## Project goal

Classify Android apps as malware or goodware from their API-call features, and find a sustainable approach that stays accurate as the data drifts over 8 years. The full assignment is in ASSIGNMENT.md.

## Dataset

- Location: input_data/ (not committed to git).
- [TODO: list the file names, or describe the naming pattern, and say how the year and the class appear in each file name]
- Each file has 1,500 rows and 1,849 columns. The first column must be dropped; the other 1,848 columns are API-call features.
- Label: [TODO: how the label is encoded in the data, e.g., 1 = malware, 0 = goodware]
- [TODO: anything else you learn about the data, e.g., feature value types]

## Folder layout

- src/: project code (builder only).
- tests/: pytest tests test_stepN_*.py and test logs stepN_test_log.md (tester only; the student adds failure decisions).
- docs/: one builder report per step.
- reviews/: one review report per step (reviewer writes comments; student writes the decision log).
- transcripts/: saved agent sessions (saved by the harness; agents never edit these).
- PLAN.md: the plan for Steps 1-7. REPORT.md and REFLECTION.md: written by the student.

## Coding conventions

- Python 3, pandas, numpy, scikit-learn, scipy. No other libraries without asking.
- [TODO: your conventions, e.g., functions with docstrings, a fixed random_state of 42, no hard-coded absolute paths]
- Train/test split is by year. Anything fit to data (scalers, feature selection, models) is fit on training years only.
- Malware is the positive class for precision, recall, and F1.

## How to run

- [TODO: the command(s) that run the whole project, e.g., python -m src.main]
- Tests: python harness.py test N (one step) or python harness.py test all.

## Commit rules

Every commit message starts with a role tag: [builder], [reviewer], [tester], or [student]. A git hook rejects other messages. Make small commits, one per task. Never rewrite or squash history.
