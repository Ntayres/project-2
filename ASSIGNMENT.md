**Project 2: Classifying malware data and finding a sustainable ML approach with agentic AI coding**

**Due Date:** Oct. 28 (Wed.) by 11:59 p.m.  
**Submission Format:** Individual  
**Required Approach:** Agentic AI coding with builder, reviewer, and tester agents

For this project, you will implement two (or more) machine learning classifiers using scikit-learn. These classifiers will be applied to an Android malware dataset in the folder input_data.

You must build this project with an agentic AI coding tool: an AI agent that plans the work, writes and edits files, runs code and tests, and fixes its own errors. Examples include Claude Code, OpenAI Codex, Cursor in agent mode, and GitHub Copilot in agent mode.

You will work with three separate agents, each with its own role: a builder agent that writes the code, a reviewer agent that reviews it, and a tester agent that writes and runs tests. Separating these roles means the code is checked by agents that do not share the builder’s assumptions, which is how agentic programming is done well in practice.

Your role is the lead engineer. You give the agents context, set the direction, decide which review comments to act on, check results, and make the final decisions. You will be graded on both the results and how well you directed the agents, so keep a complete record of your process from the first session to the last.

**Dataset**:

The data consists of malware and goodware from Android apps over an 8-year period. The Android malware dataset has a property of data drift. Data drift occurs when the data used to train a machine-learning model is no longer representative of the new data the model encounters.

Data drift refers to the change in the statistical properties of the data over time, which can cause a machine learning model’s predictions to become less accurate. It typically happens because:

1.  **Feature Drift**: The distribution of individual features changes. For example, if you trained a model to predict sales based on seasonality, a sudden change in consumer behavior might impact your model's performance.

2.  **Concept Drift**: The relationship between the input and output changes. For instance, a model predicting product demand might experience concept drift if the factors influencing demand evolve (like new trends or policies affecting consumer behavior).

Detecting and addressing data drift is crucial to maintaining model accuracy over time. Common methods for managing data drift include monitoring model performance, retraining models on recent data, and utilizing adaptive algorithms that can adjust to changes in data distribution.

The first column of each file should be discarded.

Except for the first column, the information pertains to the API calls of Android apps.

Each file has 1,500 rows and 1,849 columns.

Target class: binary (malware and goodware)

Labeling for target class: each file name includes the target class of each app in the file.

**Agentic AI Coding Requirements:**

1.  **Use three separate agents:** Run the builder, reviewer, and tester as separate agent sessions. The same tool and model may be used for all three (a different model is welcome but not required). Each must start with a fresh context: the reviewer and tester must never see the builder’s conversation. Name the tool and model for each role in your README.

2.  **Builder agent:** Writes and edits all project code. Use it for all coding work. If you write or edit any code by hand, say so in your process reflection.

3.  **Reviewer agent:** After each major step, give a fresh reviewer session only the project context file, PLAN.md, and the code changes (the git diff). It writes a review report in reviews/ (e.g., reviews/step1_review.md) listing problems such as bugs, data leakage between years, wrong metrics, and unclear code.

4.  **Tester agent:** Give a fresh tester session the assignment, the project context file, and PLAN.md, but not the builder’s explanations. It writes tests in tests/ based on what the code should do (for example, the first column is dropped, labels are extracted correctly from file names, each year has the expected number of rows and columns, and no test-year data is used in training), runs them, and saves the output.

5.  **You decide on the review:** For every review comment and failed test, record whether you accepted or rejected it and why, in a decision log at the end of each review report. Reviewer agents can be wrong, so think about each comment before accepting it.

6.  **Close the loop:** Send accepted comments and failing tests back to the builder to fix, then rerun the tests. Complete at least one full build → review → test → fix cycle for each of Steps 1–7 (small steps may be grouped together).

7.  **Set up before coding:** Create a git repository and a project context file (AGENTS.md, CLAUDE.md, or your tool’s equivalent). It should describe the dataset and folder layout, the project goal, how labels are encoded in file names, coding conventions, and how to run the code. All three agents use it.

8.  **Plan first:** Before any code is written, have the builder write a plan (PLAN.md) covering Steps 1–7 below, and have the reviewer agent review it (see Step 0). Revise it and keep both versions in the git history.

9.  **Commit by role:** Commit after each completed task, starting each message with the role that did the work, e.g., “\[builder\] Load data and extract labels”, “\[reviewer\] Step 1 review”, “\[tester\] Add data-loading tests”. Do not squash or rewrite history.

10. **Check the agents’ claims:** Do not accept reported results without checking them. Verify at least two results yourself (rerun the code, inspect the data, or run a sanity check) and record how you did it.

11. **Save every session:** Export the full transcript of every agent session. Do not edit transcripts except to remove API keys, passwords, or personal information.

**Tasks:**

Each step lists the Markdown files to generate for it. Steps 1–7 each have a builder report in docs/, a review report in reviews/, a test log in tests/, and session transcripts in transcripts/.

> **Step 0: Plan the Project**

1.  **Set Up the Project:** Create the git repository and the project context file (AGENTS.md, CLAUDE.md, or your tool’s equivalent).

2.  **Write the Plan:** Have the builder agent write PLAN.md covering Steps 1–7: the approach for each step, the files it will create, how the data will be split by year, and how each step will be tested.

3.  **Review the Plan:** Give a fresh reviewer session only the assignment, AGENTS.md, and PLAN.md. It checks for missing steps, data leakage between years, and anything that does not match the assignment, and writes reviews/step0_plan_review.md. Add your decision log at the end.

4.  **Revise and Commit:** Have the builder update PLAN.md based on the comments you accepted. Commit both the original and the revised plan before starting Step 1.

> **Markdown files for Step 0:**

- AGENTS.md **(you and the builder):** the project context file used by all three agents.

- PLAN.md **(builder):** the plan for Steps 1–7, revised after the review.

- reviews/step0_plan_review.md **(reviewer):** review comments on the plan, ending with your decision log (accepted or rejected, and why, for every comment).

- transcripts/ **(builder and reviewer):** one transcript for each session, e.g., 01_2026-10-19_builder_step0.md and 02_2026-10-19_reviewer_step0.md.

> **Step 1: Prepare the Data**

1.  **Load the Data:** Load the files from the input_data folder. Each file represents an app with its respective features and target label.

2.  **Discard the First Column:** As specified, drop the first column from each file.

3.  **Extract Labels:** Extract the label (malware/goodware) from the filename and add it as a target variable.

4.  **Combine Files per year:** Concatenate all files into a single dataset, with each row representing an app and each column (excluding the target) representing API call information.

> **Markdown files for Step 1:**

- docs/step1_data_summary.md **(builder):** files loaded per year, rows and columns after dropping the first column, and malware/goodware counts per year.

- reviews/step1_review.md **(reviewer):** review comments on this step’s code changes, ending with your decision log (accepted or rejected, and why, for every comment).

- tests/step1_test_log.md **(tester):** the tests written for this step, the test output, and any failures and how they were fixed.

- transcripts/ **(all roles):** one transcript for each builder, reviewer, and tester session, with the role and step in the file name (e.g., 05_2026-10-20_reviewer_step1.md).

> **Step 2: Handle Data Drift**
>
> Since this dataset spans 8 years, it likely has **data drift** (feature drift or concept drift). Here are some suggestions:

- **Split by Year:** Train and test on different years to observe how the models handle data drift.

- **Monitoring Drift:** Consider using statistical methods (e.g., Kolmogorov-Smirnov test) to measure the feature distribution changes over the years.

> **Markdown files for Step 2:**

- docs/step2_drift_analysis.md **(builder):** how drift was measured (e.g., Kolmogorov-Smirnov test), the features that drift most, and which years differ the most.

- reviews/step2_review.md **(reviewer):** review comments on this step’s code changes, ending with your decision log (accepted or rejected, and why, for every comment).

- tests/step2_test_log.md **(tester):** the tests written for this step, the test output, and any failures and how they were fixed.

- transcripts/ **(all roles):** one transcript for each builder, reviewer, and tester session, with the role and step in the file name (e.g., 05_2026-10-20_reviewer_step2.md).

> **Step 3: Preprocess Data for Model Training**

1.  **Normalize/Scale Features:** Scale the features to standardize the data, which is beneficial for many machine learning models.

2.  **Split Data:** Divide the data into training and testing sets. Optionally, use a time-based split (e.g., first few years for training, later years for testing) to simulate real-world drift.

> **Markdown files for Step 3:**

- docs/step3_preprocessing.md **(builder):** the scaling method, the train/test split (which years are used for training and testing), and how leakage between years is prevented.

- reviews/step3_review.md **(reviewer):** review comments on this step’s code changes, ending with your decision log (accepted or rejected, and why, for every comment).

- tests/step3_test_log.md **(tester):** the tests written for this step, the test output, and any failures and how they were fixed.

- transcripts/ **(all roles):** one transcript for each builder, reviewer, and tester session, with the role and step in the file name (e.g., 05_2026-10-20_reviewer_step3.md).

> **Step 4: Apply Machine Learning Classifiers**

1.  **Select Classifiers:** Implement at least two classifiers (e.g., Logistic Regression, Random Forest, SVM, or Gradient Boosting, etc.) from scikit-learn.

2.  **Train Models:** Train the classifiers on the training dataset.

> **Markdown files for Step 4:**

- docs/step4_models.md **(builder):** the classifiers chosen, why they were chosen, and their hyperparameters.

- reviews/step4_review.md **(reviewer):** review comments on this step’s code changes, ending with your decision log (accepted or rejected, and why, for every comment).

- tests/step4_test_log.md **(tester):** the tests written for this step, the test output, and any failures and how they were fixed.

- transcripts/ **(all roles):** one transcript for each builder, reviewer, and tester session, with the role and step in the file name (e.g., 05_2026-10-20_reviewer_step4.md).

> **Step 5: Evaluate Classifier Performance**

1.  **Accuracy Comparison:** Use accuracy as your primary metric to evaluate each classifier on the testing data.

2.  **Additional Metrics:** Consider using F1-score, precision, and recall, as malware classification is often an imbalanced problem.

> **Markdown files for Step 5:**

- docs/step5_evaluation.md **(builder):** a table of accuracy, precision, recall, and F1-score for each classifier.

- reviews/step5_review.md **(reviewer):** review comments on this step’s code changes, ending with your decision log (accepted or rejected, and why, for every comment).

- tests/step5_test_log.md **(tester):** the tests written for this step, the test output, and any failures and how they were fixed.

- transcripts/ **(all roles):** one transcript for each builder, reviewer, and tester session, with the role and step in the file name (e.g., 05_2026-10-20_reviewer_step5.md).

> **Step 6: Test Data Prediction**

1.  **Test Each Classifier:** Predict on the test data with each trained classifier.

2.  **Analyze Performance on Drifted Data:** If possible, compare the performance of classifiers on subsets of data from different years.

> **Markdown files for Step 6:**

- docs/step6_test_results.md **(builder):** test results for each classifier, broken down by year, and how performance changes as the data drifts.

- reviews/step6_review.md **(reviewer):** review comments on this step’s code changes, ending with your decision log (accepted or rejected, and why, for every comment).

- tests/step6_test_log.md **(tester):** the tests written for this step, the test output, and any failures and how they were fixed.

- transcripts/ **(all roles):** one transcript for each builder, reviewer, and tester session, with the role and step in the file name (e.g., 05_2026-10-20_reviewer_step6.md).

> **Step 7: Sustainable ML Approach**

Since you are looking for a sustainable ML approach, you might consider:

- **Periodic Retraining:** Periodically retrain models as new data becomes available.

- **Adaptive Algorithms:** Explore incremental training for classifiers to adapt to changes in data distribution over time.

> **Markdown files for Step 7:**

- docs/step7_sustainable_approach.md **(builder):** the retraining or incremental-learning approach tried, its results compared with a model that is never updated, and conclusions that feed your one-page report.

- reviews/step7_review.md **(reviewer):** review comments on this step’s code changes, ending with your decision log (accepted or rejected, and why, for every comment).

- tests/step7_test_log.md **(tester):** the tests written for this step, the test output, and any failures and how they were fixed.

- transcripts/ **(all roles):** one transcript for each builder, reviewer, and tester session, with the role and step in the file name (e.g., 05_2026-10-20_reviewer_step7.md).

> **Submission Guidelines**

Submit a link to your git repository (or a zip file that includes the .git folder) containing the following:

1.  **Code:** Your code as .py files or a Jupyter Notebook, well documented with comments that explain each major step. Include a README.md that names the tool and model used for each agent role and explains how to run the code.

2.  **Git history:** The full commit history, with role-tagged commit messages.

3.  **Project context file:** The final version of your AGENTS.md, CLAUDE.md, or equivalent.

4.  **Plan:** PLAN.md, showing the plan the builder wrote and the changes made after the Step 0 review.

5.  **Step reports:** The seven builder reports in docs/ (step1_data_summary.md through step7_sustainable_approach.md).

6.  **Agent session transcripts:** Full, unedited exports of every session, in a transcripts/ folder, named in date order with the role (e.g., 05_2026-10-20_reviewer_step1.md).

7.  **Review reports:** All eight reports in reviews/ (step0_plan_review.md, then step1_review.md through step7_review.md), each ending with your decision log (accepted or rejected, and why, for every comment).

8.  **Tests and test output:** The tests written by the tester agent and the seven test logs in tests/ (step1_test_log.md through step7_test_log.md), including runs that failed before they were fixed.

9.  **Results:** Tables or figures of accuracy, precision, recall, and F1-score for each classifier, including performance across years.

10. **Report (one page):** A report that explains your approaches to building a sustainable malware detection system.

11. **Process reflection (1–2 pages):** Answer the questions below. Each answer must point to the transcript file or commit it refers to.

    - What did the reviewer or tester catch that the builder missed (at least two cases)? How was it fixed?

    - Describe one review comment you rejected, and why.

    - Show one prompt that did not work and how you rewrote it.

    - Describe one other decision where you overrode an agent, and why.

    - How did you verify the agents’ results?

    - Which parts, if any, did you write or edit by hand?

12. **Screen recording (optional):** A 5–10 minute recording of one build → review → test cycle.

> **Live Check-in**

After the deadline, some students will be chosen at random for a 10-minute check-in. You will use your agent on your own code to make a small change chosen by the instructor, such as adding a classifier or a drift metric, and explain how your code works.

> **Grading**

- **Code, results, and report:** 60%

- **Agentic process (context file, plan, git history, transcripts, reviews, tests, reflection):** 40%

Your artifacts should tell one consistent story: the transcripts, commits, reflection, and code must agree with each other. Missing transcripts, review reports, or git history will significantly lower your grade.
