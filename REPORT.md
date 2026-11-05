# Report: A Sustainable Malware Detection System (one page)

## Problem

The Android API-call distributions change over time, so a detector trained on older apps can become stale. Mean KS drift increased from 0.014 between 2014-2015 to 0.052 between 2014-2021, and 199 features had KS >= 0.2 across that long gap. Years were also increasingly distinguishable from their API features, confirming substantial temporal drift.

## What we tried

The system uses malware as the positive class and evaluates an SGDClassifier on later years using a prequential order. The frozen baseline trains on 2014-2017 and is never updated. Periodic retraining refits the model and scaler on all labeled years before each test year. Incremental learning uses `partial_fit` after prediction, once labels for that year become available; its scaler remains fixed on the original training years.

## Results

Across 2018-2021, periodic retraining had the best unweighted mean accuracy (0.8152) and F1 (0.8508). Incremental learning reached 0.7878 accuracy and 0.8031 F1. The frozen baseline fell to 0.6132 accuracy and 0.7014 F1, with accuracy dropping to 0.3943 in 2021. Periodic retraining reached 0.8823 accuracy in 2021, while incremental learning reached 0.7063.

## Recommendation

Deploy periodic expanding-window retraining when labels can be collected in yearly or regular batches. It gives the strongest and most stable late-year performance in this experiment, while incremental updates are a lower-cost fallback when full retraining is impractical. Test each new period before using its labels, then monitor accuracy, recall, F1, class balance, feature drift, and prediction distributions; trigger investigation or retraining when these measures regress.
