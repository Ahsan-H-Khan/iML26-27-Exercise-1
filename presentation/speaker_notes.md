# CSC: development and evaluation

7 slides. Main target: 300 seconds (5.0 minutes); rehearse to set actual timing. Five-minute submission version; target timing is 300 seconds and requires rehearsal.

## 1. CSC: complexity after checking accuracy

Target: 40 seconds.

The task is to propose an interpretability metric and evaluate it. My measure summarizes tree-model structure. EPL counts average path decisions. EEPC extends paths over an ensemble. EA multiplies the mean path by a feature burden. The description-length idea motivates counting distinct rules and trees. CSC is the sum shown here; it is MDL-inspired rather than a formal code length or calibrated human score.

Source: results/metric_development_v2: candidates.csv, selections.csv, selected_test_scores.csv, metric_comparisons.csv and presentation_summary.json.

## 2. Prediction quality and the complexity trade-off

Target: 55 seconds.

We use Iris and Breast Cancer, with decision trees, random forests and gradient boosting. The original experiment has 72 validation candidates over three splits. All beat a prior-class dummy; 35 meet the two-percentage-point tolerance. Minimum CSC is selected within that competitive set. The table shows means on test sets for validation-frozen choices. On Iris both selectors agree. On Breast Cancer, lower complexity comes with lower test accuracy. Three overlapping splits make these descriptive means.

Source: results/metric_development_v2: candidates.csv, selections.csv, selected_test_scores.csv, metric_comparisons.csv and presentation_summary.json.

## 3. The threshold investigation

Target: 40 seconds.

The hypothetical budget concerns tests encountered by one prediction, not the CSC score. We flag models if more than half their validation predictions need more than seven distinct tests. In both datasets 21 of 36 candidates are flagged. CSC ranges overlap across that boundary, so one increasing CSC cutoff cannot classify all models correctly. Five- and nine-test sensitivity analyses also give different flags. These assumed budgets are not established human capacities.

Source: results/threshold_v1/pooled_thresholds.csv

## 4. Training examples and model complexity

Target: 45 seconds.

Task 2 changes training samples, features and class balance while holding validation examples fixed. We completed 756 fits across 63 unique conditions, retaining all three seeds and both datasets. This slide compares quarter-sized training samples with the original training set while holding each configuration fixed. Average CSC falls in both datasets, with small average accuracy losses. These averages summarize 36 configuration-and-seed comparisons per dataset; they are not independent replicates. Choosing different competitive models can reverse the CSC trend.

Source: results/metafeatures_v1/report.md

## 5. Feature removal can lose predictive information

Target: 45 seconds.

This Iris plot compares nested feature subsets, from one feature to all four. Solid lines show the chosen model; dashed lines show the best validation accuracy available. One-feature accuracy varies greatly across seeds because the retained feature differs. Relative to the original choice, selected accuracy changes are minus46.15, minus23.29, and zero percentage points. The Breast Cancer effects are smaller, but CSC still does not change monotonically. Feature availability changes information as well as feature count.

Source: results/metafeatures_v1/selected_endpoint_comparisons.csv

## 6. Class imbalance needs a size-matched comparison

Target: 45 seconds.

We retain only a quarter of the designated training class and compare with a stratified training subset of exactly the same total size. The results hold model configuration fixed and average the 36 configuration-and-seed pairs per dataset. Accuracy falls by about five points on Iris and3.6 on Breast Cancer. Mean CSC barely changes on Iris and decreases on Breast Cancer. Reduced complexity therefore does not imply improved prediction or easier human understanding.

Source: results/metafeatures_v1/imbalance_paired_comparisons.csv

## 7. What the project establishes

Target: 30 seconds.

CSC gives a reproducible structural comparison after checking predictive quality. The proposed budget does not establish a universal cutoff. Dataset metadata affects learned models, but direction depends on the data, seed and whether the model configuration is held fixed. Twelve of63 conditions lose the original accuracy reference. Human interpretability and generalization beyond these two datasets remain open. The long deck and reports contain the detailed definitions, plots and limitations.

Source: results/metafeatures_v1/report.md; results/threshold_v1/report.md
