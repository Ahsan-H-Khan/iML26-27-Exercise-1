# CSC: development and evaluation

21 main slides plus 2 optional appendix slides. Main target: 1440 seconds (24.0 minutes); rehearse to set actual timing. This longer working deck follows the user's instruction; the original course's five-minute submission requirement has not been changed.

## 1. Developing a complexity metric for tree models

Target: 55 seconds.

The project begins with two separate questions. Does a trained model predict accurately enough to be worth considering, and how much structure does it contain? Accuracy answers the first question. A complexity measure is intended to help with the second. I will walk through path length, its extension to ensembles, and an adjustment for features. I then use the description-length idea to motivate CSC, Cognitive-Structural Complexity. Finally, we evaluate the formula on Iris and Breast Cancer. This is a proposed structural measure. We do not yet have human-subject evidence that a lower score makes a model easier to understand. The empirical evaluation therefore focuses on competitive predictive performance and comparison with existing structural counts.

Source: results/metric_development_v2: candidates.csv, selections.csv, selected_test_scores.csv, metric_comparisons.csv and presentation_summary.json.

## 2. Start with Expected Path Length

Target: 70 seconds.

EPL measures the average number of internal decisions encountered by a reference example in one tree. It is not maximum depth, and it excludes the final leaf. In this teaching example, half the reference examples finish after one decision. A quarter finish after two decisions at each of the other two leaves. The expected path length is therefore one and a half decisions. The weights come from the reference examples reaching each leaf, not from assuming all leaves are equally likely. This answers a useful question about a typical prediction. Its limitation is that one path does not describe all the different rules stored in a model, and a single-tree definition needs to be extended before it describes an ensemble.

Source: Project definition of empirical EPL; implementation checked against scikit-learn decision_path: https://scikit-learn.org/stable/modules/generated/sklearn.tree.DecisionTreeClassifier.html. Teaching example, not experimental data.

## 3. Extend path length to the ensemble

Target: 70 seconds.

Expected Ensemble Path Complexity extends the same measurement across the actual prediction trees. We report two explicitly named versions. EEPC total is the sum of the tree EPL values; it measures all split visits for an ensemble prediction. EEPC mean divides that sum by the actual number of trees. Continuing the teaching example, one tree has EPL 1.5 and a second has EPL 2.0. Total is 3.5, while the mean is 1.75. This distinction matters: total grows when additional trees repeat work, whereas the mean can hide the number of trees. Neither version tells us how much of that work repeats identical logic. In multiclass boosting, we count every class-specific prediction tree, not just the number of boosting stages.

Source: Project definitions; total and mean conventions locked in protocols/metric_development_v2.md. Teaching example.

## 4. Ensemble-adjusted complexity (EA)

Target: 70 seconds.

The next proposed step is ensemble-adjusted complexity, EA. We multiply the path term by a logarithmic feature burden. For this revision, the declared convention uses average path length per tree, and F is the number of distinct input features actually used in internal splits. We use log base two of one plus F; unused dataset columns do not add burden. With average path length 1.75 and two features, the feature term is about 1.585, giving EA about 2.77. The total-path version is saved separately as a sensitivity measure. This adjustment recognizes that tracking several inputs may matter, but it is a design assumption, not a calibrated human cost. Counting features still misses how many different thresholds are used on each feature.

Source: Project EA definition and locked aggregation conventions. Teaching example.

## 5. A feature count can miss different rules

Target: 65 seconds.

Consider two questions about the same feature: whether a measurement is at most two, and whether it is at most five. They use one input feature, but they are different split tests. EA's feature count cannot distinguish that difference by itself. Conversely, if exactly the same rule appears several times, storing or describing it repeatedly need not add distinct logic. The same distinction applies to whole trees, although a tree signature must also account for its predictions and output channel. These observations motivate counting distinct split rules and distinct trees. They do not prove a particular logarithm or weight is optimal. The examples on this slide are explanations of the design, not results from the earlier controlled study.

Source: Teaching illustration; CSC identities are defined in iml_interpretability/tree_utils.py.

## 6. The Minimum Description Length idea

Target: 75 seconds.

Minimum Description Length, or MDL, is an established model-selection principle. In the introductory two-part formulation, we describe the model and then describe the data using the model. We seek a short combined description. A more complex model can fit better but costs more to describe; a very simple model can leave much of the data unexplained. This is useful motivation for thinking about reusable structure. However, formal MDL requires specified codes. CSC does not encode topology, thresholds, leaf predictions and residual data into a valid bit stream. Our logarithmic counts borrow the description-length idea, but they are not a formal MDL score. We therefore do not claim to benchmark CSC against implemented MDL or to derive CSC mathematically from that principle.

Source: Peter Grünwald, A Tutorial Introduction to the Minimum Description Length Principle, sections 1.3–1.4: https://homepages.cwi.nl/~pdg/ftp/mdlintro.pdf

## 7. Combine the ideas in CSC

Target: 80 seconds.

The proposed formula keeps mean path length as its first component. It adds log2 of one plus the number of distinct split tests, S, and log2 of the number of distinct full trees, U. S counts rules across the full fitted model, not only the paths visited in the reference data. Tree identities include outputs and output channels. Using an average in the path term separates per-tree path length from the ensemble term. This is an additive design that replaces EA's multiplicative feature burden; it is not an algebraic derivation. For the real Iris depth-two tree selected in seed 42, the validation mean path is 1.6579, there are 2 distinct tests and 1 tree. The resulting CSC is 3.2429. All three weights are one by choice, not human calibration.

Source: results/metric_development_v2: candidates.csv, selections.csv, selected_test_scores.csv, metric_comparisons.csv and presentation_summary.json.

## 8. Evaluate on trained models

Target: 75 seconds.

We use two bundled classification datasets: Iris has 150 examples, four features and three classes; Breast Cancer has 569 examples, thirty features and two classes. The same twelve configurations per group cover decision trees, random forests and gradient boosting. Three seeds create shared stratified train, validation and test partitions for every candidate within each group. Training uses approximately half the data, validation one quarter, and test one quarter. No learned preprocessing is needed in this pipeline. A prior dummy classifier is fitted on the training split. A candidate is competitive only if validation balanced accuracy beats the dummy and is within two percentage points of the group best. We freeze the minimum-CSC and maximum-accuracy choices on validation, then evaluate them on test without refitting.

Source: results/metric_development_v2: candidates.csv, selections.csv, selected_test_scores.csv, metric_comparisons.csv and presentation_summary.json.

## 9. First, check predictive performance

Target: 70 seconds.

Each point is one validation candidate. Balanced accuracy means the average recall across classes, so predicting only the majority class produces one-third for Iris and one-half for Breast Cancer. All 72 candidates beat those simple baselines. That alone is a low bar: the Iris one-split trees reach only about two-thirds. We therefore also apply the within-two-percentage-point rule separately in each dataset and seed. 35 of 72 candidates pass both conditions: fifteen for Iris and twenty for Breast Cancer. They are competitive for this benchmark, not certified adequate for a real-world application. The spread includes different settings and random splits; these points should not be treated as independent repeated measurements of one model.

Source: results/metric_development_v2: candidates.csv, selections.csv, selected_test_scores.csv, metric_comparisons.csv and presentation_summary.json.

## 10. Then compare accuracy and CSC

Target: 75 seconds.

Now we apply CSC to the candidates. Faint points are outside the competitive set, solid points are inside it, and black outlines identify the minimum-CSC choice for each seed. Both panels use the same scales. The useful region is high predictive accuracy with lower measured complexity, but the eligibility threshold is set within each seed rather than as one line across the whole panel. On Iris, the accuracy-only and CSC selectors choose the same models in all three runs. On Breast Cancer, CSC chooses single trees instead of the accuracy-only ensemble choices. The validation accuracy costs are 0.76, 1.12, 1.87 percentage points. This is the declared trade-off, not evidence that lower complexity improves predictions or human understanding.

Source: results/metric_development_v2: candidates.csv, selections.csv, selected_test_scores.csv, metric_comparisons.csv and presentation_summary.json.

## 11. Does CSC rank models differently?

Target: 75 seconds.

We compare rankings only among competitive models, separately within each dataset and seed. The table shows Spearman rank correlation with CSC. One means exactly the same order. CSC agrees closely with node count in all six groups; it also generally agrees with total ensemble path length. The EA ranking differs more in the groups with several eligible models. This is informative about which structural dimensions the formulas emphasize, but it is not evidence that CSC's order is the correct one for people. Iris runs two and three each contain only two eligible models, so perfect rank correlations there are especially weak evidence. The report also retains mean path length and EA based on total visits as sensitivities. We make no significance claim from the overlapping splits.

Source: results/metric_development_v2: candidates.csv, selections.csv, selected_test_scores.csv, metric_comparisons.csv and presentation_summary.json.

## 12. What do the test results show?

Target: 80 seconds.

These test results concern only choices frozen using validation data. On Iris, both selectors chose the same models and therefore have identical results. On Breast Cancer, the lower-CSC choices have a lower mean complexity but also lower mean test balanced accuracy than the accuracy-only choices. This is a real trade-off in this evaluation, not a free improvement. The displayed results are means across three overlapping splits; detailed sample standard deviations and individual rows are in the report. A two-point tolerance on validation does not guarantee that tolerance on each test split. The data were already inspected in prior project work, so this is descriptive follow-up rather than a new confirmatory experiment. We have developed a reproducible structural score and evaluated it on competitive models, but have not validated human interpretability or superiority over simpler metrics.

Source: results/metric_development_v2: candidates.csv, selections.csv, selected_test_scores.csv, metric_comparisons.csv and presentation_summary.json.

## 13. The seven-test budget does not give one CSC cutoff

Target: 65 seconds.

We count distinct tests encountered during a prediction, across every tree. The hypothetical rule flags a model if more than half of validation predictions encounter more than seven tests. Twenty-one of 36 configurations are flagged in each dataset, pooled across three seeds. Yet flagged and unflagged models have overlapping CSC scores in both datasets. A single increasing CSC cutoff therefore cannot reproduce this rule on the observed grid. Five and nine tests are sensitivity choices, not validated human capacities. Iris separates only at five tests; Breast Cancer separates at none of these three budgets.

Source: results/threshold_v1/pooled_thresholds.csv and report.md

## 14. Dataset properties and learned complexity

Target: 65 seconds.

We changed training sample count, feature availability, and the balance between classes. Each experiment uses the same validation rows within a seed. Test rows were not used for new model selection or scoring. Training samples and features use nested subsets. Imbalance has a stratified control with the same total number of training examples. We fit 12 original configurations in each unique condition, retaining all three seeds and both datasets. Identical conditions are shared. All 72 original validation configurations reproduce the previous results.

Source: protocols/remaining_tasks_v1.md; results/metafeatures_v1/condition_manifest.json

## 15. Iris: training sample count

Target: 65 seconds.

The solid curves show the selected model's validation accuracy and CSC. Dashed accuracy curves show the best candidate in each condition. Colors represent seeds. At 25% availability versus the original condition, selected accuracy changes range from 0.00 to 0.00 percentage points. CSC changes range from -4.09 to 5.47. Model changes: seed 42: dt_d2 to gb_n10_d1; seed 1337: rf_n5_d2 to rf_n25_d2; seed 2026: rf_n5_d2 to dt_d2. Fixed-configuration plots and component changes are in the detailed report. Feature removal also changes which information is available. These are exploratory within-dataset comparisons.

Source: results/metafeatures_v1/selected_model_summary.csv and selected_endpoint_comparisons.csv

## 16. Breast Cancer: training sample count

Target: 65 seconds.

The solid curves show the selected model's validation accuracy and CSC. Dashed accuracy curves show the best candidate in each condition. Colors represent seeds. At 25% availability versus the original condition, selected accuracy changes range from -1.14 to 1.51 percentage points. CSC changes range from -2.00 to 3.21. Model changes: seed 42: dt_d2 to dt_d1; seed 1337: dt_d2 to rf_n5_d2; seed 2026: dt_d4 to gb_n10_d1. Fixed-configuration plots and component changes are in the detailed report. Feature removal also changes which information is available. These are exploratory within-dataset comparisons.

Source: results/metafeatures_v1/selected_model_summary.csv and selected_endpoint_comparisons.csv

## 17. Iris: available features

Target: 65 seconds.

The solid curves show the selected model's validation accuracy and CSC. Dashed accuracy curves show the best candidate in each condition. Colors represent seeds. At 25% availability versus the original condition, selected accuracy changes range from -46.15 to 0.00 percentage points. CSC changes range from -1.24 to 3.02. Model changes: seed 42: dt_d2 to dt_d1; seed 1337: rf_n5_d2 to dt_d4; seed 2026: rf_n5_d2 to rf_n25_d2. Fixed-configuration plots and component changes are in the detailed report. Feature removal also changes which information is available. These are exploratory within-dataset comparisons.

Source: results/metafeatures_v1/selected_model_summary.csv and selected_endpoint_comparisons.csv

## 18. Breast Cancer: available features

Target: 65 seconds.

The solid curves show the selected model's validation accuracy and CSC. Dashed accuracy curves show the best candidate in each condition. Colors represent seeds. At 25% availability versus the original condition, selected accuracy changes range from -1.71 to 1.71 percentage points. CSC changes range from -2.51 to 0.00. Model changes: seed 42: dt_d2 to dt_d2; seed 1337: dt_d2 to dt_d2; seed 2026: dt_d4 to dt_d2. Fixed-configuration plots and component changes are in the detailed report. Feature removal also changes which information is available. These are exploratory within-dataset comparisons.

Source: results/metafeatures_v1/selected_model_summary.csv and selected_endpoint_comparisons.csv

## 19. Iris: training class balance

Target: 65 seconds.

The solid curves show the selected model's validation accuracy and CSC. Dashed accuracy curves show the best candidate in each condition. Colors represent seeds. Dotted square curves are the size-matched controls. At 25% class retention versus its size-matched control, selected accuracy changes range from -5.13 to 0.00 percentage points. CSC changes range from -10.67 to 0.08. Model changes: seed 42: gb_n50_d2 to dt_d2; seed 1337: rf_n25_d2 to rf_n25_d2; seed 2026: rf_n5_d2 to dt_d2. Fixed-configuration plots and component changes are in the detailed report. Feature removal also changes which information is available. These are exploratory within-dataset comparisons.

Source: results/metafeatures_v1/selected_model_summary.csv and selected_endpoint_comparisons.csv

## 20. Breast Cancer: training class balance

Target: 65 seconds.

The solid curves show the selected model's validation accuracy and CSC. Dashed accuracy curves show the best candidate in each condition. Colors represent seeds. Dotted square curves are the size-matched controls. At 25% class retention versus its size-matched control, selected accuracy changes range from -3.21 to 1.30 percentage points. CSC changes range from 0.00 to 0.82. Model changes: seed 42: dt_d2 to dt_d4; seed 1337: dt_d2 to dt_d2; seed 2026: rf_n5_d2 to rf_n5_d5. Fixed-configuration plots and component changes are in the detailed report. Feature removal also changes which information is available. These are exploratory within-dataset comparisons.

Source: results/metafeatures_v1/selected_model_summary.csv and selected_endpoint_comparisons.csv

## 21. What the remaining experiments establish

Target: 60 seconds.

The threshold study does not support a single universal CSC cutoff. The meta-feature study shows that changes in training data can alter accuracy, the chosen model, and CSC. The effects vary across seeds and datasets. Holding model configuration fixed generally reduces CSC when using fewer training examples, but selecting a different near-best configuration can reverse that pattern. Twelve of 63 conditions have no candidate reaching the original best accuracy minus two percentage points. Lower complexity alone is therefore insufficient evidence of improvement. Human interpretability and broader dataset generalization remain unvalidated.

Source: results/threshold_v1/report.md; results/metafeatures_v1/report.md

## 22. Appendix: components of the selected models

Target: 45 seconds (optional appendix).

These bars use the six minimum-CSC validation choices. Each bar adds mean path length, log2 of one plus distinct rules, and log2 of distinct full trees. Runs one, two and three correspond to seeds 42, 1337 and 2026. Iris selects a depth-two decision tree, then two five-tree depth-two forests. Breast Cancer selects depth-two trees twice and a depth-four tree once. Scores depend on the reference paths and stored representation. The figure is a decomposition, not evidence that the components require equal human effort.

Source: results/metric_development_v2: candidates.csv, selections.csv, selected_test_scores.csv, metric_comparisons.csv and presentation_summary.json.

## 23. Appendix: definitions and sources

Target: 30 seconds (optional appendix).

The presentation distinguishes established motivation from project choices. Grünwald supports the two-part MDL explanation; scikit-learn documentation supports the path-extraction implementation check. EEPC aggregation, EA feature burden and the additive CSC combination are explicit project conventions. The full protocol, reports and saved tables provide reproducibility. No historical controlled-study outputs feed this deck.

Source: References listed on this slide and in docs/metric_development_sources.md.
