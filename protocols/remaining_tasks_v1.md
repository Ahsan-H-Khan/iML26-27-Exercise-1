# CSC: remaining tasks and completion goals

Date: 15 November 2026  
Status: agreed scope translated into a plan; experiments below are not yet completed.  
Project: iML Exercise 1 — Quantifying Interpretability

## 1. Objective

Complete the two outstanding parts of the assignment:

1. **Task 1: investigate an interpretability threshold**, connecting CSC to an explicit, hypothetical prediction-level rule budget.
2. **Task 2: investigate dataset meta-features**, measuring how training sample count, feature availability and training class balance relate to CSC and predictive performance.

Keep the established narrative: develop the metric, establish predictive competitiveness, then interpret its complexity scores. Do not redesign CSC to obtain a preferred outcome.

The assignment specifies submission by 15 November 2026 at 23:59 and a five-minute presentation. The existing longer deck remains useful for explanation; it does not replace the submission time constraint unless the instructor permits that change. Verify the deadline's applicable timezone with the course system if necessary.

## 2. Preserve the completed work

- Freeze CSC-1.0: mean path length + log2(1 + distinct split tests) + log2(distinct full trees).
- Preserve original results/scores.csv, results/summary.json and results/metric_development_v2/.
- Keep the existing tree adapters, canonical identities, model grid, datasets and seeds.
- Save new measurements in separate result directories with protocol and provenance records.
- Do not revive the earlier artificial-tree controlled study. Its fixtures, plots and headline pair counts remain historical.
- Retain useful software regression tests. New dataset interventions are Task 2 experiments, not a return to the excluded artificial-tree study.
- Preserve the distinction between a reproducible structural index and a human-validated interpretability measure.

## 3. Goal A — finish Task 1's threshold investigation

### Research question

Does CSC consistently track the number of different conditions encountered during a prediction, and does that relationship support a useful operational boundary?

### Definitions

For reference example x:

```text
B(x) = number of distinct split tests encountered across all prediction trees
R_b  = fraction of reference examples for which B(x) > b
```

Repeated identical split tests count once in B(x). Different thresholds on the same feature count separately. B(x) is distinct-test demand, not total execution cost, whole-model rule count or CSC. Combining the trees' outputs requires effort that B(x) does not measure.

Primary hypothetical budget: b = 7. Sensitivity budgets: b = 5 and b = 9.

An optional operational flag is R_b > 0.50: more than half of reference predictions exceed the chosen budget. Label BOTH b and 0.50 as assumptions. No study has established these as a universal model-interpretability boundary.

### Required analysis

1. Use all 72 original real-model validation candidates, with competitive models clearly identified using the v2 gate.
2. Reuse saved R5/R7/R9 values where available. Reconstruct additional per-example measurements only if needed, using the same fitted configuration, split and reference population; reconcile aggregate values before accepting them.
3. Plot CSC against R5, R7 and R9, separately by dataset. Use common axes and identify model families and seeds. Do not pool differing splits into one supposedly independent sample.
4. Report how often the operational flag changes when the budget changes from 5 to 7 to 9.
5. Examine overlap: for each budget and dataset/seed, compare the CSC ranges of flagged and unflagged models. If the groups overlap, report that a simple cutoff does not perfectly separate them. If either group is empty, report separation as unassessable.
6. Treat any apparent CSC boundary as descriptive. Do not optimize a cutoff on these data and call its performance independent validation. A separate cutoff-learning study is outside this minimum completion plan.
7. Answer the assignment's question directly: what boundary was investigated, what happened, and why it does or does not support saying a model has stopped being interpretable.

### Evidence limits

Seven is a hypothesis about a count of distinct tests; CSC = 7 is not that hypothesis. If a memory study is cited, identify the exact paper, read the relevant tasks and findings, and explain the mismatch between memory chunks and visible model conditions. If that source cannot be established, simply call seven an illustrative budget.

CSC and B(x) both come from model structure. Their association is a structural consistency check, not independent evidence of human comprehension. A defensible outcome is that no reliable universal CSC cutoff emerges.

### Deliverables and completion gate

- threshold_summary.csv: dataset, seed, model, eligibility, CSC, R5/R7/R9 and strict flags.
- threshold_sensitivity.csv: denominators, flagged counts, switching counts and range-overlap summaries by dataset/seed.
- A clear figure and a short threshold section in the primary report.
- Explicit interpretation of whether a stable boundary was observed, without asserting a human limit.

## 4. Goal B — complete Task 2: dataset meta-features

### Common protocol

Use Iris and Wisconsin Diagnostic Breast Cancer. Preserve the original stratified approximately 50/25/25 train/validation/test partitions for seeds 42, 1337 and 2026. Within a seed, every condition and candidate uses the same validation and test row IDs.

Apply interventions only to the training partition, except that feature-subset conditions must project training, validation and test onto the same selected columns. Do not select features using validation or test labels.

Fit the original twelve configurations per condition:

- Decision trees: maximum depths 1, 2, 4 and 8.
- Random forests: 5 or 25 trees, depths 2 or 5.
- Gradient boosting: 10 or 50 stages, depths 1 or 2.

Preserve the remaining original hyperparameters. Fit a training-prior dummy baseline for every condition. Keep model randomness fixed within each seed across conditions. Record any failures rather than silently changing the grid.

### B1. Training sample count

Question: how does the number of training examples affect predictive performance and learned structural complexity?

Conditions: 25%, 50%, 75% and 100% of the original training partition.

Implementation:

- Generate one deterministic random ordering within each training class, per seed.
- Take nested class-wise prefixes, with rounded target counts and at least two examples per class when feasible.
- Preserve approximate original training class proportions; record actual class counts and total sample count.
- Keep all original features and the validation/test rows fixed.

Interpretation: this measures training-size sensitivity under the chosen sampling procedure. It does not change validation reference size and therefore avoids mixing training-size effects with reference-size effects.

### B2. Feature availability

Question: how do available input features affect predictive performance and CSC?

Conditions: approximately 25%, 50%, 75% and 100% of original features. Convert each fraction to max(1, ceil(fraction * feature_count)) and deduplicate identical counts.

Implementation:

- Generate a deterministic random feature ordering per seed, independently of labels and results.
- Use nested prefixes of that ordering.
- Fit on all original training rows, projecting all partitions onto the same columns.
- Save original feature names/IDs and their local column positions. Preserve this mapping when interpreting split identities.
- Report both available feature count and the number actually used by the fitted model.

Interpretation: this is a feature-availability experiment. Removing columns also removes particular information, so do not claim it isolates feature count independently of feature quality. Differences across seeds include feature-subset variation and split variation. Do not call the removed features irrelevant, informative or redundant unless separately established.

### B3. Training class balance

Question: does changing the training class distribution affect CSC beyond the accompanying reduction in sample count?

Conditions:

1. Original training partition.
2. Retain approximately 50% of a designated class; retain the other classes.
3. Retain approximately 25% of that class; retain the other classes.
4. A size-matched stratified control for condition 2, preserving approximate original proportions.
5. A size-matched stratified control for condition 3, preserving approximate original proportions.

Implementation:

- Designate the least frequent training class for each dataset/seed, breaking ties by class label. Determine this from training labels only, before model fitting.
- Use deterministic nested sampling and retain at least two examples of the designated class.
- For each size-matched control, allocate exactly the same total training size across classes using original training proportions and a deterministic rounding rule. Save actual counts.
- Use all original features and keep validation/test class proportions and row IDs unchanged.
- Record the designated class, per-class counts, smallest class proportion, and largest-to-smallest class count ratio.

Interpretation: compare each imbalance condition with its own size-matched control. These are paired comparisons within dataset/seed. They reduce the sample-count confound but do not remove all variation from which examples were sampled.

### Expected scale and execution limit

The nominal design has 4 + 4 + 5 = 13 conditions per dataset/seed. The full original training condition occurs in all three experiments; reuse it once when training rows, columns and hyperparameters match exactly.

That gives up to 11 unique conditions × 2 datasets × 3 seeds × 12 candidates = **792 candidate fits**, plus dummy baselines. Deduplicate further only by exact training-row/column/configuration identity, not similar scores. This is a workload estimate, not a runtime guarantee.

Lock the full design before starting. If time is insufficient, complete Task 1 first, then complete sample-count conditions, feature conditions, and finally imbalance/control pairs. Include both datasets and all declared seeds for a completed analysis; clearly label unfinished sections. Do not stop or extend an experiment because its result looks favorable or unfavorable.

## 5. Performance first, complexity second

For each dataset/seed/condition:

1. Fit candidates and the dummy baseline using that condition's training data only.
2. Measure validation balanced accuracy; retain ordinary accuracy and macro-F1 as supporting measures.
3. Compute CSC, all three CSC components, EEPC total/mean, EA, actual prediction-tree count, total nodes and used-feature count.
4. Define competitive candidates as those strictly beating the condition's dummy validation balanced accuracy AND within 0.02 of that condition's best validation candidate.
5. Choose minimum validation CSC among competitive candidates, with model-ID tie-breaking. Identify the validation accuracy-only choice separately.
6. If no candidate qualifies, record an explicit no-eligible-model status. Do not substitute a low-performing model or drop the condition from the report.

### Comparing conditions fairly

Condition-relative eligibility does not imply equal performance across conditions. Always display the best validation accuracy and the CSC-selected model's accuracy alongside complexity.

Also flag whether each candidate meets the fixed reference level from the unmodified training condition: original best validation balanced accuracy minus 0.02. This is a secondary comparability check, not a replacement selector. If no candidate reaches it after an intervention, state that predictive performance was not maintained.

Report results in two complementary ways:

- Fixed configuration: compare the same model ID across conditions, showing what happened to that fitted model type.
- Selected model: compare the minimum-CSC eligible choice and record model family/ID changes.

A reduction in CSC after removing information is not automatically an improvement. Separate structural simplification from degraded predictions and from switching to a different configuration.

### Test-set policy

Use validation results for Task 2's primary exploratory relationships. Existing Task 1 frozen-choice test results remain the primary test report. Do not use test data to choose intervention levels, metric weights or the most appealing conclusion. Any additional Task 2 test evaluation must be specified before viewing those results and evaluate only already-frozen choices.

Previously inspected data remain a limitation; these experiments do not create a fresh confirmatory test set.

## 6. Required figures and interpretation

Make one main figure for each dataset property, with two aligned panels:

- Validation balanced accuracy versus the actual dataset property.
- CSC versus that same property.

Use actual sample/feature counts, not just percentages. For imbalance, show the class-distribution measure and paired size-matched controls clearly. Show individual seed results, label selected model changes, and avoid implying that all configuration/seed rows are independent replicates.

Include fixed-configuration plots and component breakdowns in the detailed report. At minimum, answer:

- Did predictive performance change?
- Did CSC change, and which component changed?
- Did the selected model family change?
- Was the relationship consistent across seeds and datasets?
- Did the fixed configuration show the same pattern as the selected model?

With only two datasets, report these as within-dataset intervention results. Do not claim universal relationships across datasets or fit an elaborate meta-feature predictor to two dataset-level observations.

## 7. Implementation and output plan

Add new modules/scripts without changing the original experiment entry point:

```text
protocols/remaining_tasks_v1.md
iml_interpretability/dataset_variants.py
iml_interpretability/metafeature_study.py
scripts/analyze_interpretability_threshold.py
scripts/run_metafeature_study.py
scripts/plot_metafeature_study.py
tests/test_dataset_variants.py
tests/test_metafeature_study.py
results/threshold_v1/
    threshold_summary.csv
    threshold_sensitivity.csv
    report.md
    figures/
results/metafeatures_v1/
    candidates.csv
    dummy_baselines.csv
    selections.csv
    condition_manifest.json
    fixed_configuration_summary.csv
    selected_model_summary.csv
    report.md
    figures/
```

These paths are proposed additions, not existing commands. Exact module names may be adapted to the current framework while preserving the separation of old and new results.

Each condition manifest must include dataset/seed, intervention and condition ID, training/validation/test row IDs, feature IDs, actual class counts, sampling randomness, dataset and source hashes, package versions and hyperparameters. Save fitted-model measurements at full precision and round only figure labels.

Estimate execution time with one complete dataset/seed pilot. Cache completed condition outputs by verified input/configuration hashes, so interruption does not require repeating all fits. Do not reuse a fit if training examples or columns changed.

## 8. Completion checks

### Task 1 threshold checks

- [ ] R_b uses the strict comparison B(x) > b.
- [ ] All fractions have explicit reference-set denominators.
- [ ] Original R5/R7/R9 values reconcile when reconstructed.
- [ ] Sensitivity across all three budgets is reported, including contradictory findings.
- [ ] The report distinguishes a hypothetical rule-budget boundary from a human-validated CSC cutoff.

### Task 2 experiment checks

- [ ] Training, validation and test IDs are disjoint; evaluation IDs are fixed across conditions within a seed.
- [ ] Sample and feature subsets are nested and reproducible.
- [ ] Feature projection is consistent across partitions and original feature identities are retained.
- [ ] Imbalance controls have matching total sample counts; actual class proportions are saved.
- [ ] Dummy baselines and candidates fit only their condition's training data.
- [ ] Eligibility and selections are calculated within the correct condition, without test leakage.
- [ ] No-eligible conditions remain visible.
- [ ] Fixed-configuration and selected-model results are both reported.
- [ ] Reduced complexity is interpreted alongside predictive performance and reference-condition comparability.
- [ ] Original result files and CSC formula are unchanged.

### Deliverable checks

- [ ] README explains both tasks, current findings, limitations and exact reproduction commands.
- [ ] The report explicitly answers the threshold and meta-feature questions.
- [ ] The longer explanatory deck is updated with the new findings.
- [ ] A five-minute submission deck is prepared unless the instructor explicitly permits another duration.
- [ ] PDFs are visually reviewed and all displayed values reconcile with saved tables.
- [ ] Installation and relevant tests pass; incomplete work is labelled honestly.

## 9. Submission priorities

1. Finish the Task 1 threshold investigation and its written conclusion.
2. Complete the bounded Task 2 study in the declared order.
3. Update README, results report and slides to match the evidence.
4. Preserve the long deck under a clearly named file; put the required short submission deck at presentation/presentation.pdf if the course constraint still applies.
5. Verify the actual GitHub fork, stage only intended deliverables, and confirm the course submission procedure. Local files alone are not a submission. This planning document does not authorize an external submission or message.

Suggested five-minute story: metric and motivation; predictive performance and CSC results; hypothetical threshold investigation; Task 2 findings; limitations and conclusion. Keep formula-development detail and additional plots in the longer deck/report.

## 10. Definition of done

The project is complete when the audience can understand what CSC measures, see that predictive performance was checked, follow the investigated interpretability boundary, and understand how dataset properties affected the learned models and their CSC scores.

The goals do not require CSC to outperform every simpler measure, a universal threshold to exist, or every dataset property to show a strong effect. A clear, reproducible negative or mixed result is a valid outcome.


Implementation freeze: training subsets preserve original row order; feature subsets preserve original column order. RNG streams 811 (class pools) and 812 (features). Class allocations use rounded counts with a minimum of two. Exact shared input conditions are fitted once. All Task 2 scores are validation-only.
