# CSC improvement strategy: controlled comparisons, rule demand, and clearer evidence

**Project:** iML 26/27, Exercise 1  
**Date:** 8 November 2026  
**Status:** Proposed implementation strategy; no experiments described below have been executed by creating this document.  
**Primary project:** `D:\Mech&Robotics\Sem-3\iML\Ex1\csc-project\csc-project`  
**Scope:** Strengthen Task 1 through three controlled comparisons, repair the interpretation of rule budgets, and improve the plots. No human study, new model family, or optional Task 2 is required.

## 1. What we want to establish

Answer these three questions in order:

1. **At the same node count, does CSC distinguish different prediction-path burdens?**
2. **At the same tree count, does CSC distinguish repeated logic from diverse logic?**
3. **At similar predictive performance, does CSC reveal useful structural differences beyond node count and total expected path length?**

Success is a reproducible answer, including a negative or qualified answer. Success is not obtaining a more favorable scatterplot or lowering the correlation with node count.

Keep the current primary definition:

```text
CSC = C_path + C_encoding + C_ensemble
C_path     = average internal splits visited per actual tree and reference example
C_encoding = log2(1 + S), where S counts distinct split conditions across the model
C_ensemble = log2(U), where U counts distinct full-tree signatures
TEPL       = average total internal splits visited across all trees per example
           = M * C_path
```

Lower CSC means lower measured structural complexity under this definition. It does not establish better human comprehension.

**Decision:** Test the existing formula before revising it. Any later formula revision must receive a new version, a stated rationale, new tests, and a separate result file. Do not adjust weights or thresholds to produce preferred rankings.

## 2. Starting evidence and limitations

The existing project has 72 validation candidates, six selected-model test evaluations, two datasets and three seeds. Selection uses minimum validation CSC within 0.02 absolute balanced accuracy of the best candidate. These results must remain available as the original experiment.

Current findings:

- CSC and total node count have Spearman correlations of approximately 0.985 on Iris and 0.975 on Breast Cancer.
- Several shallow models satisfy the declared accuracy tolerance.
- The current seven-rule majority condition is crossed by 42/72 validation configurations.
- None of these findings validates a cognitive limit or establishes that CSC is superior to simpler baselines.

Important implementation facts, checked against the current source:

- `CognitiveStructuralComplexity.components(model, X_ref)` returns aggregate components and R5/R7/R9.
- `traverse()` already returns visited split identities for each reference row and tree.
- Complete signatures include topology, leaf outputs, class/output channel, and effective scale. Forest scaling is 1/M; boosting scaling is the learning rate.
- The original CSV does **not** contain per-example rule counts or path counts. The original experiment runner does not persist fitted models.
- Therefore, arbitrary rule-budget curves cannot be reconstructed from R5/R7/R9 alone. Reproducing original fits or running a new explicitly versioned experiment is necessary to collect that missing information.

## 3. Freeze the baseline and register the new protocol

Before implementing experiments:

- Back up the current project, figures, presentation and reporting scripts.
- Record SHA-256 hashes of `results/scores.csv`, `results/summary.json`, model/metric code, `pyproject.toml`, and `uv.lock`.
- Keep those original results unchanged. Write additions under `results/controlled_v1/`.
- Create `protocols/controlled_v1.md` stating fixtures, reference populations, seeds, matching rules, planned plots, and expected mathematical properties.
- Record the code revision or a source-file hash manifest, dependency versions, dataset identifiers and split IDs in every new run.
- Retain the existing decision-tree, random-forest and gradient-boosting families. Do not expand the grid just because an anticipated result fails to appear.

The first two studies are deliberately constructed mechanism demonstrations. The third uses predictive models and held-out evaluation. Keep these types of evidence visibly separate.

## 4. Controlled comparison 1: identical size, different paths

### Question

Can CSC distinguish two trees with identical size and split vocabulary but different routing structure? Can reference occupancy reverse their ordering?

### Exact construction

Use one numeric feature x in [0, 1] and these three split thresholds:

```text
x <= 0.25
x <= 0.50
x <= 0.75
```

Build two binary trees representing the same four intervals:

- **Balanced:** root split at 0.50; left child at 0.25; right child at 0.75. All four leaves have depth 2.
- **Chain:** root split at 0.25; its right child at 0.50; the next right child at 0.75. Leaf depths, in interval order, are 1, 2, 3, 3.

Assign identical leaf output distributions to corresponding intervals, for example alternating class-0/class-1 predictions. Keep class order, threshold precision and missing-routing policy consistent. Use complete numeric reference inputs, so missing routing is not exercised.

For both trees:

- Total nodes = 7; leaves = 4; distinct split tests S = 3.
- Actual trees M = 1; unique trees U = 1.
- C_encoding = 2; C_ensemble = 0.
- Predictions agree for corresponding input values.

**Implementation approach:** Prefer an explicit, small tree-fixture representation and adapter shared with metric helpers. If controlled sklearn tree arrays are constructed instead, isolate that code in the fixture module, pin its supported sklearn version, and validate every node and traversal. Do not rely on ordinary training to accidentally produce the intended topology. Do not add fake fitted attributes merely to bypass validation.

### Reference populations and expected results

Use deterministic inputs away from thresholds. Supply the same reference rows to both trees within each comparison. Use 100 rows for exact occupancy arithmetic; these are constructed examples, not 100 independent experimental replications.

| Reference profile | Leaf occupancy, left to right | Balanced C_path | Chain C_path | Balanced CSC | Chain CSC |
|---|---|---:|---:|---:|---:|
| Uniform intervals | 25%, 25%, 25%, 25% | 2.00 | 2.25 | 4.00 | 4.25 |
| Most inputs exit early | 80%, 10%, 5%, 5% | 2.00 | 1.30 | 4.00 | 3.30 |
| Most inputs reach deep leaves | 5%, 5%, 45%, 45% | 2.00 | 2.85 | 4.00 | 4.85 |

These are analytical test targets, not reported experimental measurements. Generate and verify them before publishing them as results. Include boundary-value tests separately to verify <= routing.

### What this answers

Node count cannot distinguish the trees. CSC distinguishes them through reference-weighted paths. A deeper tree can receive a lower CSC when most reference inputs exit early. Maximum depth alone cannot represent that effect.

**Required limitation:** For a single tree, TEPL equals C_path. With S and U held fixed here, CSC adds a constant to TEPL. This study demonstrates value beyond node count, not beyond TEPL. It also demonstrates representation dependence: the two trees can implement the same function yet receive different CSC scores.

### Outputs and figure

- `q1_path_controls.csv`: fixture ID, reference profile, node/leaf counts, S/U/M, components, TEPL, prediction-agreement result.
- One figure with the two small tree diagrams, followed by paired path/CSC results for the three occupancy profiles.
- Caption: “Constructed functionally equivalent trees; same size and split vocabulary; different routing and reference occupancy.”

### Acceptance gate

Node/leaf/rule counts match exactly; leaf outputs match by interval; predictions agree; analytical path means and CSC values match; ranking reverses under early-exit occupancy. Failure means fixing the fixture or implementation, not changing the expected story.

## 5. Controlled comparison 2: identical ensemble size, different redundancy

### Question

Does CSC detect diversity that raw tree count, node count and TEPL cannot distinguish?

### Primary fixed-size design

Construct twelve depth-1 classification stumps over twelve numeric features. Stump j splits feature j at zero, with identical class/output conventions. Use an exact full-factorial reference set of {-1, +1} values across twelve features: 4,096 rows. This is small enough for deterministic evaluation and ensures all split conditions are exercised.

Fix ensemble size M = 12. Vary the number of unique stumps:

```text
K in {1, 2, 3, 4, 6, 12}
```

For each K, take the first K stumps and repeat each exactly 12/K times. Treat the model as an equal-weight probability-averaging ensemble, with every tree contribution scaled by 1/12.

| Quantity | Fixed or varied |
|---|---|
| Actual trees M | Fixed at 12 |
| Total nodes / leaves | Fixed at 36 / 24 |
| Each path length | Fixed at 1 |
| C_path / TEPL | Fixed at 1 / 12 |
| Distinct rules S | K |
| Unique full trees U | K |
| Unique encountered rules B(x) | K for every reference row |
| Expected CSC | 1 + log2(1 + K) + log2(K) |

The constructed ensemble needs a validated prediction/traversal interface. Do not claim it is a conventionally trained random forest; label it an artificial averaging ensemble used to test the metric.

### Interpretation

This directly separates CSC from tree count, node count and TEPL: those baselines remain constant while distinct logic increases.

**Do not claim predictive performance is controlled in this primary design.** Different feature-specific stump banks can produce different functions. It is a mechanism test, not evidence that one ensemble is better for a real task.

### Secondary replication-invariance check

Start with a fixed averaging ensemble and replicate **every constituent tree equally**, for example 1x, 2x and 4x. Verify:

- Averaged predictions remain equal, within numeric tolerance.
- C_path, S, U, CSC and unique encountered-rule counts remain unchanged.
- Actual tree count, total nodes and TEPL increase proportionally.

Because contribution scale is included in signatures, verify the actual implementation rather than assuming this property. Equal scaling within each ensemble should preserve the number of unique signatures even though signature values differ between ensemble sizes.

Do not generalize this to copying one member of a heterogeneous ensemble: changing mixture weights can change predictions and average path length. Do not apply the prediction-invariance claim to additive boosting.

### Outputs and figure

- `q2_redundancy_controls.csv`, with actual and unique tree counts, S, components, TEPL, B summaries, and prediction agreement where applicable.
- Main figure: CSC against K; a separate aligned panel shows fixed node count and TEPL. Avoid unrelated units on a dual axis.
- Small replication table showing unchanged predictions/CSC versus growing storage and traversal cost.

### Acceptance gate

Verify exact equality controls, the closed-form CSC formula, and prediction equality only for the equal-replication check. Explain that these are intended responses of the formula, not independent validation of human interpretability.

## 6. Controlled comparison 3: comparable accuracy, different complexity

### Stage A: use existing validation results first

This stage requires no training. Treat it as exploratory evidence on already-inspected datasets.

1. Partition the original 72 validation rows by dataset and seed. Never match candidates across different splits.
2. Enumerate every unordered candidate pair within each group.
3. Mark a pair as performance-matched when its absolute balanced-accuracy difference is <= 0.02. This is an operational tolerance, not a statistical equivalence test.
4. Separate near-best pairs, where **both** members are within 0.02 of that group's best validation accuracy, from other performance-matched pairs. This prevents two equally poor candidates from dominating the story.
5. For each pair report signed and absolute differences in accuracy, CSC, nodes, TEPL, S, U and the three components. Fix pair orientation by configuration ID before calculating signs.
6. Report whether CSC and each baseline agree, tie, or disagree about which member is simpler. Use a prespecified float tolerance only for computational equality, e.g. absolute 1e-10, not a post-hoc “meaningful difference” threshold.
7. Include all eligible pairs in a machine-readable table. Summarize separately by dataset and seed; pairs share models and are not independent observations.
8. Display a small deterministic subset: closest accuracy difference first, then lexicographic configuration IDs, up to three near-best pairs per group. Do not choose only the largest CSC differences. If showing a disagreement example, label it illustrative and report how many such pairs exist.

Useful questions:

- Are any near-best pairs equal in node count but different in CSC?
- Are any equal in TEPL but different in CSC?
- When rankings disagree, which component explains the difference?
- Does the apparent difference arise mainly from exact threshold diversity, ensemble size or reference occupancy?

If exact matches do not exist, report that absence. Do not silently widen matching tolerances until a desired example appears. A later approximate-size analysis must be explicitly secondary with prespecified bins.

**Interpretation boundary:** Structural disagreement demonstrates additional descriptive information, not that CSC is the more valid ranking. Without an external outcome such as human task performance, superiority remains untested.

### Stage B: optional stronger comparison of selection procedures

Do this only after Q1, Q2, the budget redesign and Stage A work. Keep the original grid and formula unchanged.

Compare four fixed selectors using the same validation accuracy eligibility rule:

- Minimum CSC.
- Minimum total nodes.
- Minimum TEPL.
- Maximum validation balanced accuracy, with a deterministic configuration-ID tie-break.

Use nested stratified evaluation: five outer folds per dataset, three inner folds per outer training set. Use the existing twelve configurations. Average inner-fold validation accuracy and each complexity statistic before selecting configurations. Apply delta=0.02 to the mean accuracy, never outer-test accuracy.

Refit each distinct chosen configuration on the entire outer training fold, then evaluate it once on the outer holdout. Recompute reference-dependent CSC/TEPL on that outer holdout for reporting; never feed those outer values back to selection. If several selectors choose the same configuration, fit it once and reuse the evaluation.

Expected maximum workload across two datasets: 2 x 5 x 3 x 12 = 360 inner fits, plus at most 40 outer refits. Predeclare fold seeds and a stop/fallback rule before execution. Do not run repeated CV merely to chase a stronger conclusion.

Report per-fold paired differences and selection agreement. Fold results are dependent; do not treat them as independent samples for ordinary significance tests. Two datasets are insufficient for broad population-level performance claims.

**Previously inspected data:** Nested evaluation is stronger separation of tuning and assessment, but these datasets have already informed project development. New seeds or shuffled folds do not create a genuinely unseen confirmation dataset. Describe this as a locked follow-up evaluation on known benchmarks. Truly confirmatory claims require independent data reserved before development.

### Outputs and figures

- `q3_matched_pairs.csv`: all pairs and eligibility flags, not just displayed examples.
- `q3_pair_summary.csv`: counts and ranking agreement per dataset/seed.
- Optional `q3_selector_outer_results.csv`: selector, outer fold, chosen configuration and test metrics.
- Figure: paired dot plots for matched accuracy and complexity, with the same pair ordering in each panel.
- Optional selector figure: accuracy versus complexity plus a table of selection agreement. Use fold points and clearly labeled descriptive spread.

## 7. Fix the budget issue by changing the claim and measurement display

### The problem to solve

Neither choosing seven rules nor using R7 > 0.50 establishes a human limit. Moving the cutoff to another number would preserve the same problem. A CSC cutoff cannot be inferred from these data either.

**Primary decision:** Replace the headline “interpretability boundary” with a **rule-demand profile**. Keep R5/R7/R9 only as explicitly hypothetical operating points and for comparison with the original report.

### Record per-reference-example measurements

For each model and reference example x, retain:

```text
B(x) = number of distinct split identities encountered across all trees
L(x) = total internal-node visits across all trees, counting repeats
M    = number of actual contributing trees
```

B measures unique encountered conditions; L measures repeated traversal. Record M alongside them to expose ensemble size, but do not invent a calibrated aggregation-cost score. None directly measures mental effort.

Extend metric reporting through a separate method such as `reference_details()` or a backward-compatible details option. Preserve the scalar metric interface and existing aggregate output schema. Avoid duplicating split canonicalization or traversal logic in the new reporter.

Persist `reference_demands.parquet` or CSV with:

```text
dataset, study, fixture_or_model_id, seed_or_fold, reference_split,
reference_row_id, B, L, M, metric_version, run_id
```

Use stable row IDs and split manifests. No labels or original sensitive feature values are required in the demand table. Persist aggregate summaries and, optionally, trusted local fitted artifacts for later reproducibility.

### Measurement collection without rewriting history

- Q1/Q2: record details during fixture evaluation.
- Original real-data models: rerun the exact frozen fits in the locked environment to collect details into a **new** directory; the old fitted objects were not saved.
- Check reconstructed aggregate scores against the original CSV, excluding timing fields. Require the same selections and numeric agreement within a declared tolerance, such as 1e-10 absolute and relative.
- If reconstruction differs, stop and diagnose versions, data ordering and implementation. Do not overwrite old outputs or claim new details describe the old fitted models without reconciliation.
- These reconstructed predictions are additional measurements of an existing protocol, not independent new validation evidence.

### Primary budget-free views

1. Plot the empirical distribution or ECDF of B(x) for a small prespecified set of selected models.
2. Plot the full exceedance curve:

   `R(b) = fraction of reference examples with B(x) > b`

   Evaluate every integer b from zero to the maximum observed B; show the full relevant range. Use the same budget axis when comparing models in a panel.
3. Report median B, empirical 90th/95th coverage budgets, mean L and M.
4. Define a coverage budget explicitly:

   `b_q = smallest nonnegative integer b for which at least q of reference rows have B(x) <= b`.

   Thus b_90 describes the budget covering 90% of observed reference predictions. It is not a claim that humans can process that many rules. Use this integer order-statistic definition, not an interpolated fractional “number of rules.”

For ties, use exact integer comparisons. R(b) must be non-increasing and reach zero at max(B). It may be nonzero at b=0. A constant-leaf model has B=L=0.

Do not present the area under R(b) as independent new evidence: for nonnegative integer B, the sum of R(b) over b >= 0 equals mean B.

### Hypothetical scenarios, if a boundary is needed for the assignment

Separate the assumed capacity b from the tolerated exceedance alpha. A scenario flags a model when R(b) > alpha. Use a predeclared grid, such as budgets 1 through 15 and alpha in {0.10, 0.25, 0.50}, with the full demand curve available beyond 15. This grid is a sensitivity illustration, not calibration.

Retain the original (b=7, alpha=0.50) scenario as a labeled reference point. Do not optimize b or alpha against accuracy, CSC separation or a desired flagged percentage.

If the rubric requires a genuine human interpretability threshold, explicitly state that it remains unanswered without a human-grounded task and calibration data. The scope of this project can support hypothetical operational limits, not validate them.

### Budget acceptance gate

- Mean L equals TEPL within tolerance.
- Every row satisfies 0 <= B <= L and B <= global S.
- Curves reproduce stored R5/R7/R9 when evaluated on the reconstructed reference populations.
- Exact repeated stumps increase L when replicated but do not increase B.
- Quantile coverage and strict greater-than semantics pass hand-calculated tests.
- The display never labels models “uninterpretable” or equates CSC=7 with a cognitive boundary.

## 8. Plot redesign linked to the research questions

### Existing accuracy figure

Keep validation and test results separate. Add explicit `seed 42` labels rather than bare numbers. Preserve all candidates and disclose overlapping points. Provide a common-scale version for comparison across datasets; retain a clearly disclosed zoomed version only if useful. Draw a validation accuracy-CSC frontier only within each dataset/seed; do not construct a pooled frontier across different splits.

Use selected configuration labels or a compact adjacent table. Label the two-percentage-point tolerance clearly. Display test summaries separately and identify the selection procedure, not a single supposedly fixed model.

### Controlled path figure

Show the two seven-node diagrams with identical split thresholds and corresponding leaves. Below them, plot the paired changes under the three reference populations. Make the reversal under early-exit occupancy visually obvious. Include TEPL so the audience sees that this effect is already captured by path length.

### Controlled redundancy figure

Plot actual M and unique U separately. Show CSC changing with K while nodes and TEPL remain fixed. Explain artificial construction in the title/caption. Include replication invariance as a small supporting table.

### Matched-performance figure

Use paired dot plots or a compact comparison table. Connect only models evaluated on the same split. Include accuracy differences, CSC, nodes and TEPL. If no meaningful disagreement is found, show that agreement instead of highlighting numerical noise.

### Components and baseline figures

Keep configuration labels and component totals. Grayscale/hatching remains suitable for components; reserve family colors/shapes for model identity. Keep full baseline correlations in the report. In the talk, use one sentence explaining strong agreement and one controlled example showing what is held fixed.

### Rule-demand figure

Replace the main R7-versus-CSC scatter with full R(b) curves or an ECDF of B. Add coverage-budget summaries and L/M to expose the distinction between unique logic and total work. Put the old R5/R7/R9 percentages in a secondary sensitivity panel. Curves for constructed fixtures and trained models belong in separate panels.

### Shared visual requirements

- Consistent family labels, accessible colors and marker shapes.
- Accuracy always shown as percentages; differences labeled percentage points.
- Raw data visible; no unlabelled jitter or hidden low-performing models.
- Vector PDF/SVG plus 300-dpi PNG; slide variants without dense report captions.
- Repeated-split spread labeled descriptive, never automatically a confidence interval.
- One principal result per slide, readable at the final presentation size.
- Regenerate slide previews before rebuilding `presentation/presentation.pdf`; packaging old previews does not update charts.

## 9. Proposed files and commands

These are **future deliverables and proposed entry points**, not commands currently guaranteed to exist.

```text
protocols/controlled_v1.md
iml_interpretability/controlled_fixtures.py
scripts/run_controlled_comparisons.py
scripts/analyze_matched_performance.py
scripts/collect_reference_demands.py
scripts/plot_controlled_results.py
tests/test_controlled_comparisons.py
tests/test_reference_demands.py
results/controlled_v1/
    manifest.json
    q1_path_controls.csv
    q2_redundancy_controls.csv
    q3_matched_pairs.csv
    q3_pair_summary.csv
    reference_demands.csv
    rule_demand_summary.csv
    figures/
```

Avoid a second, incompatible implementation of CSC. Refactor shared measurement helpers only when needed, with regression tests proving the original formula is unchanged.

Proposed workflow after implementation:

```powershell
uv sync --extra dev
uv run pytest
uv run python scripts/run_controlled_comparisons.py
uv run python scripts/analyze_matched_performance.py
uv run python scripts/collect_reference_demands.py
uv run python scripts/plot_controlled_results.py
# Then regenerate presentation previews and rebuild the required PDF.
```

The first two commands exist today. Document the exact arguments, output directories and behavior of every new script when implemented. Do not change the original experiment entry point to silently overwrite the baseline study.

## 10. Priorities, decision gates and stopping rules

| Stage | Deliverable | Gate before proceeding |
|---|---|---|
| 0. Freeze | Backup, hashes, locked protocol | Original study recoverable; hypotheses fixed |
| 1. Q1 | Equal-size tree controls | Exact analytical checks and matching predictions pass |
| 2. Q2 | Fixed-size diversity and replication controls | Formula, fixed baselines and applicable invariance pass |
| 3. Q3 Stage A | All matched validation pairs | Same-split matching, complete pair tables, no cherry-picking |
| 4. Budget repair | Per-row measurements and full curves | Original aggregates reproduced; invariants pass |
| 5. Communication | New plots, README, five-minute PDF | Every claim traceable; all pages visually inspected |
| 6. Optional Q3 Stage B | Locked nested selector comparison | Adequate time; no reopening evaluation to tune the story |

**Minimum complete improvement:** Stages 0–5. Do not let optional nested evaluation displace clear controls or a correct budget explanation.

If Q1/Q2 contradict the formula's intended behavior, investigate implementation first. If Q3 shows almost complete agreement with baselines, retain that finding and narrow the claim. If accurate demand reconstruction fails, publish only the fixture curves and label real-model curves unavailable until resolved.

## 11. Revised five-minute presentation story

| Slide | Time | Main message |
|---|---:|---|
| 1. Question, CSC and compact protocol | 40 s | What size counts miss; what CSC claims to describe |
| 2. Q1: size held fixed | 55 s | Path occupancy changes CSC; TEPL captures this too |
| 3. Q2: ensemble size held fixed | 55 s | Distinct logic changes CSC while nodes/TEPL stay fixed |
| 4. Q3: matched predictive performance | 75 s | Show actual agreement or differences beyond baselines |
| 5. Rule demand and limitations | 75 s | Continuous demand profiles, hypothetical capacities, no human cutoff |
| **Total** | **300 s** | Rehearse; timing is a target, not a measured duration |

Keep full original scatters, baseline correlations, detailed component bars and additional sensitivity plots in the README or backup material. Do not shrink all of them onto five slides.

## 12. What the finished work may conclude

A defensible conclusion, conditional on verified results:

> Controlled examples show exactly when CSC distinguishes models that share raw size or traversal totals. Real-data comparisons show whether those distinctions occur among similarly accurate candidates. Rule-demand profiles describe the distribution of encountered conditions without assuming a universal human capacity. CSC remains a representation-dependent structural proxy; its practical cognitive validity is untested.

If real-data rankings mostly reproduce node count, say so. The improvement is a more precise, falsifiable account of the metric's behavior, not a promise that every new comparison will favor it.

## 13. Final completion checklist

- [ ] Original CSV, summary, metric version and results remain intact.
- [ ] Q1 has exact controls, matching functions and verified occupancy-dependent expectations.
- [ ] Q2 fixes M, nodes and TEPL while varying diversity; replication claims have appropriate scope.
- [ ] Q3 includes every eligible pair and labels exploratory evidence honestly.
- [ ] Any optional outer evaluation is isolated from model selection.
- [ ] Per-example rule-demand provenance is recorded; old aggregate measurements reconcile.
- [ ] Full rule-demand curves and coverage budgets replace the unsupported human-threshold headline.
- [ ] Figures distinguish constructed examples, validation comparisons and test evaluation.
- [ ] Tests, audit checks and documented regeneration commands pass.
- [ ] README, charts, notes and latest five-minute PDF tell the same evidence-backed story.
- [ ] No claim of human validation, novelty, superiority or causal accuracy improvement exceeds the evidence.
- [ ] GitHub fork/commit/course submission are handled separately once an actual repository is available.

## Source boundary

This strategy builds on the current project's inspected metric, traversal and experiment code, the saved validation/test summaries, the supplied blueprint and figure-polishing plan, and the user's requested three research questions. Proposed controls, numerical fixture targets and follow-up evaluation designs are plans, not completed research results. This document does not authorize modifying recorded results to match an anticipated conclusion.
