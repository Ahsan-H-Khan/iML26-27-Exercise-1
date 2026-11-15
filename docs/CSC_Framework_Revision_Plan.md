# CSC framework revision plan

Date: 14 November 2026  
Status: implementation instructions, not a report of new results  
Scope: refocus the existing project on developing CSC and evaluating it on trained Iris and Breast Cancer models.

## 1. Goal and narrative

The project should answer: **After establishing that trained models predict adequately, what does CSC tell us about their complexity?**

Present the development in this order:

1. Expected Path Length (EPL).
2. Expected Ensemble Path Complexity (EEPC): apply path-length measurement across the ensemble.
3. Ensemble-adjusted complexity (EA): path length multiplied by a logarithmic feature burden.
4. Minimum Description Length (MDL) as motivation for describing distinct model structure.
5. The final additive CSC formula.
6. Predictive performance on Iris and Breast Cancer.
7. CSC results among competitive models, followed by comparison with simpler measures.

Explain why each step was considered, what it captures, and what it misses. These are design motivations, not a claim that every successive formula has been empirically proven superior.

## 2. Exclude the previous controlled study

The user explicitly wants to ignore the previous controlled experiments.

- Remove Q1/Q2/Q3 controlled-study plots, artificial stump/tree fixtures, replication experiments and controlled pair-count headlines from the active README, presentation and primary results report.
- Do not use the 3/116 or 7/116 reversal counts as evidence in the new presentation.
- Do not run `run_controlled_comparisons.py`, `analyze_matched_performance.py`, `collect_reference_demands.py`, `plot_controlled_results.py` or `plot_plain_slides.py` as part of the new main workflow.
- Preserve `results/controlled_v1/`, the previous presentations and their scripts as historical material. Label them superseded; do not delete results or silently overwrite them.
- Retain useful regression tests and shared traversal helpers. Excluding a research study does not require removing implementation safeguards.
- New primary results must come from original real-data results or a separately identified reproducible real-data evaluation. They must not depend on controlled-v1 tables.

## 3. Fix the mathematical definitions before implementation

### Common notation

For reference examples X = {x_1, ..., x_n}, let d_t(x_i) be the number of internal split nodes visited in prediction tree t. Count a leaf-only tree as zero. M is the number of actual prediction trees, including class-specific trees in multiclass boosting; it is not necessarily the number of boosting stages.

Use empirical expectations over a named reference split. Use the same reference examples when comparing candidates within a dataset and seed.

### EPL: individual tree

```text
EPL(T; X) = (1/n) * sum_i d_T(x_i)
```

Equivalent explanation: multiply each leaf's path length by the fraction of reference examples reaching that leaf, then sum.

Motivation: typical prediction paths are more informative than maximum depth alone.

Limitation: EPL alone does not express the full amount of distinct logic stored across an ensemble.

### EEPC: ensemble path measurement

The user defines EEPC as Expected Ensemble Path Complexity, applying EPL over the ensemble. That statement leaves sum versus average unspecified. Implement and report both explicitly:

```text
EEPC_total(F; X) = sum_t EPL(T_t; X)
EEPC_mean(F; X)  = EEPC_total / M
```

Working convention for the narrative: use `EEPC_total` when discussing all split visits required for an ensemble prediction, and explicitly introduce `EEPC_mean` when moving to CSC's per-tree average. This is an implementation convention, not a claim that the user previously specified summation.

Existing-code mapping:

- `tepl` is already EEPC_total.
- `c_path` is already EEPC_mean.
- For a single tree, EPL, EEPC_total and EEPC_mean coincide.

Do not put a bare “EPL” column on an ensemble without specifying the aggregation. Total visits count repeated execution; they are not necessarily distinct reasoning steps.

### EA: path length times feature burden

The user specifies EA as EPL multiplied by a logarithmic feature burden. Preserve that multiplicative structure.

Recommended explicit operational definition:

```text
F_used = number of distinct input feature positions used by internal splits
feature_burden = log2(1 + F_used)
EA(F; X) = EEPC_mean(F; X) * feature_burden
```

For a single tree, replace EEPC_mean with EPL. This plan chooses used features and the `1 +` convention so unused input columns do not increase complexity and a leaf-only model remains well-defined. These details, and the use of mean rather than total ensemble path length, are proposed conventions because the user's short definition did not specify them. Record them prominently before running the new evaluation; do not present them as recovered historical formulas.

If the intended EA used total ensemble visits, retain a separately named sensitivity column:

```text
EA_total = EEPC_total * log2(1 + F_used)
```

Do not choose which EA to headline based on which produces the preferred results.

Feature count and rule count are different: two thresholds on the same feature contribute one used feature but two distinct split tests. EA's limitation should be explained using that distinction. Multiplication also assumes path and feature burdens interact in a particular way; this is a design choice.

### MDL: motivation, not an unimplemented numerical baseline

Explain the description-length principle in plain language, then distinguish it from the current code. A formal predictive MDL treatment generally requires a model code and a data-given-model code; its numerical value depends on a specified coding scheme.

Before finalizing the presentation, find and read authoritative primary references for EPL/ensemble path measures where appropriate and MDL. Cite the actual formulation used. Do not invent provenance for EEPC or EA; label them project design steps unless a source is established.

Do not call MDL “the state-of-the-art metric” without specifying the task and supporting evidence. Do not claim CSC was experimentally compared against formal MDL when no such code exists. Implementing a complete MDL baseline is outside this revision's required scope.

### CSC: preserve the implemented formula

```text
CSC = EEPC_mean + log2(1 + S) + log2(U)
S = distinct canonical split tests across the entire fitted model
U = distinct full-tree signatures, including leaf outputs and output channels
```

Keep `csc.py`'s current formula and canonical identities unchanged. Its description is **an MDL-inspired structural proxy**, not a description length measured in bits.

Explain the design transition honestly: CSC changes from a multiplicative feature burden to additive distinct-rule and distinct-tree terms. This is a proposed design, not an algebraic derivation from EA or a formal consequence of MDL. Equal weights and logarithms remain assumptions.

## 4. Predictive performance comes before complexity-based interpretation

Keep Iris and Wisconsin Diagnostic Breast Cancer, the existing three model families, original 12-configuration grid, and seeds 42/1337/2026 for the first revision. Do not expand the search simply to obtain a more flattering result.

### Phase A: assess prediction quality

1. Preserve identical stratified train/validation/test partitions across candidates within each seed.
2. Fit all preprocessing only on training data. The current bundled numeric datasets require no learned preprocessing in the existing pipeline.
3. Fit a `DummyClassifier(strategy="prior")` on each training partition and evaluate its validation predictions as a simple predictive baseline.
4. Report validation balanced accuracy for every candidate. Ordinary accuracy and macro-F1 remain supporting columns.
5. Plot all candidates by model family and dataset, with the dummy baseline clearly marked. Show variation across seeds rather than only a best-case score.
6. State separately whether candidates beat the dummy baseline and whether they are close to the best candidate. Neither condition is an application-level guarantee of adequacy.

### Phase B: define which models enter the primary complexity comparison

Use the following transparent benchmark eligibility rule, fixed before producing the new analysis:

```text
eligible = (validation balanced accuracy > dummy validation balanced accuracy)
           AND (validation balanced accuracy >= group best - 0.02)
```

Here “group” means the same dataset and seed. This adds a baseline check to the original near-best rule. Call these **competitive benchmark models**, not universally “good enough” models. There is no justified universal accuracy floor in the supplied task. If an absolute adequacy threshold is later needed, justify and declare it separately before evaluation.

If no model qualifies, report that fact and do not silently select a winner. Computing complexity for every candidate is acceptable; predictive eligibility must govern the primary interpretation and selection.

### Phase C: compare and select

- Calculate EPL/EEPC, EA, CSC, total nodes, maximum depth, tree count and used-feature count for the same fitted models and validation reference set.
- Among eligible candidates, choose minimum validation CSC, breaking ties by model ID as in the original code.
- Also identify a validation accuracy-only reference choice, using maximum balanced accuracy and then model ID. This enables a direct account of the accuracy cost, if any, of selecting for lower complexity.
- Freeze both choices before scoring test data. Evaluate each unique selected model once and reuse its record if the choices coincide. Do not refit under a different protocol without a new explicit version.
- Compare test performance of these frozen choices descriptively. Test performance must not decide eligibility, metric weights, model IDs or chart examples.
- The old test results have already been inspected. Label this follow-up as a revision using existing benchmark data, not a new untouched confirmatory study.

## 5. Exact framework changes

| File | Required change |
|---|---|
| `iml_interpretability/tree_utils.py` | Reuse existing prediction-tree extraction and traversal. Add a used-feature collector from non-leaf nodes if needed. Preserve boosting channels and canonical identities. |
| `iml_interpretability/csc.py` | Preserve the scalar CSC API, formula and component meanings. Do not replace split count with feature count. |
| `iml_interpretability/path_metrics.py` — new | Implement per-tree EPL, EEPC_total, EEPC_mean, F_used, feature_burden, EA and explicitly named EA_total sensitivity using shared tree helpers. |
| `iml_interpretability/experiment.py` | Preserve original entry-point behavior. Extract reusable split/fit/scoring helpers only with regression checks; put the revised protocol in a separate function or module. |
| `iml_interpretability/metric_development.py` — new | Run the real-data revision: dummy baseline, candidate metrics, eligibility, frozen selectors and test evaluations. Record protocol/version metadata. |
| `scripts/run_metric_development.py` — new | Run the revised protocol and write only to `results/metric_development_v2/`. |
| `scripts/plot_metric_development.py` — new | Produce performance, accuracy-versus-CSC, component and baseline-comparison figures from v2 tables. No dependency on controlled-v1. |
| `scripts/build_presentation.mjs` | Replace the current simplified five-slide story with the development-and-evaluation narrative below. Read only the new primary tables and declared source notes. |
| `scripts/build_pdf.py` | Remove the hardcoded five-page loop. Read a slide manifest or ordered list of generated slides so all intended pages are included and stale previews are excluded. |
| `README.md` | Lead with metric development, predictive assessment, CSC evaluation, limitations and current reproduction commands. Move controlled-study links into a short historical section. |
| `tests/test_path_metrics.py` — new | Verify definitions and compatibility; these are software tests, not revived research experiments. |
| `tests/test_metric_development.py` — new | Verify eligibility, selectors, split boundaries and output schema. |

Recommended new output layout:

```text
protocols/metric_development_v2.md
results/metric_development_v2/
    candidates.csv
    dummy_baselines.csv
    selections.csv
    selected_test_scores.csv
    performance_summary.csv
    metric_comparisons.csv
    manifest.json
    report.md
    figures/
presentation/
    presentation.pdf
    speaker_notes.md
    slide_manifest.json
    archive/
```

Candidate rows must include dataset, seed, model ID/family, parameters, split ID, predictive scores, baseline score, eligibility and all metric components. Record package versions, metric definitions and source/input hashes. Retain full precision in tables and round only displayed labels.

Original `results/scores.csv` and `results/summary.json` remain immutable reference artifacts. Reconstruct existing fits with the locked environment and compare old numeric metrics before accepting extended results. Do not copy old predictive values onto a differently fitted model.

## 6. Figures that serve the main question

### Figure 1: predictive performance

Two panels, one per dataset. X-axis: “Model family”. Y-axis: “Validation balanced accuracy (%)”. Define balanced accuracy once as average recall across classes. Show the individual candidate/seed results and the dummy baseline. Do not show test scores for candidates used in selection.

### Figure 2: accuracy and CSC

X-axis: “CSC score (lower means less structural complexity)”. Y-axis: “Validation balanced accuracy (%)”. Use color and shape for model families. Show all candidates faintly, competitive candidates clearly, and selected models with an outline. Define the two-percentage-point tolerance in the caption. Avoid a single horizontal eligibility line when group-best values differ by seed; use seed panels or individual eligibility markers.

### Figure 3: what contributes to CSC

Use a small number of validation-selected real models. Label the components “Average path length”, “Distinct-rule term” and “Distinct-tree term”. Include the mathematical definitions in the caption or notes. Do not imply the three terms are calibrated units of cognitive effort.

### Figure 4: comparison with earlier measures

Use the same competitive real models to compare EEPC, EA, node count and CSC. Report within-dataset/seed ranking relationships without treating repeated comparisons as independent evidence. A different numerical scale is not evidence of a different ranking. Correlation is descriptive; neither high nor low correlation establishes usefulness to people.

Select a displayed real-model example by a documented rule, or explicitly label it illustrative if chosen after inspecting results. Do not reuse the controlled-study pair selection or headline counts.

### General presentation rules

Use short, literal titles, large labels and one question per slide. Expand each abbreviation on first use. Keep technical precision without stuffing definitions into every axis. Use shared scales for directly comparable panels, or clearly disclose differences. Put dense tables and extra metrics in the report. No decorative dashboards or promotional claims.

## 7. Presentation outline: approximately 12–15 minutes

The user has removed the five-minute restriction for this working presentation. The older assignment blueprint mentions a five-minute submission; keep that distinction in the documentation and make a separate short submission deck only if later required. Do not claim the course requirement itself changed.

1. **Why measure model complexity?** State the project question and distinguish prediction quality from interpretability.
2. **Expected Path Length.** Explain the average number of decisions using one small illustrative tree.
3. **From one tree to an ensemble.** Define EEPC and make total versus average explicit.
4. **Adding feature burden.** Show EA = path term times log2(1 + used features); explain why it was considered.
5. **What these measures miss.** Explain feature reuse, different thresholds and repeated structure. Use a teaching illustration, not results from the excluded controlled experiments.
6. **The description-length idea.** Introduce MDL with verified citations and distinguish formal coding from an intuitive motivation.
7. **The final CSC formula.** Connect each term to its motivation; show one worked calculation from a real fitted model.
8. **How we evaluated it.** Datasets, model families, shared splits, baseline and performance eligibility.
9. **Did the models predict well?** Show validation performance first. State what “competitive” means and how many qualify.
10. **CSC among competitive models.** Show accuracy versus CSC and the selected models.
11. **Does CSC add information?** Compare with EEPC, EA and node count on real models; report agreement as well as differences.
12. **Findings and limits.** Show frozen-choice test performance, summarize what CSC measures and what remains unvalidated.

Use the same teaching illustration through the conceptual slides where possible. The real-data evaluation, not the illustration, supplies the empirical evidence. Speaker notes must distinguish assumptions, mathematical consequences and observed results.

## 8. Budget analysis

Remove rule-demand curves and the seven-rule threshold from the main narrative. They are not necessary to explain metric development and predictive evaluation.

Retain historical budget outputs for traceability. If the assignment still requires a hypothetical boundary, provide a clearly labelled appendix or report section, using real-model demand measurements and explicitly chosen scenario parameters. Do not represent seven tests as an established human limit, and do not replace the main evaluation with a budget discussion.

## 9. Validation and acceptance criteria

- Original CSC values and original result hashes are preserved.
- EEPC_total equals the sum of individual tree EPL values and equals the existing TEPL value; EEPC_mean equals c_path.
- Used-feature counts exclude leaf sentinels and unused input columns. Different thresholds on one feature do not inflate F_used.
- EA follows the explicitly documented multiplicative convention; CSC remains additive and unchanged.
- Dummy models are trained on training data only. All candidate comparisons use shared splits.
- Eligibility, model selection and tie-breaking use validation information only. Test evaluation cannot revise those choices.
- No-eligible-model cases and selector coincidences are handled explicitly.
- Every displayed number is traceable to a saved row or declared aggregation.
- No active report or slide relies on controlled-v1 results.
- The full PDF includes every intended slide, with no stale slides, clipped text or unreadable axes. Review every rendered page.
- The README, speaker notes and figures explain the same experiment and formula conventions.
- MDL citations are verified; no unsupported “state of the art”, formal code-length, human-validity or superiority claim appears.

## 10. Implementation sequence

1. Archive the current README, presentation and script versions. Preserve original results and hashes.
2. Write `protocols/metric_development_v2.md` with the precise EEPC/EA conventions, eligibility rule and selectors before running the analysis.
3. Implement shared metric extensions and meaningful regression tests.
4. Reconstruct original fits, check reconciliation, add dummy baselines, and produce new real-data tables.
5. Review predictive performance before drawing conclusions about CSC.
6. Generate the four primary figures and the real-data results report.
7. Verify external references for the development narrative and state which steps are project proposals.
8. Build the longer presentation, update PDF packaging, and visually inspect all pages.
9. Update the README and reproduction commands. Mark controlled-v1 as historical.

Proposed commands after implementation (the new scripts do not exist yet):

```powershell
uv sync --extra dev
uv run pytest
uv run python scripts/run_metric_development.py
uv run python scripts/plot_metric_development.py
# Rebuild slide previews using the documented artifact-tool runtime.
node scripts/build_presentation.mjs
python scripts/build_pdf.py
```

The slide tool requires its existing artifact-tool setup; PDF packaging requires reportlab. Document exact environment setup rather than assuming those are installed by the Python project's development extra.

## 11. Completion standard

The revision is complete when a reader can follow how CSC was proposed, see whether the trained models predict competitively, inspect CSC on those models, and judge what it adds over simpler measures. A result that CSC mostly agrees with those measures is acceptable. The goal is a clear evaluation of the proposed metric, not a predetermined claim that it is better.
