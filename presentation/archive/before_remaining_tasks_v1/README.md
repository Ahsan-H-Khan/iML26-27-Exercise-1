# CSC: developing and evaluating a tree-model complexity metric

This project develops Cognitive-Structural Complexity (CSC), then evaluates it on decision trees, random forests and gradient boosting trained on Iris and Wisconsin Diagnostic Breast Cancer. Predictive performance is checked first. Complexity is then compared among competitive models.

## Start here

- [Presentation: 12 main slides and 2 optional appendices](presentation/presentation.pdf)
- [Speaker notes](presentation/speaker_notes.md)
- [Full real-data report](results/metric_development_v2/report.md)
- [Locked protocol](protocols/metric_development_v2.md)
- [Definitions and source attribution](docs/metric_development_sources.md)

The main presentation targets roughly 14 minutes. This longer working version follows the user's instruction; it does not change any original five-minute course submission requirement.

## How the metric developed

1. **Expected Path Length (EPL):** mean internal decisions visited in one tree, weighted by reference-example occupancy.
2. **Expected Ensemble Path Complexity (EEPC):** sum the tree EPL values for total visits; divide by the actual number of prediction trees for the per-tree mean.
3. **Ensemble-adjusted complexity (EA):** multiply that mean path length by a logarithmic used-feature burden. This recognizes inputs used, but does not distinguish multiple thresholds on the same feature.
4. **MDL motivation:** consider how much structure needs describing. Formal MDL uses specified codes; CSC borrows this motivation without computing a genuine code length.
5. **CSC:** retain mean path length and add logarithmic distinct-rule and distinct-tree terms.

```text
EPL(T)     = mean internal decisions visited in tree T
EEPC_total = sum of individual tree EPL values
EEPC_mean  = EEPC_total / actual prediction-tree count
EA         = EEPC_mean * log2(1 + features used)
CSC        = EEPC_mean + log2(1 + distinct split tests) + log2(distinct full trees)
```

EA_total, based on total rather than mean paths, is saved as a separate sensitivity. A bare EPL field is populated only for single trees; ensembles have per-tree EPL lists and explicit total/mean columns. These conventions were fixed before the revised run. Used features are distinct split feature positions, not all dataset columns. Tree count includes class-specific trees in multiclass boosting.

CSC remains version 1.0. Split identities retain exact thresholds and routing; full-tree identities retain topology, outputs and channels. Its equal weights and logarithms are assumptions. A lower score is not proven easier for people to understand. The development is a sequence of proposals, not a formal derivation from MDL or a claim of demonstrated superiority.

## Evaluate prediction quality first

The original 12-configuration grid and seeds 42/1337/2026 are preserved. All configurations within a dataset/seed share stratified approximately 50% training, 25% validation and 25% test partitions. No learned preprocessing is needed for this numeric pipeline.

A `DummyClassifier(strategy="prior")` is trained on each training partition. A model enters the primary complexity comparison only when its validation balanced accuracy:

- strictly exceeds the dummy baseline; and
- is within 2 percentage points of the best candidate in that dataset/seed.

Balanced accuracy is average recall across classes. “Competitive” is a benchmark criterion, not a universal application threshold. All 72 candidates beat the dummy baseline; 35 also meet the near-best criterion (15 Iris, 20 Breast Cancer).

![Predictive performance](results/metric_development_v2/figures/predictive_performance.png)

## Apply and evaluate CSC

Among competitive candidates, choose minimum validation CSC, breaking ties by model ID. An accuracy-only selector chooses maximum validation balanced accuracy, with the same deterministic model-ID tie-break. Both choices are frozen before test scoring and are not refitted. Coincident choices reuse a single test evaluation.

![Accuracy and CSC](results/metric_development_v2/figures/accuracy_csc.png)

| Dataset | Selection | Mean test balanced accuracy | Mean test CSC |
|---|---|---:|---:|
| Iris | Accuracy-only | 96.44% | 6.10 |
| Iris | Minimum CSC | 96.44% | 6.10 |
| Breast Cancer | Accuracy-only | 94.39% | 11.12 |
| Breast Cancer | Minimum CSC | 92.45% | 4.59 |

Iris selects the same models under both rules. Breast Cancer trades some predictive performance for lower measured complexity. These are descriptive means across three overlapping splits; individual rows and sample SDs are in the report. Test data were already inspected in prior work, so this is follow-up rather than independent confirmation.

Within-group rankings show strong agreement between CSC and node count. EA differs more in groups with several competitive models. Differences in ranking do not establish which metric is better for people. The two Iris groups with only two competitive models provide especially weak evidence about ranking agreement.

## Reproduce

From this project directory:

```powershell
uv sync --extra dev
uv run pytest
uv run python scripts/run_metric_development.py
uv run python scripts/plot_metric_development.py
```

These scripts take no arguments. They write only to `results/metric_development_v2/`, reconstruct fits from the locked original setup, reconcile all 78 original result rows, and preserve original `results/scores.csv` and `results/summary.json`. New tables contain 72 validation candidates, 6 dummy baselines, 12 selector decisions and 9 distinct selected test evaluations. Split IDs, dataset hashes, environment versions, source hashes and reconciliation errors are saved.

Do not run `scripts/run_experiment.py` to reproduce v2: it remains the original baseline entry point and would overwrite the old results. No controlled-study script is part of the v2 workflow.

### Rebuild the presentation

Generate v2 tables and figures first. The slide builder uses Node.js and `@oai/artifact-tool`, supplied by the desktop runtime or an existing installation. If the module is outside Node's search path, set `CSC_ARTIFACT_MODULE` to the absolute `artifact_tool.mjs` path returned by the desktop runtime's workspace-dependencies tool.

```powershell
node scripts/build_presentation.mjs
python scripts/build_pdf.py
```

The second command needs reportlab in that Python environment. The bundled desktop Python includes it; reportlab is not part of the project's modeling dependency extra. The slide builder writes a manifest, notes and high-resolution previews. PDF packaging uses exactly the manifest-listed slides, excluding stale images. The supplied PDF contains rendered pages; chart PDFs/SVGs in the results directory remain vector assets. A private editable draft, when supported by the runtime, is not the reviewed submission artifact.

## Files and history

`path_metrics.py` adds the earlier path/feature measures; `metric_development.py` implements the revised protocol. The existing CSC formula and original experiment entry point are unchanged. `metric_comparisons.csv` includes EEPC_mean and EA_total sensitivities; `selector_test_comparison.csv` joins each selector to its test record. See [implementation details](docs/implementation.md) for original adapter policies.

The earlier controlled study is superseded for the primary analysis and presentation. Its results and scripts remain historical and are not read by the v2 pipeline. Previous presentation/README copies are in `presentation/archive/before_metric_development_v2/`. Budget measurements remain a clearly hypothetical report appendix; they are not selection criteria or human-memory limits. Optional Task 2, formal MDL coding, human studies and GitHub/course submission are outside this revision.
