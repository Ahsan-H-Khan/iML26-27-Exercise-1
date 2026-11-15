# Remaining tasks: implementation and completion

Completed locally on 15 November 2026. External submission is still the user's responsibility.

## Implemented

- Threshold analysis for budgets 5, 7 and 9, both all-candidate and eligible-only groups, explicit denominators, sensitivity counts and CSC range overlap.
- Training-sample, feature-availability and class-balance experiments on Iris and Breast Cancer, seeds 42, 1337 and 2026.
- Nested deterministic subsets and paired size-matched imbalance controls. Validation/test IDs remain fixed; all new model scoring uses validation only.
- 63 unique conditions and 756 candidate fits, plus 63 dummy baselines. Exact shared conditions retain all experiment aliases.
- Within-condition eligibility, minimum-CSC and accuracy-only choices, and an original-accuracy reference check.
- Fixed-configuration and selected-model comparisons, component changes, per-seed plots, provenance manifests and resumable caches.
- Updated README, threshold report and meta-feature report.
- Seven-slide, 300-second submission deck and a 23-slide long deck (21 main slides and two appendices), with notes and reproducible builders. The original presentation is archived.

## Main conclusions

The seven-test budget does not map to one increasing CSC cutoff on either dataset: flagged and unflagged ranges overlap. Budgets describe hypothetical rule demand, not demonstrated human capacity.

Dataset properties affect the learned models and CSC, with effects depending on dataset, seed and selection. Fewer training examples usually reduce CSC for fixed configurations, while selected configurations can change. Twelve of 63 conditions cannot reach the original best accuracy minus two percentage points. Lower complexity alone is insufficient evidence of improvement.

## Verification

- 40 tests passed, including nested subsets, disjoint splits, exact matched-control counts, deterministic selection, no-eligible handling and complete output coverage.
- Ruff checks passed.
- Source distribution and wheel built; the wheel was installed into an isolated verification directory and its new modules imported.
- The 72 original-condition validation measurements reproduced saved metrics, including R5/R7/R9. Existing original test evaluations remain untouched.
- All original metric-development-v2 output files, original scores/summary, CSC code, adapters, dependency specification and lock file match their original hashes.
- Both PDF decks rendered for visual review; page counts and the 300-second short-deck timing match their manifests.

## Remaining limits

This is an exploratory two-dataset study using previously inspected data. It establishes neither human interpretability nor universal causal metadata effects. Formal MDL coding and human experiments remain future work. No GitHub push or course submission was performed. Speaker timings are targets and require rehearsal.
