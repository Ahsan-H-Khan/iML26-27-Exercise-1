# Current revision note

The active workflow is metric_development_v2; see the repository README. The original CSC implementation and experiment described below are preserved. New path_metrics.py adds EEPC_total (= TEPL), EEPC_mean (= c_path), used-feature burden and EA without altering CSC. The controlled-study research workflow is historical, while shared adapter regression tests remain useful.

# Implementation and verification

## Starter interfaces inspected

- `Dataset(name: str, X: pandas.DataFrame, y: pandas.Series)`; `load_titanic(data_home=Path("data")) -> Dataset`.
- `get_models(seed=0) -> dict[str, ClassifierMixin]`: extended to twelve configurations.
- `InterpretabilityMetric.score(model, dataset) -> float`: unchanged signature. New CSC declares `requires_reference=True`.
- `evaluate(datasets, models, metrics, seed=0, test_size=.25) -> DataFrame`: unchanged arguments and original columns. CSC receives the held-out Dataset; old metrics retain training context.
- `scripts/run_experiment.py`: preserved entry point and results/scores.csv destination. Main study uses `run_protocol()` for validation selection before testing.

The original three tests passed. Original Titanic demonstration completed with accuracy 0.832317 and relative_n_leaves 1.0 in the installed sklearn 1.7.2 environment. Original download and optional uv installation required network access. Final experiments and all fourteen tests ran in the original uv lock's isolated environment (sklearn 1.9.1); package versions are saved per row.

## Exact policies

Splits are tuples (feature position, <= operator, exact hexadecimal float threshold, native missing-left flag). Missing routing remains part of the identity even for a reference dataset without missing values. Rounding to six decimals is a secondary sensitivity calculation; traversal always uses the actual threshold. Inputs are converted to float32 to match sklearn. Dense numeric inputs are supported; empty, infinite and mismatched matrices are rejected.

Classification leaf values are normalized within each output to probabilities, accommodating both count and probability storage conventions. Regression leaves retain numeric values. Full signatures include leaf shapes, ordered topology, class labels, output-channel index and scale. Forest trees have scale 1/M; boosting trees have learning_rate. No tolerance is applied to leaf outputs. Feature labels do not affect identity.

Boosting uses every actually fitted stage and output channel. Default constant initialization is outside the tree proxy; custom initializers are rejected. XGBoost, histogram boosting, categorical splits and arbitrary pipelines are unsupported. Supported regressors share extraction, but the reported experiment concerns classification only.

The independent traversal is tested against sklearn apply and decision_path on the three classifier families; NaN leaf routing is checked separately against apply. Tests cover root-only trees, stumps, exact duplicates, distinct features, changed leaf payloads, exact threshold changes, output channels and scales, unbalanced 90/10 occupancy, reference-row permutations, feature renaming, budget deduplication, deterministic runs and validation selection.

## Reproducibility scope

The fixed grid and delta=.02 were chosen from the blueprint before interpretation. A selected model is not refit: its training size remains about 50% of the full dataset, and test labels never select configurations. Identical split indices are used for every model within a dataset/seed. Small stratified splits can produce noisy selections. The three repeated splits overlap and are not independent confidence-interval samples.

CSV numeric metrics are deterministic for the locked environment and seeds; timings are inherently variable. No scientific novelty or superiority claim has been established. Optional bonus Task 2 and XGBoost were not included.
