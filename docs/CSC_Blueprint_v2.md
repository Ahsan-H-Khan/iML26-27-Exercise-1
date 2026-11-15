# CSC Implementation Blueprint — iML 26/27, Exercise 1

**Version:** 2.0 — aligned with the supervising professor's `README.md` (8 November 2026)  
**Project:** iML 26/27 — Exercise 1: Quantifying Interpretability  
**Status:** Implementation plan, **not** an implementation or a report of completed experiments  
**Deadline:** **Sunday, 15 November 2026, 23:59**, as specified by the professor  
**Primary objective:** Complete **Task 1 (1 point)**; attempt **Task 2 (2 optional bonus points)** only after Task 1 works end to end.

> **Source boundary.** The assignment requirements, submission paths, template package structure, and setup commands below come from the professor's supplied `README.md`. The **Cognitive-Structural Complexity (CSC)** formula and experiment design are **our proposed approach**, not instructions or an existing metric supplied by the professor. The actual repository source files (`pyproject.toml`, `.py` interfaces, tests) were **not** supplied with the README; inspect them in the fork before making interface-level assumptions.

---

## 0. Executive plan — what to build and in what order

**Minimum successful deliverable (Task 1):**

1. Fork the professor's repository; keep it **installable**.
2. Confirm the existing starter runs: `uv sync --extra dev`, `uv run pytest`, `uv run python scripts/run_experiment.py`.
3. Replace the example `RelativeNumberOfLeaves` **as the primary metric** with CSC, retaining it as a baseline if useful.
4. Evaluate multiple decision-tree configurations, a random forest, and a gradient-boosted model on **at least two datasets**.
5. Save reproducible numeric results to `results/scores.csv`.
6. Investigate an **explicitly hypothetical**, non-human-validated interpretability boundary. Compare model accuracy and CSC.
7. Rewrite the **repository's** `README.md` to describe **your** code, exact reproduction commands, and key results.
8. Commit a **5-minute presentation** as **`presentation/presentation.pdf`** and submit the fork by the deadline.

**Bonus deliverable (Task 2):** Explore associations between dataset meta-features (sample count, feature count, class balance, etc.) and the CSC of models that maintain reasonable predictive performance.

**Priority:** Correctness and a reproducible result across models/datasets > XGBoost support > extensive datasets/hyperparameter searches > visual polish. Three working model families are preferable to four partially implemented ones.

---

## 1. Understand the professor's repository contract

The professor describes an **installable Python package** with the following starter layout. **Preserve these entry points**; adding helper modules is allowed, but rebuilding the repository from scratch is unnecessary.

```text
<forked-repository>/
├── README.md                               # REWRITE before submission
├── pyproject.toml                          # existing package/dependencies/ruff configuration
├── iml_interpretability/
│   ├── data.py                             # dataset container + load_titanic(); extend here
│   ├── models.py                           # get_models(); extend model variants
│   ├── metric.py                           # InterpretabilityMetric base + example metric
│   ├── evaluate.py                         # evaluate(): fit and score combinations
│   ├── tree_utils.py                       # ADD, if helpful: extraction, path traversal
│   └── signatures.py                       # ADD, if helpful: canonical split/tree signatures
├── scripts/
│   ├── run_experiment.py                   # REQUIRED existing experiment entry point
│   └── plot_results.py                     # OPTIONAL: regenerate presentation figures
├── tests/
│   ├── ...                                 # existing tests, retain and extend
│   ├── test_csc.py                         # ADD: hand-calculated metric tests
│   └── test_tree_adapters.py               # ADD: verify extraction/traversal
├── results/
│   ├── scores.csv                          # REQUIRED-by-template output location
│   └── figures/                            # OPTIONAL outputs
└── presentation/
    └── presentation.pdf                    # REQUIRED submission artifact
```

The actual fork may contain additional files (lockfile, `__init__.py`, data cache, etc.). **Do not delete them blindly**. In particular, use the professor's existing `pyproject.toml` rather than replacing it with an unrelated requirements-only project.

### Original commands — run these before editing anything

From the **repository root**:

```bash
uv sync --extra dev
uv run pytest
uv run python scripts/run_experiment.py
```

The professor also provides a pip alternative on Linux/macOS:

```bash
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"
pytest
python scripts/run_experiment.py
```

On Windows, activate the virtual environment with the appropriate platform-specific command.

**Environment detail (checked against current uv documentation):** `--extra dev` installs a `dev` **optional extra** declared under `[project.optional-dependencies]`. Modern uv also supports a `[dependency-groups]` `dev` group, which is a **different mechanism**. Keep the command from the professor's README **if it works with the supplied `pyproject.toml`**. Do not silently replace it with `--group dev` without inspecting that file. The `uv run` command executes programs in the managed project environment.

**Before coding, inspect these actual files in the fork:** `pyproject.toml`, `data.py`, `models.py`, `metric.py`, `evaluate.py`, `scripts/run_experiment.py`, and existing tests. Record their real function signatures. The README does **not** specify the calling convention of `InterpretabilityMetric` or `evaluate()`, so no implementation guide can reliably invent it.

**Data note:** The starter fetches **Titanic from OpenML** on first execution and caches it under `data/` relative to the working directory. An internet connection may be required on the first run. This is a *starter demonstration*, not a required research dataset. Built-in scikit-learn datasets avoid that first-run dependency.

---

## 2. Research question, scope, and claims

**Research question:** Can a three-component additive structural-complexity score rank the *auditing burden* of individual decision trees and tree ensembles more informatively than maximum depth, total node count, or estimator count alone?

**Primary scope:** Classification with `DecisionTreeClassifier`, `RandomForestClassifier`, and `GradientBoostingClassifier` from scikit-learn. The same structural definitions extend to regressors. **XGBoost is an optional extension**, not necessary for a valid Task 1 submission.

**Explicit hypotheses:**

- **H1 — path sensitivity:** Models with longer typical prediction paths tend to have higher path complexity.
- **H2 — split vocabulary:** Additional genuinely distinct split conditions increase the encoding component.
- **H3 — redundancy:** Exact copies of an identical tree should not increase CSC, under the chosen **structural** interpretation.
- **H4 — trade-off:** At similar predictive performance, models can differ substantially in CSC.
- **Bonus H5:** Dataset meta-features may relate to the minimum CSC needed for near-best predictive performance.

**Important claim boundary:** CSC is a **novel proposed combination of existing complexity concepts**, subject to a prior-art check. Without human-subject measurements, it is **not a validated measurement of human understanding**. Do not label estimated cognitive-budget exceedance rates as "human simulation accuracy."

---

## 3. Freeze the mathematical specification *before* running experiments

Let a fitted model contain **M actual contributing prediction trees**: `T_1, ..., T_M`. Let `X_ref` contain **n evaluation examples** drawn from a declared reference population (validation or held-out data). Define `d_t(x)` as the number of **internal split nodes visited** in tree `t` for input `x`; a root-only leaf contributes zero.

### Component 1: expected prediction-path complexity

\[
C_{\mathrm{path}}(F;X_{\mathrm{ref}})
=\frac{1}{nM}\sum_{i=1}^{n}\sum_{t=1}^{M}d_t(x_i).
\]

This is the average path length **per tree and sample**, not the total number of splits traversed by an ensemble. Using the average avoids building ensemble size into both the first and third components. Unbalanced trees are weighted by **actual leaf occupancy** in `X_ref`.

### Component 2: distinct split-encoding complexity

Define a **canonical split test** by:

```text
(feature_position, operator, threshold_or_category_set, missing_value_routing)
```

Count unique split tests **across the entire model**, not merely those visited on `X_ref`:

\[
S(F)=\bigl|\{\operatorname{split\_id}(v):v\text{ is an internal node of }F\}\bigr|.
\]

Then:

\[
C_{\mathrm{encoding}}(F)=\log_2(1+S(F)).
\]

Repeated tests count once; tests with different thresholds count separately. The logarithm models diminishing incremental burden. **It is not a genuine Minimum Description Length (MDL) code** and does not represent bits in a prefix-free encoding. Call it a *logarithmic distinct-rule proxy*, not exact information content.

### Component 3: distinct-tree synthesis complexity

Represent each tree by a deterministic, recursive **full-tree signature** including:

- Split topology and ordered branches.
- Canonical split tests (including missing routing, where applicable).
- Leaf prediction/output payloads and relevant class/output channel.
- Any effective output scaling needed to distinguish trees' contributions, when stored outside the leaves.

Let `U(F)` be the number of **distinct full-tree signatures** among the `M` prediction trees. Define:

\[
C_{\mathrm{ensemble}}(F)=\log_2(U(F))\quad\text{with }U(F)\ge1.
\]

A **single tree** has ensemble component zero; several **exact duplicate trees** also have zero under this proxy. Trees with identical split structures but **different leaf outputs** must generally produce different signatures.

### Final additive score

\[
\boxed{
\operatorname{CSC}(F;X_{\mathrm{ref}})
=C_{\mathrm{path}}+C_{\mathrm{encoding}}+C_{\mathrm{ensemble}}
=\frac{1}{nM}\sum_i\sum_t d_t(x_i)
+\log_2(1+S)+\log_2(U)
}
\]

**Interpretation:** lower CSC = lower **proposed structural** complexity. The score is **dimensionless**. One unit of path length, one unit of logarithmic rule count, and one unit of logarithmic tree count are **not empirically calibrated equivalent mental costs**. This is an explicit assumption to probe through ablations.

### Worked mini-cases (unit-test targets)

| Construction | `C_path` | `S` | `U` | Expected CSC |
|---|---:|---:|---:|---:|
| Single constant leaf | 0 | 0 | 1 | **0** |
| Single decision stump | 1 | 1 | 1 | **2** |
| Two exact copies of the stump | 1 | 1 | 1 | **2** |
| Two distinct stumps on different features | 1 | 2 | 2 | **1 + log2(3) + 1 ≈ 3.585** |
| Unbalanced tree, 90% depth 1 and 10% depth 3 | 1.2 | model-dependent | 1 | `1.2 + log2(1+S)` |

**Consistency caveat:** In a boosting model, adding an identical tree can **change the predicted function**, even if CSC does not change. That is a limitation of measuring *distinct structural logic*, not a universal functional-complexity invariant. Mention this explicitly in the presentation.

---

## 4. Exact implementation steps within the existing package

### Step 4.1 — Record the existing interfaces

Inspect the fork and document:

```text
iml_interpretability/data.py      Dataset type, load_titanic() return type
iml_interpretability/models.py    get_models() return type; estimator configuration
iml_interpretability/metric.py    InterpretabilityMetric public API
iml_interpretability/evaluate.py  evaluate() signature; when X/y are available
scripts/run_experiment.py         CLI/entry-point arguments and saved CSV schema
```

**Implementation constraint:** CSC needs **both a fitted estimator and an evaluation feature matrix** (`X_ref`), whereas the starter's example `RelativeNumberOfLeaves` may only require the fitted estimator. If the metric base class accepts only a model, carefully **extend the metric/evaluation interface** or provide a validated evaluation context. Avoid using training node counts as an undocumented substitute for held-out path length. Keep other existing metrics executable.

A **conceptual**, not repository-verified API might look like:

```python
# Illustrative ONLY: adapt to the actual InterpretabilityMetric interface.
result = csc_metric.score(fitted_model, X_ref)
# result contains csc, c_path, c_encoding, c_ensemble, S, U, M, ...
```

### Step 4.2 — Extract a normalized tree representation

Implement in `iml_interpretability/tree_utils.py` (or keep small helpers in `metric.py` if the starter is tiny):

```text
Tree(root, output_channel, effective_scale)
SplitNode(feature_position, operator, threshold, missing_route, left, right)
LeafNode(leaf_payload)
```

Contract: internal nodes have two children; leaf nodes have none. Use **position-based feature IDs after preprocessing** so column-name formatting cannot spuriously change CSC. Keep a saved mapping from encoded feature positions to human-readable names for plots.

Do **not** use a tree's storage node index as its structural identity; node IDs can differ across equivalent serializations.

### Step 4.3 — Implement the scikit-learn single-tree adapter FIRST

With a fitted `DecisionTreeClassifier` or `DecisionTreeRegressor`:

1. Obtain `estimator.tree_`.
2. Traverse from root node `0` using `children_left`, `children_right`, `feature`, and `threshold`.
3. For a leaf, store its `tree_.value` payload in a documented representation. **Check the installed scikit-learn version's leaf-value semantics**; do not assume these arrays are always raw class counts or always probabilities.
4. If supported, preserve native missing-value routing. Otherwise fail clearly on unsupported missing inputs.
5. Validate that every evaluation row reaches the **same leaf** as `estimator.apply(X_ref)`.
6. Validate path split counts against `estimator.decision_path(X_ref)` (visited-node count minus one for a normal binary tree, since the leaf is included).

When validation passes, compute all three CSC components for the single tree; test stump and constant-leaf cases.

### Step 4.4 — Extend to random forests

Use `RandomForestClassifier.estimators_` / `RandomForestRegressor.estimators_` to obtain the fitted trees. Apply the same adapter to **each** estimator.

- `M = len(estimators_)`.
- Compute path lengths for **every tree** on the same `X_ref`.
- Union all internal split identifiers into the model-wide set of `S` rules.
- Count unique full-tree signatures `U`.
- Do not discard a tree simply because it resembles another: it remains a prediction tree for **`C_path`**. Only its *distinctness* is deduplicated in `S` and `U`.

### Step 4.5 — Extend to scikit-learn gradient boosting

For `GradientBoostingClassifier` / `GradientBoostingRegressor`, inspect the fitted `estimators_` **array of trees** and flatten its contributing elements. Preserve class/output channels for multiclass classification.

- In current scikit-learn documentation, the classifier stores an array with shape `(n_estimators, n_trees_per_iteration_)`, and `n_trees_per_iteration_` is 1 for binary classification or the number of classes for multiclass classification.
- These are **regression trees that contribute to raw scores**, not independent class-probability trees.
- Tree count `M` means **actual contributing prediction trees**, not merely boosting rounds.
- Record `learning_rate` and any effective output scale; be consistent about including it in full-tree signatures.
- If early stopping reduces the fitted stage count, use the trees actually used in the deployed model.

**Version-specific guard:** Verify the real estimator properties in the installed environment; do not assume private internals are stable across scikit-learn releases.

### Step 4.6 — XGBoost ONLY if the core project is finished

For `XGBClassifier` / `XGBRegressor` with tree boosters, one possible extraction route is:

```python
import json
booster = model.get_booster()
trees = [json.loads(s) for s in booster.get_dump(dump_format="json")]
```

The JSON is an **interpretation dump**, not the native model-saving format. Parse leaf nodes, split tests, `yes`/`no` branch IDs and missing routing, then verify traversal against XGBoost's `apply(X_ref)` outputs. Watch for categorical splits, multi-output trees, and early stopping: `best_iteration` can cause prediction to use a subset of stored trees. **Do not claim general XGBoost support until those cases have been tested.**

**Pragmatic recommendation:** For the 15 November deadline, first finish `DecisionTreeClassifier`, `RandomForestClassifier`, and `GradientBoostingClassifier` with at least two datasets. Treat XGBoost as optional validation of portability, not a blocking dependency.

---

## 5. Canonicalization and metric algorithm — implementation specification

### Step 5.1 — Define exact split IDs

Use a deterministic tuple such as:

```python
split_id = (
    feature_position,                  # stable integer after preprocessing
    operator,                          # e.g. "<=", "<", "in"
    canonical_threshold_or_categories, # stable numeric/category serialization
    missing_route,                     # left/right/none
)
```

For numeric thresholds in the primary analysis, serialize the actual stored floating-point value losslessly where possible (e.g., `float(value).hex()`). Do **not** round silently. Categorical sets should have deterministic ordering. Distinct operators and missing routing should be preserved when they change decisions.

Perform an **optional sensitivity analysis** with documented threshold rounding (e.g., six decimals) to determine whether near-identical thresholds dominate `S`. Always retain exact-match CSC as the primary reported result.

### Step 5.2 — Define full-tree signatures

Recursive pseudocode:

```text
signature(leaf) = ("leaf", canonical_leaf_payload)
signature(node) = (
    "split", split_id(node), signature(left_child), signature(right_child)
)
full_tree_signature = (output_channel, effective_scale, signature(root))
```

Use immutable nested tuples or deterministic JSON serialization. The `leaf_payload` must use a consistent task-specific meaning: for classification trees, a class distribution/output vector; for boosting, the raw-score contribution, including scale if appropriate. Do not hash only topology or only thresholds and call the result a full signature.

**Known limitation:** Different representations may implement equivalent functions. Computing exact **functional equivalence** of arbitrary ensembles is outside the scope of Task 1.

### Step 5.3 — Count traversed decisions on reference data

```text
for each row x in X_ref:
    visited_split_ids_for_x = set()
    for each contributing tree T:
        node = T.root
        depth = 0
        while node is a SplitNode:
            visited_split_ids_for_x.add(split_id(node))
            depth += 1
            node = select_child(node, x)
        save depth in path_lengths[x, T]
    save len(visited_split_ids_for_x) in rule_budget[x]
```

At the end, `path_lengths` should have shape `(n_reference_examples, M)`; `rule_budget` length `n_reference_examples`.

### Step 5.4 — Compute and return *all* components

```python
import numpy as np

# Pseudocode: path_lengths, split_ids, and full_tree_signatures already extracted.
c_path = float(np.mean(path_lengths))
s = len(set(split_ids))
u = len(set(full_tree_signatures))
assert path_lengths.size > 0 and u >= 1
c_encoding = float(np.log2(1 + s))
c_ensemble = float(np.log2(u))
csc = c_path + c_encoding + c_ensemble

# Additional measurements (not the CSC score):
# tepl = path_lengths.sum(axis=1).mean()
# budget_r7 = np.mean(rule_budget > 7)
```

Return e.g. `csc`, `c_path`, `c_encoding`, `c_ensemble`, `num_trees`, `num_unique_splits`, `num_unique_trees`, `tepl`, `mean_epl`, and `budget_r5/r7/r9` in a stable structure. **Do not just return a rounded scalar** if analysis later needs components; preserve full precision in the CSV.

### Step 5.5 — Implement independent baseline metrics

At minimum include `total_nodes`, `total_leaves`, `max_tree_depth`, `n_trees`, **TEPL** (sum of mean tree-path lengths), and `RelativeNumberOfLeaves` if useful. These let you test whether CSC actually captures more than an existing trivial count.

---

## 6. Verify implementation before any model comparison

Add tests under the existing `tests/` directory. Maintain the starter tests and extend them; run `uv run pytest` after each adapter.

| Test | Setup | Required result |
|---|---|---|
| Constant leaf | One root-only leaf | `C_path=0, S=0, U=1, CSC=0` |
| Stump | Root split with two leaves | `CSC=2` |
| Exact duplicate | Add an identical tree with identical leaf outputs | CSC **unchanged** |
| Two feature-distinct stumps | One split on feature 0 and one on feature 1 | `CSC ≈ 3.5849625` |
| Different leaf payload | Identical topology, altered output | `U` increases |
| Unbalanced path | 90% of rows depth 1, 10% depth 3 | `C_path=1.2` for the single tree |
| Data sensitivity | Same tree, shifted evaluation occupancy | Only `C_path` may change; `S` and `U` fixed |
| Sample permutation | Reorder reference rows | All CSC values unchanged |
| Feature renaming | Rename labels, preserve feature positions/data | All CSC values unchanged |
| Exact threshold | Change only a split threshold | `S` can increase / signature changes |
| Library traversal | Compare custom leaf IDs to `apply(X_ref)` | Exact match for every row and contributing tree |
| Multiclass boosting | Several trees per stage | No omitted class/channel trees |
| Reproducibility | Same data, seed, model, environment | Same score within specified tolerance |

**Failure rule:** If the adapter cannot represent a model feature correctly (missing routing, categoricals, unusual multiclass output), **raise an explicit unsupported-model error** rather than silently computing a plausible but inaccurate CSC score.

**Quality gate:** Do not start the full experiment matrix until single-tree, forest, and gradient-boosting adapters pass their traversal tests on tiny synthetic datasets.

---

## 7. Plan datasets, train/test protocol, and model grid

### 7.1 Datasets in `iml_interpretability/data.py`

**Minimum useful selection:**

1. **Iris** — 4 features, three classes (tests multiclass handling).
2. **Breast Cancer Wisconsin (Diagnostic)** — 30 numerical features, binary outcome (tests dimensionality).
3. **Controlled `make_classification`** — variable informative/redundant feature counts and noise (useful for Task 2).

All three are available through `sklearn.datasets`, without downloading OpenML data. You **may** keep the starter's `load_titanic()` for comparison, but it is not mandatory. Keep its implementation working unless there is a clearly documented reason not to.

**Important:** `data.py` already contains a `Dataset` container. Add loaders using **that existing container and its actual constructor**; do not invent a new incompatible type.

### 7.2 Evaluation protocol

For each dataset and random seed:

1. Split data using a stratified, reproducible procedure, e.g. `train_test_split(..., test_size=0.25, stratify=y, random_state=seed)`.
2. Use identical split indices for every compared model on that dataset/seed.
3. Fit preprocessing **only on training data**. Trees generally do not need standard scaling. If categoricals are present, keep encoded feature mapping stable.
4. Use a **validation fold** (or cross-validation within the training portion) to choose configurations and any performance tolerance. Keep a genuinely untouched **test set** for final unbiased performance reports.
5. Calculate `C_path` on a declared `X_ref`, preferably a validation feature matrix during development and test features after all choices are frozen. Record which split was used.
6. Evaluate predictive performance using `balanced_accuracy` and `macro_f1`; ordinary accuracy is optional. For a three-class dataset, macro-F1 is particularly useful.
7. Repeat with at least **three seeds** if feasible: `42`, `1337`, and `2026`.

Do not tune models or choose a threshold based on test labels, then report that same test accuracy as an unbiased estimate. The **structure-only** CSC calculation does not require test labels, but model selection still must avoid test-set reuse.

### 7.3 Small, deadline-appropriate model grid in `models.py`

| Family | Minimal configurations | Why |
|---|---|---|
| Decision tree | `max_depth ∈ {1, 2, 4, 8}` | Depth sensitivity |
| Random forest | `n_estimators ∈ {5, 25}` and `max_depth ∈ {2, 5}` | Tree count and average depth |
| Gradient boosting | `n_estimators ∈ {10, 50}` and `max_depth ∈ {1, 2}` | Boosting stages and depth |

Initial constants: `random_state=seed`, reasonable `min_samples_leaf` for trees/forests, and `learning_rate=0.1` for gradient boosting. **Freeze them in the experiment protocol** before interpreting results.

This is 4 + 4 + 4 = **12 model configurations** per dataset and seed. With three datasets and three seeds, it produces **108 fitted runs** (assuming all combinations succeed). If time is tight, run **two datasets × one seed** first, validate `scores.csv`, then expand.

**Dataset count is not causal evidence.** With only three natural datasets, meta-feature correlations are descriptive. Task 2 becomes stronger with controlled synthetic parameter sweeps.

---

## 8. Wire the experiment through the professor's entry points

The preferred implementation path is:

| Existing file | Action |
|---|---|
| `iml_interpretability/data.py` | Extend current dataset loader functions/container |
| `iml_interpretability/models.py` | Replace/expand `get_models()` with named reproducible variants |
| `iml_interpretability/metric.py` | Implement CSC using the real `InterpretabilityMetric` interface; preserve optional baselines |
| `iml_interpretability/evaluate.py` | Fit each model on each dataset, pass `X_ref` into CSC, collect predictions and scores |
| `scripts/run_experiment.py` | Keep a single documented CLI entry point; save `results/scores.csv` |
| `tests/` | Add unit tests and integration tests |
| `README.md` | **Rewrite at the end** with accurate instructions and findings |
| `presentation/presentation.pdf` | Add final 5-minute presentation PDF |

**Conceptual evaluation logic; adapt to the actual repository APIs:**

```text
for dataset in load_datasets():
    for seed in seeds:
        train, validation, test = make_reproducible_split(dataset, seed)
        for name, estimator in get_models(seed):
            estimator.fit(X_train, y_train)
            csc_parts = compute_csc(estimator, X_validation)
            val_predictions = estimator.predict(X_validation)
            record(dataset, seed, name, hyperparameters,
                   csc_parts, validation_performance, meta_features)
write_all_records("results/scores.csv")
```

Keep **results collection separate from visualization** so all reported figures can be regenerated from `scores.csv`. If models are selected using validation, fit or evaluate their frozen configurations on untouched test data in a separate final pass.

### Required output columns (recommended extension of the starter CSV)

```text
dataset,seed,model_family,model_name,hyperparameters_json,
n_train,n_validation,n_test,n_features_raw,n_features_encoded,n_classes,
n_trees,n_unique_trees,n_unique_splits,total_nodes,total_leaves,max_depth,
c_path,c_encoding,c_ensemble,csc,tepl,
budget_r5,budget_r7,budget_r9,
accuracy,balanced_accuracy,macro_f1,
score_split,fit_seconds,metric_seconds,package_versions_json
```

The professor requires the **location** `results/scores.csv`; the exact column schema above is **our design**. Preserve any existing required columns from the starter implementation.

---

## 9. Interpretability threshold: what is defensible *without* human subjects

### 9.1 Why Miller's 7 ± 2 is NOT a direct CSC cutoff

George A. Miller's 1956 paper, *The Magical Number Seven, Plus or Minus Two*, discusses immediate-memory/information-processing phenomena and chunking. It **does not establish** that humans can track exactly seven tree splits, seven distinct features, or seven CSC points. Later working-memory research also discusses smaller chunk capacities under some conditions.

**Do not write:** `CSC > 7 => human cannot interpret this model` or report simulated human accuracy without a validated human-response model.

### 9.2 Separate rule-budget exceedance (primary threshold experiment)

For each example `x`, collect the **set of distinct split conditions actually visited across all trees**, `S_path(x)`. Define:

\[
B(x)=|S_{\mathrm{path}}(x)|,
\qquad
R_b(F)=\frac{1}{n}\sum_{i=1}^{n}\mathbf{1}[B(x_i)>b],
\qquad b\in\{5,7,9\}.
\]

- `B(x)` = number of **unique encountered tests** across the entire ensemble for prediction `x`.
- `R_b` = fraction of reference predictions exceeding **hypothetical** rule budget `b`.
- If all paths reuse the same tests, duplicates count once; this is an explicit **chunking assumption**.
- Model aggregation (averaging leaf outputs or summing boosting contributions) is **not accounted for** in `B(x)`.

**Operational boundary for this assignment (declare as a hypothesis):** Label a model **"above the proposed 7-rule budget on most predictions"** if `R_7 > 0.50`. Also report `R_5` and `R_9` to show sensitivity. This is a **design choice**, not a scientifically established human interpretability threshold. If no model crosses the boundary, report that transparently rather than forcing one.

A useful visual: x-axis = CSC, y-axis = `R_5/R_7/R_9`, with separate markers for model families. Explicitly state what a low CSC with high `R_7` (or vice versa) means: the two measures capture different constructs.

### 9.3 Different threshold: minimum CSC for near-best accuracy

The assignment also emphasizes that interpretable models should make good predictions. Define a **performance-constrained selection rule**:

1. On **validation** data, find the highest balanced accuracy `A_best` among the candidate configurations for a given dataset.
2. Prespecify a tolerance, for example `delta = 0.02` (two absolute percentage points of balanced accuracy).
3. Among models with `A >= A_best - delta`, choose the one with **minimum CSC**.
4. Report its CSC, predictive performance, and `R_5/R_7/R_9`; then evaluate the frozen choice on untouched test data.

Call this a **performance–complexity operating point**, **not** an empirically established threshold of human cognition.

### 9.4 What you may and may not conclude

**You may conclude:** "Our 7-rule heuristic classified X% of predictions as over budget," "The lowest-CSC near-best-performing model was configuration Y," or "CSC separates models with similar depth but differing split diversity."

**You may not conclude without human validation:** "Participants understand models below CSC 10," "Miller proves our threshold," or "our predictions of human simulation accuracy are correct."

---

## 10. Execute Task 1 experiments and analyze them

**Run order:**

1. Pilot: Iris, one seed, 4 decision-tree depths. Inspect expected monotonic-ish path growth; **do not assume CSC always strictly increases with depth**.
2. Add random forest and gradient boosting on Iris; validate output counts, component values, and multiclass handling.
3. Add Breast Cancer Wisconsin; rerun identical configurations and seed(s).
4. Add controlled synthetic classification only after `scores.csv` and plots work for the first two datasets.
5. Repeat with additional seeds; summarize means/medians and variability, not only one run.

**Main Task 1 plots:**

- **Figure A:** CSC versus balanced accuracy (color/model family), **one panel or chart per dataset**. Identify low-CSC near-best-accuracy configurations.
- **Figure B:** Three CSC components across selected model configurations (stacked bars or grouped components).
- **Figure C:** CSC versus estimator count or maximum depth, including baselines (`TEPL`, total nodes) for comparison.
- **Figure D:** `R_5/R_7/R_9` versus CSC or model complexity with the proposed heuristic boundary clearly marked.

**Baseline questions:** Does CSC rank models differently from raw node count, tree count, or TEPL? Does it ever give counterintuitive rankings? Show at least one deliberate duplicate-tree or unbalanced-tree sanity example rather than claiming superiority without evidence.

**Robustness:** Compare results across seeds; examine sensitivity to threshold equality/float canonicalization; report where two components are strongly correlated or potentially double-count complexity.

---

## 11. Optional Task 2 — dataset meta-features and CSC

The professor explicitly labels Task 2 as **2 bonus points**, and the template contains **no existing code for Task 2**. Implement it only after Task 1 is reproducible.

**Candidate dataset meta-features:**

| Meta-feature | Definition / computation |
|---|---|
| Sample count | Rows in the original dataset |
| Raw feature count | Number of original predictors |
| Encoded feature count | Number of post-preprocessing columns used by trees |
| Class balance | Majority/minority class ratio (guard against zero counts) |
| Class count | Number of unique target labels |
| Feature redundancy | Mean absolute pairwise correlation for numeric inputs (document NaN handling) |
| Missingness | Fraction of missing feature cells |
| Synthetic noise | Generator's `flip_y` parameter (known by design) |
| Synthetic informative dimensions | Generator's `n_informative` parameter (known by design) |

**Recommended bonus experiment:** Use `make_classification` to vary **one parameter at a time** (e.g., feature count or label noise), with fixed `n_samples`, class count, generator seed, and training protocol. Train the same model grid, then compare the **minimum CSC achieving accuracy within `delta` of best validation performance**. This avoids confusing a universally harder dataset with a poorly tuned model.

For descriptive analyses, use scatter plots and Spearman rank correlation. With only a few datasets, do not infer broad causal effects or strong statistical significance. Compare like with like: the same family, similar performance, and consistent preprocessing.

Suggested optional file: `scripts/analyze_meta_features.py`. It should read the existing `results/scores.csv` or a well-documented metadata table, not create a separate irreproducible workflow.

---

## 12. Suggested execution milestones aligned to the deadline

The following is a **planning schedule**, not a statement that these actions have been completed.

| Date (2026) | Priority | Definition of done |
|---|---|---|
| **Nov 8** | Fork + run baseline + inspect APIs | Existing tests and starter experiment run successfully; source signatures recorded |
| **Nov 9** | Implement tree adapter + split/full-tree signatures | Hand-calculated stump, constant tree, unbalanced-tree tests pass |
| **Nov 10** | Implement forest + gradient boosting support | Adapter traversal tests pass; CSC works on all three families |
| **Nov 11** | Connect `evaluate()` + run Iris/Wisconsin | `results/scores.csv` contains actual CSC components and accuracy |
| **Nov 12** | Threshold + plots + robustness | `R_5/R_7/R_9`, near-best accuracy rule, and core figures reproducible |
| **Nov 13** | Optional Task 2 or XGBoost | Bonus synthetic meta-feature analysis **if** Task 1 is complete |
| **Nov 14** | Rewrite README + build/review presentation | Reproduction commands verified; `presentation/presentation.pdf` prepared |
| **Nov 15** | Final smoke test + submit **before 23:59** | Fresh environment/test run, code and PDF committed to fork |

**Time-saving fallback:** If the full grid is too large, prioritize two datasets, one seed, several depths, a small forest, and a small gradient-boosted ensemble. Real validated comparisons meet the spirit of Task 1 better than untested XGBoost support.

---

## 13. Five-minute presentation outline (required)

Keep the talk to roughly **five substantive slides** (plus an optional references slide), since the professor specifies **five minutes**, not a long seminar.

| Slide | Suggested emphasis | Approx. speaking time |
|---|---|---:|
| 1. Problem + definition | Why depth/tree count alone are weak; introduce CSC | 50 s |
| 2. Formula + intuition | `C_path + C_encoding + C_ensemble`; one small worked example | 70 s |
| 3. Experimental protocol | Model families, datasets, baselines, seeds, scoring | 45 s |
| 4. Findings + thresholds | Actual accuracy–CSC results; `R_5/R_7/R_9`; accuracy trade-off | 95 s |
| 5. Conclusions + limitations | What CSC improves, what remains unvalidated, optional Task 2 | 40 s |

Save the final exported artifact at **`presentation/presentation.pdf`**. Include actual experimental results; hypothetical tables are not a substitute. If Task 2 is done, place its strongest meta-feature plot on the findings slide or a short additional slide, without exceeding five minutes.

---

## 14. Rewriting the repository README: required final step

The professor explicitly requires replacing the **template README**. Do this **only after** the experiment is working so its instructions accurately describe your fork. Suggested structure:

```markdown
# Cognitive-Structural Complexity: iML Exercise 1

## Overview
Brief research question, which model families are implemented, and what CSC measures.

## Metric
CSC = mean expected path length + log2(1 + distinct split tests)
      + log2(distinct full-tree signatures).
Assumptions: structural proxy, not human-validated.

## Setup
uv sync --extra dev

## Reproduce
uv run pytest
uv run python scripts/run_experiment.py
# State any further real commands for optional figure generation.

## Datasets and models
List actual loaders, configurations, seeds, preprocessing, and source/version.

## Results
Summarize actual observed findings and link results/scores.csv and figures.

## Interpretability boundary
Define R5/R7/R9 and explain why 7±2 is only heuristic.

## Presentation
presentation/presentation.pdf

## Limitations and references
Explain model representation sensitivity and absence of human evaluation.
```

Avoid claiming that `uv run python scripts/run_experiment.py` regenerates a PDF or extra plots if the actual script does not do so. Include **precise, working** commands corresponding to your completed implementation. Preserve installability and any important existing project/license information.

---

## 15. Submission smoke-test and checklist

From a fresh clone or a clean local environment, run the commands **actually supported** by the repository:

```bash
uv sync --extra dev
uv run pytest
uv run python scripts/run_experiment.py
# If ruff is installed/configured in the fork:
uv run ruff check .
uv run ruff format --check .
```

Then inspect outputs and version-control state:

```bash
# Confirm the script generated this exact path:
ls results/scores.csv
# Confirm the exported presentation is in the required location:
ls presentation/presentation.pdf
# Check changed/untracked files before committing:
git status
```

**Final review:**

- [ ] The project is a **fork** of the provided repository, not a standalone unrelated package.
- [ ] Its Python package remains installable and the documented setup works.
- [ ] `InterpretabilityMetric` integration matches the *actual* template interface.
- [ ] CSC formula/components in code, README, and slides **agree exactly**.
- [ ] At least two datasets and several configurations have real recorded results.
- [ ] Single trees, forests, and gradient-boosted trees are correctly traversed (or unsupported families are candidly excluded).
- [ ] Stump/constant/duplicate/unbalanced tests and integration tests pass.
- [ ] `results/scores.csv` is produced by the required experiment entry point.
- [ ] Predictive quality is compared against CSC, not omitted.
- [ ] The threshold is an **operational hypothesis**, not a human-validated assertion.
- [ ] No unsupported LLM-based human-score claim is made.
- [ ] Task 2, if included, explicitly examines dataset meta-features.
- [ ] The template `README.md` is rewritten to explain the approach and reproduce results.
- [ ] **`presentation/presentation.pdf`** contains the actual five-minute presentation.
- [ ] Code and presentation are committed/submitted by **Nov 15, 2026, 23:59**.

---

## 16. Research validity and potential failure modes

1. **No validated human simulation:** CSC does not supply a human response model; Miller is motivation only.
2. **Representation versus function:** Two different ensembles can implement the same predictions. Conversely, adding duplicate boosted trees can change predictions without increasing unique structural signatures.
3. **Hybrid local/global construct:** `C_path` uses reference-distribution path occupancy, while `C_encoding` and `C_ensemble` use global stored structure. State this directly.
4. **Additive scaling assumption:** The three components have no empirically justified common cognitive unit.
5. **Correlated components:** New estimators often introduce both new splits and new tree signatures, potentially counting related complexity twice.
6. **Numeric precision:** Almost identical split thresholds may be counted as different under the exact policy.
7. **Class/output semantics:** Probability payloads and raw boosting contributions require distinct handling. Verify family-specific adapters.
8. **Sample dependence:** `C_path` and `R_b` depend on the reference input distribution; document which data split was used.
9. **No universal cutoff:** Neither `CSC = 7` nor `R_7 > 0.5` is a published cognitive threshold. The latter is merely the declared evaluation rule for this project.
10. **Novelty:** Expected path lengths, counts, entropy-like transformations, and representation complexity are established ideas. Claim only a **specific original construction** after checking prior literature.

If CSC does not outperform TEPL or tree size in interpretive usefulness, that is still a legitimate experimental result. Show the failure rather than retrofitting weights to force favorable rankings.

---

## 17. References and up-to-date technical documentation

**Primary assignment source (authoritative for deliverables):** Supervising professor's uploaded `README.md`, titled *iML 26/27 – Exercise 1: Quantifying Interpretability* (provided 8 Nov 2026). The README alone does not contain the actual repository Python interfaces or its dependency definitions.

**Documentation checked 8 Nov 2026:**

- uv: [Syncing optional dependencies and groups](https://docs.astral.sh/uv/concepts/projects/sync/), [Run commands in a project](https://docs.astral.sh/uv/concepts/projects/run/).
- scikit-learn: [Understanding the decision tree structure](https://scikit-learn.org/stable/auto_examples/tree/plot_unveil_tree_structure.html) (`apply`, `decision_path`, node arrays).
- scikit-learn: [GradientBoostingClassifier API](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.GradientBoostingClassifier.html) (`estimators_` layout; multiclass trees per stage).
- XGBoost: [Python API reference](https://xgboost.readthedocs.io/en/stable/python/python_api.html) (`get_dump`, `apply`, `best_iteration`). **Optional**, version-specific adapter.

**Cognition / threshold background:**

- Miller, G. A. (1956). *The magical number seven, plus or minus two: Some limits on our capacity for processing information*. **Psychological Review**, 63(2), 81–97.
- Cowan, N. (2001). *The magical number 4 in short-term memory: A reconsideration of mental storage capacity*. **Behavioral and Brain Sciences**, 24(1), 87–114. (Use as context that "7" is not a universal working-memory limit.)

**Prior-art search terms:** `interpretable tree complexity`, `expected decision path length`, `forest complexity metric`, `model description length`, `tree redundancy`, `minimum description length tree ensemble`. Distinguish *an original aggregation for this exercise* from an established universal interpretability metric.

---

## 18. One-page operational algorithm

```text
INPUT: a fitted supported tree-based model F and a declared feature matrix X_ref

1. Read the existing repository's actual metric and evaluation interfaces.
2. Extract all prediction trees used by F and normalize their nodes/outputs.
3. For each internal split, generate an exact canonical split_id.
4. For each tree, generate an exact full-tree signature.
5. S = number of unique split_id values across F.
6. U = number of unique full-tree signatures across F.
7. For every x in X_ref and every tree t:
       a. Traverse to the reached leaf.
       b. Record visited internal-node count d_t(x).
       c. Add split_ids along the path to the per-x visited-rule set.
8. C_path = mean(d_t(x)) across reference examples and trees.
9. C_encoding = log2(1 + S).
10. C_ensemble = log2(U).
11. CSC = C_path + C_encoding + C_ensemble.
12. For b in {5, 7, 9}: R_b = fraction of x with >b unique visited split_ids.
13. Compute baseline structural metrics and held-out predictive performance.
14. Write all components and metadata to results/scores.csv via run_experiment.py.
15. Analyze accuracy/complexity trade-offs and exploratory thresholds.
16. Rewrite the fork's README and export presentation/presentation.pdf.
17. Rerun the professor's documented commands and submit before deadline.

OUTPUT: reproducible Task 1 results (and optional Task 2 meta-feature analysis)
```

**Non-negotiable interpretation:** CSC is a testable **structural interpretability proxy**, not evidence that actual humans can remember or simulate particular models. The strength of the assignment is a sound metric, carefully executed comparisons, and transparent limits.
