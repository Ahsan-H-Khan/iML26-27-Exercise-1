# Controlled extension completion audit

## Scope

Stages 0–5 implemented. Optional stage 6 (nested selector comparison) was not run; no selector-generalization claim is made. Original study snapshot is preserved separately in the workspace backup; frozen source/result hashes are in `protocols/baseline_hashes.json`.

## Checks

- Exact Q1 topology, size, predictions and occupancy expectations checked.
- Q2 fixed M/nodes/TEPL and distinct-logic expectations checked; replication probability invariance limited to averaging.
- Every one of 396 same-split pairs retained; display rule deterministic; 218 matched and 116 near-best pairs.
- All 78 original rows and six selection keys reconstructed; original CSV and summary hashes unchanged.
- B bounds, mean L = TEPL, legacy strict R5/R7/R9, monotone demand curves, curve area = mean B, and integer coverage budgets checked.
- 28 unit tests cover original behavior and new controls, boundaries, quantiles and pair matching.
- Five-slide PDF rebuilt with 300-second target notes; pages visually reviewed.

## Evidence interpretation

Controls establish measurement behavior, not human validity. Q1 is already detected by TEPL. Q2 changes predictions in the diversity sweep but preserves them in equal replication. Real validation rankings mostly agree. Coverage budgets summarize structural demand; they do not estimate memory capacity. Aggregation effort is not included. No new test-based tuning occurred.

## Output map

`results/controlled_v1/report.md` is the primary report. CSV files contain full controls, pair differences, per-example demands, summary quantiles, curves and hypothetical scenarios. Manifest, dataset hashes and split IDs record provenance. All main plots have PNG/PDF/SVG variants. The original plots remain supporting material.
