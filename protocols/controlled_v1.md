# Controlled v1: locked before implementation and execution

Registered 8 November 2026. CSC formula remains version 1.0; reporting adds per-row details.

Q1: balanced versus chain trees with 7 nodes, 4 leaves, thresholds .25/.5/.75,
corresponding alternating class outputs, missing goes right. Occupancies are
25/25/25/25, 80/10/5/5, and 5/5/45/45. Expected chain path means 2.25, 1.30, 2.85;
balanced mean 2.0. Reference inputs are midpoints, with separate threshold boundary tests.

Q2: 12 feature-specific stumps; all 4096 {-1,+1} combinations. M=12;
K=1,2,3,4,6,12 equally repeated. Expected CSC=1+log2(1+K)+log2(K).
Replication control: three distinct stumps, repeated equally 1x/2x/4x.
Artificial averaging ensembles are not trained random forests.

Q3A: enumerate all 396 same-dataset/seed unordered validation pairs. Orient by model ID.
Accuracy match <=.02; near-best requires both within .02 of group maximum.
Float ranking equality tolerance 1e-10. Display up to three near-best pairs per group,
ordered by absolute accuracy difference then configuration IDs, without maximizing CSC differences.

Demands: reconstruct all 72 original fits with original seeds, split order and model grid.
Collect validation details for all candidates and test details only for the six frozen choices.
Reconcile all original numeric non-timing metrics at atol=rtol=1e-10; stop on mismatch.
Full integer exceedance curves; coverage budgets use minimum integer meeting coverage.
Scenario grid b=1..15; alpha=.10,.25,.50. No human capacity is inferred.

Main real-model demand plot uses the six frozen validation selections; all candidate curves
are retained as numeric data. Original test results are not used for pair or plot selection.

Optional Q3B: only after mandatory gates pass. Fixed outer five-fold stratified CV
(random_state=2026), inner three-fold (random_state=42+outer_fold), model seed=2026+outer_fold.
12 existing candidates; min CSC/nodes/TEPL among candidates within .02 of maximum mean
inner balanced accuracy, plus maximum accuracy selector. Lexicographic ties. Refit each
unique chosen configuration once on outer training; evaluate outer holdout once.
Stop at 30 minutes or any validation failure; report incomplete stage rather than changing grid.
No repeated search. These are already-inspected datasets, not independent confirmation.

All outputs under results/controlled_v1/. Preserve baseline scores and summary byte-for-byte.
Use source/protocol/data hashes, original row IDs and split manifests for provenance.
No human study, weights tuning, new family, Task 2 or claims of superiority.
