# Frozen-results post-processing audit

Validated 72 validation candidates and 6 frozen selected test evaluations.
Selection keys include dataset, seed and exact configuration; no test scores enter selection or charts.
Maximum validation CSC component-sum error: 0.

## Held-out test summary

Mean +/- sample SD over three overlapping splits; descriptive spread, not confidence intervals.

| Dataset | Balanced accuracy (%) +/- SD (pp) | CSC +/- SD |
|---|---:|---:|
| Iris | 96.44 +/- 3.09 | 6.10 +/- 2.45 |
| Breast Cancer | 92.45 +/- 0.39 | 4.59 +/- 1.03 |

## Hypothetical budget sensitivity

42/72 tested validation configurations (58.3%) satisfy strict R7 > 0.50.

| Dataset | R5 > 0.50 | R7 > 0.50 | R9 > 0.50 |
|---|---:|---:|---:|
| Iris | 66.7% | 58.3% | 47.2% |
| Breast Cancer | 63.9% | 58.3% | 50.0% |

## Evidence limits

CSC is structural, not human-tested. Path length uses reference-data occupancy; split and tree counts describe global structure. The logarithmic distinct-split count is not an MDL bit code. Full signatures are representation-dependent, not a test of functional equivalence. Repeating an identical boosted tree can alter predictions without altering CSC. Rule budgets omit output aggregation arithmetic. Strong correlation with node count does not establish superiority. Two datasets and three overlapping splits limit generalization.

## Frozen source

scores.csv SHA-256: `5ff110ebff0f8c50227fa4349e771070e32e96c7a3059696a0314dcdeca1d0df`

The plotting command checks that scores.csv and the original summary.json remain byte-for-byte unchanged.
