# Threshold investigation

These are hypothetical rule-demand boundaries, not measured human limits.
R_b is the fraction of validation examples with strictly more than b distinct encountered tests. A flag means R_b > 50%. CSC=7 does not mean seven tests.

All 72 validation candidates are retained; eligible-only results are also saved. Seed groups have 12 candidates each. Reference-example denominators are explicit in threshold_summary.csv.

## Across the three seeds (descriptive)

- breast_cancer, budget 5: 23/36 candidates flagged; an increasing CSC cutoff separates these observed groups: False.
- breast_cancer, budget 7: 21/36 candidates flagged; an increasing CSC cutoff separates these observed groups: False.
- breast_cancer, budget 9: 18/36 candidates flagged; an increasing CSC cutoff separates these observed groups: False.
- iris, budget 5: 24/36 candidates flagged; an increasing CSC cutoff separates these observed groups: True.
- iris, budget 7: 21/36 candidates flagged; an increasing CSC cutoff separates these observed groups: False.
- iris, budget 9: 17/36 candidates flagged; an increasing CSC cutoff separates these observed groups: False.

Separation in this finite grid is not evidence of a universal cutoff. When ranges overlap, a single increasing cutoff cannot reproduce the rule-budget flags. Empty groups are unassessable. Both measures share model structure, so association is not independent human validation.

The analysis reads the original R5/R7/R9 measurements; the meta-feature original-condition reruns independently reconcile them. No artificial controlled-study outputs are used.