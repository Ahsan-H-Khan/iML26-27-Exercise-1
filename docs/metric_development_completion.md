# Metric-development v2 completion audit

Completed 14 November 2026.

- EPL/EEPC total/mean, used-feature burden, EA and EA_total implemented through shared tree traversal; CSC formula, original experiment code and frozen result hashes unchanged.
- Original 78 result rows reconciled with zero numeric discrepancy at saved precision.
- Training-fitted dummy baselines evaluated; 72 validation candidates, 35 competitive candidates, 12 frozen selector decisions, 9 distinct test rows.
- Validation-only selection, no-eligible handling, deterministic ties, selector coincidence, split integrity, path counts and feature reuse covered by tests. All 36 tests passed.
- Full project lint checks passed.
- Primary figures/report/README and 12 main slides plus 2 appendices use metric_development_v2 outputs, not controlled-v1 tables.
- All 14 PDF pages visually inspected. Two clipped axis labels were corrected and the affected final pages rechecked. PDF page count matches the explicit slide manifest.
- Main speaker-note target: 860 seconds (14.3 minutes), not a measured rehearsal.
- Prior work archived. No GitHub or course submission performed.

The evaluation remains descriptive on previously inspected benchmark splits. Formal MDL and human interpretability validation were not implemented or claimed. On Breast Cancer, minimum-CSC selection reduces mean test CSC but also lowers mean test balanced accuracy relative to accuracy-only selection. Iris selectors coincide.
