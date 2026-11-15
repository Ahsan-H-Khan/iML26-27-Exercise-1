# Five-minute presentation

Target timing: 40 + 55 + 55 + 75 + 75 = 300 seconds. Approximately 571 words including headings; rehearse aloud and adjust pauses to your speaking pace. This is a timing estimate, not a measured rehearsal.

## Slide 1 — What does CSC add beyond model size? (40 seconds)

CSC combines average path length, distinct split tests and distinct full trees. It is a structural proxy, not a measured cognitive cost. We preserve its formula and original study: two datasets, three seeds, twelve candidates per group, and six frozen test selections. This extension asks three questions using controlled fixtures and same-split validation comparisons. The original test data are not used to choose the new story. All 78 saved result rows were reconstructed exactly.

## Slide 2 — Q1 · Equal size and predictions; different paths (55 seconds)

Both constructed trees have seven nodes, four leaves and exactly the same binary prediction function. Their balanced and chain representations encounter the same three thresholds in different orders. Uniform, early-exit and deep-exit reference distributions isolate occupancy. The balanced tree has CSC four throughout. The chain scores 4.25, 3.30 and 4.85, so which representation is simpler depends on where examples fall. Node count cannot distinguish these cases. Total expected path length already detects the path difference. This establishes representation and reference dependence, not superiority to every baseline.

## Slide 3 — Q2 · Fixed size; more distinct logic increases CSC (55 seconds)

These artificial ensembles always contain twelve stumps and thirty-six nodes. Every example visits twelve split nodes, so total expected path length stays twelve. We vary the number of distinct stumps from one to twelve. CSC rises from two to 8.29 as distinct split and tree signatures increase. This isolates a distinction that raw size and traversal totals miss. Unlike the first control, predictive functions are not held fixed in this diversity sweep. The separate replication control averages three unique stumps repeated equally. Its probabilities and CSC remain unchanged as raw nodes and visits increase. That prediction-invariance claim applies to equal-weight averaging, not arbitrary boosting duplication.

## Slide 4 — Q3 · Near-best models mostly agree with simpler baselines (75 seconds)

We enumerate all 396 unordered pairs within the same dataset and seed. 218 pairs differ by at most two percentage points in validation balanced accuracy. For 116, both models are also within two points of their group best. The displayed example is the first deterministically selected pair from each group; all eligible pairs and selection rules remain in the report. Among near-best pairs, CSC reverses node-count rankings three times and TEPL rankings seven times. One pair ties on both measures. This largely confirms agreement on this small grid. These overlapping pairs are dependent observations, not hundreds of independent experiments. Different rankings do not establish which metric better predicts human understanding.

## Slide 5 — Rule demand replaces an assumed human cutoff (75 seconds)

B counts distinct split tests encountered by one example; L counts all split-node visits, including repeated tests, and M counts prediction trees. The curve shows the fraction of reference examples whose B exceeds each integer budget. A coverage budget b90 is the smallest integer covering at least ninety percent of examples, and b95 covers ninety-five percent. For selected Iris models, both budgets are two, ten and nine across the three seeds. Breast Cancer gives two, two and four. These describe reference-data demand, not human capacity. Original aggregate metrics reconcile exactly for all 78 rows. The five-, seven- and nine-rule scenarios remain secondary sensitivity analyses. Neither seven rules nor a fifty-percent exceedance boundary is validated cognitively. Aggregation effort remains unmodeled; controlled structural behavior is the defensible contribution.

## Sources and reproduction

Sources: results/controlled_v1 tables, summary.json and report.md; protocols/controlled_v1.md. Original scores and summary preserved. Optional nested selector evaluation was not run.

Figures and displayed statistics are loaded from CSV-derived reports. Rebuild previews with scripts/build_presentation.mjs before repackaging with scripts/build_pdf.py. The PDF contains rendered slide images; vector charts are also available separately.
