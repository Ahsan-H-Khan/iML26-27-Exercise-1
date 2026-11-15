# Five-minute presentation

Target timing: 40 + 55 + 55 + 75 + 75 = 300 seconds. Approximately 618 words including headings; rehearse aloud and adjust pauses to your speaking pace. This is a timing estimate, not a measured rehearsal.

## Slide 1 — How much logic is inside a model? (40 seconds)

A decision tree makes a prediction by asking a sequence of questions about its input. CSC, short for Cognitive-Structural Complexity, is our proposed score for how much logic a tree model contains. It adds the average number of questions encountered per tree and example, log2 of one plus the number of distinct split tests, and log2 of the number of distinct full trees. Those weights are assumptions. A smaller score means less measured structural complexity, not proven ease of understanding. We test its behavior using constructed examples and compare models trained on Iris and Breast Cancer.

## Slide 2 — The same model size can hide different paths (55 seconds)

Here we hold predictions and size fixed: both constructed trees have seven nodes, four leaves and the same classification function. One tree is balanced, so every prediction takes two questions. The chain-shaped tree can finish after one question or require three. We change which examples arrive. When most exit early, chain CSC is 3.30; when most go deeper, it rises to 4.85. The balanced tree stays at four. The uniform-reference case, 4.25 for the chain, is in the full report. Counting nodes misses this difference, although average path length also captures it. Lower CSC is the direction of lower measured complexity.

## Slide 3 — Repeating a rule adds no new rule (55 seconds)

We keep twelve one-question trees, called stumps, and thirty-six total nodes. Every example still visits twelve split nodes. Only the number of distinct rules changes. When all twelve trees repeat one rule, CSC is two. When all twelve use different rules, CSC is 8.29. This controlled setup shows what simple node counts miss. These are artificial averaging ensembles, and their predictions can change as rule diversity changes. A separate control repeats each of three base trees equally: its probabilities and CSC stay fixed while nodes and visits grow. That invariance applies to averaging, not arbitrary boosting duplication.

## Slide 4 — The measures sometimes disagree (75 seconds)

This is an illustrative disagreement selected after looking at the saved comparisons, not a representative sample. Both models use the same Breast Cancer validation split, seed 42. Their balanced accuracy is identical at 91.1 percent and both are within two percentage points of the best candidate. Gradient boosting uses fifty depth-one trees; random forest uses five trees with maximum depth five. Node count favors the forest, while CSC slightly favors boosting. Total expected path length also favors the forest. CSC weights average paths and distinct logic differently from raw size. Across all 116 near-best pairs, CSC reverses node-count rankings only three times and TEPL rankings seven times. These pairs overlap and are not independent experiments. We have not established which ranking better matches human understanding.

## Slide 5 — How many different tests does a prediction need? (75 seconds)

Each bar shows the smallest number of distinct split tests sufficient for at least ninety percent of validation examples, for one frozen selected model. Run one, two and three correspond to seeds 42, 1337 and 2026. For Breast Cancer the selected models need two, two and four tests. For Iris they need two, ten and nine. Runs can select different models, which explains part of that variation: Iris selects a single tree first, then two five-tree forests. These are measurements of model demand, not limits on human memory. The ninety-five-percent values happen to be the same in these six cases. Full exceedance curves, all visits and tree counts are retained in the report. Output aggregation effort is still omitted. We can describe structural differences; we cannot yet say how hard people will find these models.

## Sources and reproduction

Sources: unchanged results/controlled_v1 CSV tables and summary.json. Q3 uses the saved Breast Cancer seed-42 gb_n50_d1 / rf_n5_d5 pair, deliberately chosen to explain a disagreement. All pairs remain in the report. Run labels 1/2/3 map to seeds 42/1337/2026. Optional nested evaluation was not run.

Figures and displayed statistics are loaded from CSV-derived reports. Rebuild previews with scripts/build_presentation.mjs before repackaging with scripts/build_pdf.py. The PDF contains rendered slide images; vector charts are also available separately.
