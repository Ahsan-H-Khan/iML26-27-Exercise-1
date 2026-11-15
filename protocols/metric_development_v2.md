# Metric development v2: locked protocol

Locked before running the v2 evaluation, 14 November 2026.

- Original datasets, 12 configurations, seeds 42/1337/2026 and stratified 50/25/25 splits retained. No fitted preprocessing.
- EPL counts internal splits; EEPC_total sums tree EPL; EEPC_mean divides by actual tree count, including multiclass prediction trees.
- F_used counts distinct feature positions in internal nodes, excluding unused inputs. EA = EEPC_mean * log2(1+F_used). EA_total is a separately labelled sensitivity, not a selected alternative.
- CSC remains c_path + log2(1+S) + log2(U), unchanged CSC-1.0. MDL-inspired, not formal code length.
- DummyClassifier(prior) fitted on training partition. Competitive = validation balanced accuracy strictly above dummy AND at least best minus 0.02 within dataset/seed.
- CSC selector: competitive minimum CSC, ties by model ID. Accuracy selector: maximum validation balanced accuracy over all candidates, ties by model ID. If no competitive candidate, CSC selector is absent, with explicit status.
- Freeze both choices before test scoring. Evaluate each distinct selected model once, without refitting. Previously inspected test data are reused; this is descriptive follow-up, not fresh confirmation.
- Reconcile all old validation and selected test metrics with tolerance 1e-10; exclude timing fields. Fail before publishing tables if reconciliation fails.
- Primary metric comparisons: within dataset/seed Spearman ranks among competitive candidates; constant metrics return undefined, not zero. No pair counts or p-values.
- Worked real-model example: Iris seed 42 minimum-CSC choice. Component figure includes all six CSC choices. No controlled-v1 input files.
- Original result/source hashes preserved in metric_development_v2_hashes.json. New results only in results/metric_development_v2.
