# Sources and attribution

Verified on 14 November 2026.

- Peter Grünwald, *A Tutorial Introduction to the Minimum Description Length Principle*, sections 1.3–1.4: https://homepages.cwi.nl/~pdg/ftp/mdlintro.pdf . Supports the introductory two-part model-plus-data code explanation and the need to specify codes.
- scikit-learn, `DecisionTreeClassifier.decision_path`: https://scikit-learn.org/stable/modules/generated/sklearn.tree.DecisionTreeClassifier.html . Supports the visited-node indicator used for an independent path-count check; internal splits equal visited nodes minus the terminal leaf.

EPL is an empirical expectation of internal path counts as defined in this project. EEPC total/mean, the EA feature-burden convention, and CSC's additive combination are explicitly project definitions; this document does not attribute those formulas to the above references. CSC is MDL-inspired, not a formal code length. No numerical formal-MDL baseline was implemented. The chronological presentation explains design motivation rather than proving successive proposals improve human interpretability.
