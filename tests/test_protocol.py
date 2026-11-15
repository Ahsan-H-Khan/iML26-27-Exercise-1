import numpy as np
from sklearn.tree import DecisionTreeClassifier

from iml_interpretability import CognitiveStructuralComplexity, evaluate
from iml_interpretability.data import load_benchmarks
from iml_interpretability.experiment import run_protocol


def test_protocol_selection_and_reproducibility(monkeypatch):
    monkeypatch.setattr(
        "iml_interpretability.experiment.get_models",
        lambda seed: {
            "stump": DecisionTreeClassifier(max_depth=1, random_state=seed),
            "depth3": DecisionTreeClassifier(max_depth=3, random_state=seed),
        },
    )
    datasets = load_benchmarks()[:1]
    first = run_protocol(datasets, seeds=(42,))
    second = run_protocol(datasets, seeds=(42,))
    validation = first[first.score_split == "validation"]
    test = first[first.score_split == "test"]
    assert len(validation) == 2 and len(test) == 1
    eligible = validation[validation.balanced_accuracy >= validation.balanced_accuracy.max() - 0.02]
    assert test.iloc[0].model == eligible.sort_values(["csc", "model"]).iloc[0].model
    assert first.iloc[0].n_train + first.iloc[0].n_validation + first.iloc[0].n_test == 150
    np.testing.assert_allclose(first.csc, second.csc)
    np.testing.assert_allclose(first.balanced_accuracy, second.balanced_accuracy)


def test_original_evaluate_supports_csc():
    result = evaluate(
        load_benchmarks()[:1], {"stump": DecisionTreeClassifier(max_depth=1)}, [CognitiveStructuralComplexity()]
    )
    assert result.iloc[0].csc == 2
