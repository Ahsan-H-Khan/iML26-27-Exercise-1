from pathlib import Path

import pandas as pd
import pytest

from iml_interpretability.metric_development import choose_candidates


def test_selection_is_per_condition_and_can_be_empty():
    rows = [dict(model="a", balanced_accuracy=0.5, csc=1), dict(model="b", balanced_accuracy=0.49, csc=0.5)]
    assert choose_candidates(rows, 0.5) == {"accuracy_only": "a"}
    assert not any(r["eligible"] for r in rows)
    rows = [dict(model="a", balanced_accuracy=0.9, csc=5), dict(model="b", balanced_accuracy=0.88, csc=2)]
    assert choose_candidates(rows, 0.5)["minimum_csc"] == "b"


def test_saved_study_covers_all_groups_without_test_scores():
    path = Path("results/metafeatures_v1/candidates.csv")
    if not path.exists():
        pytest.skip("Run study to verify its output")
    df = pd.read_csv(path)
    assert df.groupby(["dataset", "seed", "condition"]).size().eq(12).all()
    assert len(df.groupby(["dataset", "seed"])) == 6
    assert set(df.score_split) == {"validation"}
    assert df[df.condition == "original"].shape[0] == 72
    for _, group in df.groupby(["dataset", "seed", "condition"]):
        best = group.balanced_accuracy.max()
        expected = (group.balanced_accuracy > group.dummy_balanced_accuracy) & (group.balanced_accuracy >= best - 0.02)
        assert (expected == group.eligible).all()
        assert group.selected_csc.sum() == int(expected.any())
