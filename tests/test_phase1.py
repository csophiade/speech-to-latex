import json

import pytest

from src import config


def latest(system, dataset):
    if not config.RESULTS_JSON.exists():
        return None
    rows = [r for r in json.loads(config.RESULTS_JSON.read_text())
            if r["system"] == system and r["dataset"] == dataset]
    return rows[-1] if rows else None


def test_system_a_cer_in_sane_band():
    """If A's CER on en_test_500 is outside 0.1-0.6, normalisation, column or split is broken."""
    row = latest("A", "s2l_test500")
    if row is None:
        pytest.skip("run scripts/run_baseline.py --system A --dataset s2l_test500 first")
    assert row["n"] == 500
    assert 0.1 <= row["metrics"]["cer"] <= 0.6, row["metrics"]


@pytest.mark.parametrize("system", ["A", "B"])
@pytest.mark.parametrize("dataset,n", [("s2l_test500", 500), ("mathspeech", 1101)])
def test_results_logged(system, dataset, n):
    row = latest(system, dataset)
    if row is None:
        pytest.skip("baseline not run yet")
    for key in ("system", "dataset", "metrics", "n", "timestamp", "git"):
        assert key in row
    assert row["n"] == n
