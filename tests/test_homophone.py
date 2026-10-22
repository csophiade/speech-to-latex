"""Homophone stress set: swap sites, and ground truth of every item in data/derivations/homophone.json."""

import json
from collections import Counter

import pytest

from src.config import DATA_DIR
from src.derivgen import domain_of, solution_set
from src.homophone import CONFUSION, check, swap, swap_sites

HOM = DATA_DIR / "derivations" / "homophone.json"


def test_confusion_map():
    assert CONFUSION[15] == 50 and CONFUSION[50] == 15 and CONFUSION[13] == 30 and CONFUSION[90] == 19
    assert len(CONFUSION) == 14


def test_swap_sites_whole_numbers_only():
    vals = lambda s: [v for *_, v in swap_sites(s)]  # noqa: E731
    assert vals("2x + 15 = 50") == [15, 50]
    assert vals("x = -\\frac{13}{4}") == [13]
    assert vals("x = 150") == [] and vals("x = 1.5") == [] and vals("x = 15.2") == []
    assert vals("x_{13} = 2") == [] and vals("x^{14} = 1") == [] and vals("x_13 = 2") == []
    assert vals("x = 3") == []
    assert swap("2x + 15 = 50", swap_sites("2x + 15 = 50")[0]) == "2x + 50 = 50"


@pytest.fixture(scope="module")
def items():
    if not HOM.exists():
        pytest.skip("homophone.json not generated (scripts/gen_homophone.py)")
    return json.loads(HOM.read_text())


def test_counts_and_ids(items):
    splits = Counter(it["split"] for it in items)
    assert 90 <= splits["calib"] <= 110 and 90 <= splits["test"] <= 110
    assert all(it["id"].startswith("hom-") and it["set"] == "homophone" for it in items)
    assert len({it["id"] for it in items}) == len(items)
    old0 = {it["lines"][0] for it in json.loads((DATA_DIR / "derivations" / "derivations.json").read_text())}
    assert not old0 & {it["lines"][0] for it in items}


def test_labels(items):
    for it in items:
        t = it["error"]["line"]
        assert it["error"] == {"type": "homophone", "line": t} and t >= 1 and len(it["lines"]) == t + 1
        assert it["step_label"] == ["given"] + ["valid"] * (t - 1) + ["error"]
        assert it["step_valid"] == [True] * t + [False]


def test_said_number_is_confusion_partner(items):
    for it in items:
        h = it["homophone"]
        assert h["said"] == CONFUSION[h["correct"]]
        assert any(swap(h["correct_line"], s) == it["lines"][-1]
                   for s in swap_sites(h["correct_line"]) if s[2] == h["correct"])


def test_wrong_step_and_original_valid(items):
    """Line t (as said) changes the solution set; line t-1 -> original line t keeps it (SymPy solveset)."""
    for it in items:
        check(it)
        lines, h = it["lines"], it["homophone"]
        dom = domain_of(lines[0])
        assert solution_set(lines[-1], dom) != solution_set(lines[-2], dom)
        assert solution_set(h["correct_line"], dom) == solution_set(lines[-2], dom)
