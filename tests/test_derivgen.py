"""Phase 3.1: derivation generator ground truth."""

import json
from collections import Counter

import pytest

from src.config import DATA_DIR

from src.derivgen import ERROR_TYPES, TYPES, domain_of, generate, parse_line, solution_set, step_labels


@pytest.fixture(scope="module")
def items():
    return generate()


def test_deterministic():
    assert generate(per_type=10) == generate(per_type=10)
    assert generate(seed=1, per_type=10) != generate(per_type=10)


def test_counts_and_error_rate(items):
    assert len(items) == 400 and len({i["id"] for i in items}) == 400
    assert Counter(i["type"] for i in items) == {t: 80 for t in TYPES}
    assert sum(bool(i["error"]) for i in items) == 120   # exactly 30%
    assert Counter(i["error"]["type"] for i in items if i["error"]) == {e: 20 for e in ERROR_TYPES}
    for split in ("calib", "test"):
        part = [i for i in items if i["split"] == split]
        assert len(part) == 200 and sum(bool(i["error"]) for i in part) == 60
        assert Counter(i["type"] for i in part) == {t: 40 for t in TYPES}


def test_error_type_applicability(items):
    for i in items:
        e = i["error"] and i["error"]["type"]
        if e == "missed_flip":
            assert i["type"] == "inequality"
        if e == "divide_by_variable":
            assert i["type"].startswith("quadratic")
        if e == "square_both_sides":
            assert "^2" in i["lines"][i["error"]["line"]]


def test_schema(items):
    for i in items:
        assert 4 <= len(i["lines"]) <= 6 and len(i["step_valid"]) == len(i["lines"])
        assert i["step_valid"][0]
        bad = [k for k, v in enumerate(i["step_valid"]) if not v]
        assert bad == ([i["error"]["line"]] if i["error"] else [])
        for s in i["lines"] + [i["solution"]]:
            parse_line(s)


def test_step_label(items):
    """given / valid / error / follows_error, consistent with step_valid and error.line."""
    assert step_labels(5, None) == ["given"] + ["valid"] * 4
    assert step_labels(6, 2) == ["given", "valid", "error", "follows_error", "follows_error", "follows_error"]
    for i in items:
        lab = i["step_label"]
        assert len(lab) == len(i["lines"]) and lab[0] == "given"
        assert [k for k, v in enumerate(lab) if v == "error"] == [k for k, v in enumerate(i["step_valid"]) if not v]
        if i["error"]:
            assert all(v == "follows_error" for v in lab[i["error"]["line"] + 1:])
        else:
            assert "follows_error" not in lab


def test_saved_dataset_matches_generator(items):
    """derivations.json (from which the audio was synthesised) equals generate() field for field."""
    saved = json.loads((DATA_DIR / "derivations" / "derivations.json").read_text())
    assert saved == items


def test_steps_match_solution_sets(items):
    """Valid steps keep the solution set (on the original domain); the error step changes it."""
    for i in items:
        dom = domain_of(i["lines"][0])
        sets = [solution_set(s, dom) for s in i["lines"]]
        for k in range(1, len(sets)):
            assert (sets[k] == sets[k - 1]) == i["step_valid"][k], (i["id"], k)
        assert solution_set(i["solution"], dom) == sets[0], i["id"]
        if not i["error"]:
            assert i["lines"][-1] == i["solution"]
        else:
            assert sets[-1] != sets[0]


def test_solution_convention():
    assert solution_set(r"x = 2 \text{ or } x = 5") == solution_set(r"x^2 - 7x + 10 = 0")
    assert solution_set(r"x > 4") == solution_set(r"-2x < -8")
    assert solution_set("x = 2", domain_of(r"\frac{1}{x - 2} = 1")).is_empty
