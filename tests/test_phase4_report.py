"""Regression checks on the Phase 4 write-up (reviewer fixes): framing, bound wording, pairing, margin story."""
import json
import math
import re
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

REPORT_MD = ROOT / "results" / "reports" / "phase4.md"
REPORT_JSON = ROOT / "data" / "cache" / "derivations" / "gpt-4.1-mini" / "phase4" / "report.json"


@pytest.fixture(scope="module")
def rep():
    if not REPORT_JSON.exists():
        pytest.skip("phase 4 report.json not built")
    return json.loads(REPORT_JSON.read_text())


def test_no_power_safety_framing_in_report_or_figure_code():
    md = REPORT_MD.read_text()
    src = (ROOT / "scripts" / "run_phase4.py").read_text()
    for bad in ("safety without power", "power without safety"):
        assert bad not in src
        # may only be quoted as the retracted framing
        for line in md.splitlines():
            if bad in line.lower():
                assert "retracted" in line.lower() or "not used" in line.lower(), line
    assert "dominat" in md.lower()


def test_bound_not_described_as_held():
    md = REPORT_MD.read_text().lower()
    assert "held empirically" not in md and "it held within" not in md
    assert "not rejected" in md and "beta-binomial" in md


def test_no_augmentation_weakly_dominates_shipped_on_pooled_test(rep):
    g = rep["groups"]
    for v in ("v1", "v2"):
        na, cal = g["noaug_inf"][f"pooled_{v}"], g["cal"][f"pooled_{v}"]
        assert na["fix_rate"] >= cal["fix_rate"] and na["scr"] <= cal["scr"]


def test_bound_tests_and_pairing(rep):
    bt = rep["bound_tests"]
    assert bt["hom_v1"]["beta_binomial"] < 0.05 and bt["hom_v2"]["beta_binomial"] < 0.05
    assert bt["pooled_v1"]["beta_binomial"] > 0.05 and bt["pooled_v2"]["beta_binomial"] > 0.05
    for g in bt.values():
        assert g["beta_binomial"] >= g["binomial"]
    cp = rep["calib_pairing"]
    assert cp["n_items"] == rep["n"] and cp["n_clusters"] < cp["n_items"]
    assert cp["spearman_rho"] > 0.5


def test_margin_one_items_are_two_candidate_pools(rep):
    for split in ("calib", "test"):
        ms = rep["margin_structure"][split]
        assert ms["n_margin_1_0"] == ms["n_margin_1_0_in_two_candidate_pools"]
        assert ms["share_raw_phoneme_prefers_heard"] > 0.9   # raw acoustics do separate the pair
    md = REPORT_MD.read_text()
    assert "only weakly" not in md and "0.2 z" not in md
