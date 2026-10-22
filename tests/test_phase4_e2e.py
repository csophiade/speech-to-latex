"""End-to-end derivation mode on real cached clips: frozen System D + checker + decide at the calibrated δ.

Uses the stage-1 pools cached by scripts/run_derivations.py (no GPU or LLM calls); skipped if absent.
"""

import json
import math
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from run_derivations import cache_dir, clip_key, frozen_d, load_clips, read_cache, run_derivation  # noqa: E402

from src import config  # noqa: E402

STAGE1 = cache_dir(config.LLM_MODEL) / "stage1.jsonl"
CALIB = cache_dir(config.LLM_MODEL) / "phase4" / "calibration.json"
pytestmark = pytest.mark.skipif(not (STAGE1.exists() and CALIB.exists()), reason="needs cached stage-1 pools")


@pytest.fixture(scope="module")
def run():
    done = read_cache(STAGE1)
    clips = load_clips(("v1", "v2"))
    delta = json.loads(CALIB.read_text())["delta"]
    delta = math.inf if delta == "inf" else delta

    def go(source, id_):
        lines = sorted((c for c in clips if c["source"] == source and c["id"] == id_), key=lambda c: c["line"])
        return run_derivation((lines, [done[clip_key(c)] for c in lines], delta, frozen_d()))
    return go


def statuses(records):
    return [r["status"] for r in records]


def test_clean_derivation_all_verified(run):
    recs = run("v1", "lin-0000")
    assert statuses(recs) == ["given"] + ["verified"] * 4
    assert all(r["out_eq_said"] for r in recs)


def test_injected_error_flagged_then_followed(run):
    recs = run("v2", "lin-0004")
    assert [r["label"] for r in recs] == ["given", "error"] + ["follows_error"] * 3
    assert statuses(recs) == ["given", "math_error"] + ["verified"] * 3
    assert "dropped" in recs[1]["explanation"]
    assert recs[1]["output"] == recs[1]["heard"]          # the user's own (wrong) line is kept


def test_recording_error_flagged(run):
    recs = run("rec", "d025")
    assert statuses(recs) == ["given", "math_error", "verified"]
    assert recs[1]["out_eq_said"]                         # flagged, never silently rewritten


def test_mishearing_corrected(run):
    rec = run("v1", "lin-0018")[1]                        # heard 4(x - 25), said 4x - 25
    assert rec["status"] == "corrected" and rec["heard"] != rec["output"] and rec["out_eq_said"]
