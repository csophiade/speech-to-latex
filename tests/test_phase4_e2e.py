"""End-to-end derivation mode on real cached clips: frozen System D + checker + decide at the calibrated δ.

Uses the stage-1 pools cached by scripts/run_derivations.py (augmented: confusable-number variants are
in every v1/v2/rec/homophone pool; no GPU or LLM calls); skipped if absent. The calibrated δ is the
pooled one (first-pass + homophone calibration errors), finite since the second Phase 4 pass.
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
    clips = load_clips(("v1", "v2"), homophone=True)
    delta = json.loads(CALIB.read_text())["delta"]
    delta = math.inf if delta == "inf" else delta

    def go(source, id_, at=None):
        lines = sorted((c for c in clips if c["source"] == source and c["id"] == id_), key=lambda c: c["line"])
        if not all(clip_key(c) in done for c in lines):
            pytest.skip(f"{source}/{id_} not in the stage-1 cache")
        return run_derivation((lines, [done[clip_key(c)] for c in lines], delta if at is None else at, frozen_d()))
    go.delta = delta
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


def test_mishearing_corrected_only_without_gate(run):
    # heard 4(x - 25), said 4x - 25; the valid reading sits at Delta = 1.0 (rank prior): corrected with no
    # gate (δ = +inf, the first-pass system), refused at the calibrated δ, which then cascades into a flag
    # on the next (correct) line because it is checked against the misheard output.
    assert 0.5 < run.delta < 1.0
    ungated = run("v1", "lin-0018", at=math.inf)
    assert ungated[1]["status"] == "corrected" and ungated[1]["out_eq_said"]
    recs = run("v1", "lin-0018")
    assert recs[1]["status"] == "math_error" and not recs[1]["out_eq_said"]
    assert recs[2]["label"] == "valid" and recs[2]["status"] == "math_error"


def test_homophone_error_not_silently_fixed(run):
    # said "five x plus fifty plus eight" where 15 is right: augmentation puts the valid 5x + 15 + 8 in the
    # pool; with no gate it would silently "fix" the speaker's mistake, at the calibrated δ it is flagged.
    ungated = run("v1", "hom-0138", at=math.inf)
    assert ungated[1]["status"] == "corrected" and not ungated[1]["out_eq_said"]
    recs = run("v1", "hom-0138")
    assert [r["label"] for r in recs] == ["given", "error"]
    hom = [c for c in recs[1]["cands"] if c["valid"] and not c["eq_said"]]
    assert hom and all(c["augmented"] for c in hom) and min(c["delta"] for c in hom) >= run.delta
    assert recs[1]["status"] == "math_error" and recs[1]["out_eq_said"]
    assert recs[1]["output"] == "5x + 50 + 8 = x + 20"
