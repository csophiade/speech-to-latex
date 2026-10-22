"""Live demo pipeline (src/pipeline.py) with stubbed ASR, LLM pool and acoustic scores: offline, no GPU."""

import threading

import numpy as np
import pytest

from src import tts_live
from src.augment import augment_pool
from src.pipeline import LinePipeline, spoken_explanation

AUDIO = np.zeros(16_000, dtype=np.float32)


def stub(latexes, phon):
    """A pipeline whose pool is `latexes` (+ confusable variants) with phoneme score phon[latex]."""
    def build_pool(nbest, llm, workers, ctx):
        pool = [{"latex": x, "level": None, "src_rank": 0, "src_logprob": -0.1, "cand_rank": j,
                 "spoken": [x]} for j, x in enumerate(latexes)]
        pool += augment_pool(pool)
        return pool, {"llm_top_s": 0.0, "llm_latex_s": 0.0, "llm_spoken_s": 0.0}

    def score_pool(audio, pool):
        for p in pool:
            p["whisper"] = [0.0] * len(p["spoken"])
            p["phoneme"] = [phon.get(p["latex"], -10.0)] * len(p["spoken"])
        return {"whisper_s": 0.0, "phoneme_s": 0.0}

    def make(**kw):
        return LinePipeline(asr=lambda a: [("heard", -0.1)], build_pool=build_pool,
                            score_pool=score_pool, warm=False, **kw)
    return make


HIST = [{"output": "x^2 + 5x = 50"}]
PLANTED = ["x^2 + 5x - 15 = 0"]


def test_line0_given_and_verified_step():
    pipe = stub(["x^2 + 5x - 50 = 0"], {"x^2 + 5x - 50 = 0": -1.0})()
    assert pipe.process_line(AUDIO, 16_000, [])["status"] == "given"
    r = pipe.process_line(AUDIO, 16_000, HIST)
    assert r["status"] == "verified" and r["output"] == "x^2 + 5x - 50 = 0"
    assert set(r["latency_parts"]) >= {"asr_s", "llm_s", "phoneme_s", "checker_s", "decide_s"}


def test_homophone_threshold_routes_the_correction():
    """Same pool: the fifty reading is a homophone-class candidate, so only delta_hom decides."""
    make = stub(PLANTED, {"x^2 + 5x - 15 = 0": -1.0, "x^2 + 5x - 50 = 0": -1.2})
    r = make(delta=0.0, delta_hom=100.0).process_line(AUDIO, 16_000, HIST)
    assert r["status"] == "corrected" and r["output"] == "x^2 + 5x - 50 = 0" and r["heard"] == PLANTED[0]
    assert "heard fifteen and wrote fifty" in r["spoken_explanation"]
    r = make(delta=100.0, delta_hom=0.0).process_line(AUDIO, 16_000, HIST)
    assert r["status"] == "math_error" and r["output"] == PLANTED[0]


def test_red_flag_names_the_sound_alike():
    r = stub(PLANTED, {"x^2 + 5x - 15 = 0": -1.0})(delta_hom=0.0).process_line(AUDIO, 16_000, HIST)
    assert r["status"] == "math_error" and r["flagged"]
    assert r["suggestion"]["latex"] == "x^2 + 5x - 50 = 0"
    assert r["suggestion"]["words"] == ("fifteen", "fifty")
    assert r["spoken_explanation"].startswith("You said fifteen; fifty would make this step valid. Did you mean fifty?")
    assert any(c["hom_class"] for c in r["candidates"])


def test_ask_back_two_valid_readings():
    make = stub(["60x^2 = 16"], {"60x^2 = 16": -1.0, "16x^2 = 16": -1.1, "60x^2 = 60": -1.1})
    r = make(delta_hom=100.0).process_line(AUDIO, 16_000, [{"output": "16x^2 - 16 = 0"}])
    assert r["status"] == "ask_back" and r["output"] == "60x^2 = 16"
    assert sorted(r["options"]) == ["16x^2 = 16", "60x^2 = 60"]
    assert "Did you mean" in r["spoken_explanation"] and "sixteen instead of sixty" in r["spoken_explanation"]


def test_divide_by_x_flag_explained():
    r = stub(["x^2 + 5x = 50"], {"x^2 + 5x = 50": -1.0})().process_line(
        AUDIO, 16_000, [{"output": "x^3 + 5x^2 = 50x"}])
    assert r["status"] == "math_error" and r["explanation"] == "dividing by x may lose the solution x = 0"
    assert r["spoken_explanation"] == "Dividing by x may lose the solution x equals 0."
    assert r["suggestion"] is None


def test_off_main_thread_needs_worker_and_worker_matches_inline():
    make = stub(PLANTED, {"x^2 + 5x - 15 = 0": -1.0})
    inline = make().process_line(AUDIO, 16_000, HIST)
    out = {}

    def run(pipe, key):
        try:
            out[key] = pipe.process_line(AUDIO, 16_000, HIST)
        except RuntimeError as e:
            out[key] = e
    pipe = make(cpu_worker=True)
    try:
        for p, key in ((make(), "inline"), (pipe, "worker")):
            t = threading.Thread(target=run, args=(p, key))
            t.start()
            t.join()
    finally:
        pipe.close()
    assert isinstance(out["inline"], RuntimeError)
    assert {k: out["worker"][k] for k in ("status", "output", "candidates")} == \
        {k: inline[k] for k in ("status", "output", "candidates")}


def test_short_audio_rejected():
    with pytest.raises(ValueError):
        stub(PLANTED, {})().process_line(AUDIO[:100], 16_000, HIST)


def test_unflagged_lines_have_no_spoken_explanation():
    assert spoken_explanation({"status": "verified", "heard": "x", "output": "x"}) is None


def test_tts_client_falls_back_without_server(tmp_path):
    sock = str(tmp_path / "none.sock")
    assert not tts_live.available(sock)
    assert tts_live.synth("hello", tmp_path / "a.wav", path=sock) is None
