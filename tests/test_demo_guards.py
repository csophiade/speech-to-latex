"""Demo guards from the Phase 5 dry-run review: speech gate, mic scaling, pending ask-back, stale clicks."""

import sys
from pathlib import Path

import numpy as np
import pytest

from src.audio import load_audio, speech_seconds, to_16k_mono

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "demo"))
app = pytest.importorskip("app")
CLIP = ROOT / "demo" / "examples" / "quadratic" / "line_2.wav"


def test_speech_gate_rejects_silence_and_noise_keeps_speech():
    rng = np.random.default_rng(0)
    assert speech_seconds(np.zeros(32_000, np.float32)) == 0.0
    assert speech_seconds(rng.uniform(-0.1, 0.1, 32_000).astype(np.float32)) < app.MIN_SPEECH_S
    if CLIP.exists():
        assert speech_seconds(load_audio(CLIP)[0]) > 1.0


def test_int16_mic_scaled_by_type_range_not_peak():
    dither = np.array([1, -1, 0, 1] * 4000, dtype=np.int16)
    audio, _ = to_16k_mono(dither, 16_000)
    assert np.abs(audio).max() < 1e-3
    with pytest.raises(ValueError):
        to_16k_mono(np.zeros(0, np.int16), 16_000)


def line(status, **kw):
    return dict({"status": status, "output": "x", "heard": "x", "options": ["a", "b"], "latency_s": 0.1,
                 "latency_parts": {}}, **kw)


def test_unresolved_ask_back_blocks_new_lines():
    st = dict(app.fresh(), history=[line("given", typed=True), line("ask_back")])
    assert app.pending_choice(st)
    with pytest.raises(app.gr.Error):
        app._process(st, np.zeros(16_000, np.float32), 16_000)
    st["history"][-1]["resolved_from"] = "you chose B"
    assert not app.pending_choice(st)


def test_stale_resolution_click_is_a_no_op():
    assert app.resolve(app.fresh(), "A")[0]["history"] == []
    st = dict(app.fresh(), history=[line("verified")])
    assert app.resolve(st, "A")[0]["history"][-1]["status"] == "verified"


def test_display_helpers():
    assert app.plain(r"50 ( x^2 + 5x ) = 50 \times 50") == "50 ( x² + 5x ) = 50 × 50"
    assert app.pretty("x = -sqrt(85)/2") == "x = -√85/2"
    assert "$" not in app.m("")
