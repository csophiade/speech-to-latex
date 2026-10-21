import json

import numpy as np
import pytest

from src import config
from src.audio import load_audio, to_16k_mono


def first_clip(name):
    folder = config.DATASETS[name]
    row = json.loads(open(folder / "metadata.jsonl").readline())
    return folder / row["file_name"], row


def test_config_values():
    assert config.SEED == 42
    assert config.ALPHA == 0.05 and config.BETA == 0.5
    assert config.K_BEAMS == 5 and config.M_CANDIDATES == 3
    assert config.N_CHECK_POINTS == 20 and config.NUM_TOL == 1e-6
    assert config.DEVICE in {"cuda", "cpu"}
    for path in config.DATASETS.values():
        assert (path / "metadata.jsonl").exists(), path


# One real clip per audio dataset: Speech2Latex human, Speech2Latex TTS, MathSpeech (mp3)
@pytest.mark.parametrize("dataset,want_tts", [("s2l_test500", 0), ("s2l_test500", 1), ("mathspeech", None)])
def test_loader_on_real_clip(dataset, want_tts):
    folder = config.DATASETS[dataset]
    rows = [json.loads(l) for l in open(folder / "metadata.jsonl")]
    if want_tts is not None:
        rows = [r for r in rows if r["is_tts"] == want_tts]
    row = rows[0]
    audio, sr = load_audio(folder / row["file_name"])
    assert sr == 16_000
    assert audio.dtype == np.float32 and audio.ndim == 1
    assert len(audio) == pytest.approx(row["duration_s"] * sr, rel=0.02)
    assert np.sqrt(np.mean(audio ** 2)) > 1e-3, "clip is silent"


def test_to_16k_mono_from_int16_stereo():
    t = np.linspace(0, 1, 48_000, endpoint=False)
    tone = (np.sin(2 * np.pi * 440 * t) * 20_000).astype(np.int16)
    audio, sr = to_16k_mono(np.stack([tone, tone], axis=1), 48_000)
    assert sr == 16_000 and audio.dtype == np.float32 and audio.shape == (16_000,)
    assert 0.3 < np.abs(audio).max() <= 1.0


@pytest.mark.parametrize("name", ["speech2latex", "mathspeech", "mathbridge", "processbench"])
def test_data_readmes_exist(name):
    readme = config.DATA_DIR / name / "README.md"
    assert readme.exists() and readme.read_text().strip()
