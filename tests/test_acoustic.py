import pytest
import torch

from src import config
from src.audio import load_audio
from src.candidates import to_spoken

DEV = config.DATASETS["s2l_dev"]

# en_dev clips where Whisper confirms the speaker said the number we corrupt.
# (Some `pronunciation` fields contain words the speaker never said, e.g. stray equation
# numbers, so probes must be verified this way.)
PROBES = [
    ("audio/2129_spk_7.wav", "f of c equals 6", "f of c equals 8"),
    ("audio/4794_spk_7.wav", "One half divided by one sixth equals three",
     "One half divided by one sixth equals two"),
    ("audio/2680_spk_39.wav", "Fraction: 18 over 19", "Fraction: 80 over 19"),
]

gpu = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs the GPU models")


def test_to_spoken_says_squared():
    assert any("squared" in s for s in to_spoken("x^2 + 1"))


@gpu
@pytest.mark.parametrize("file_name,true_text,corrupted", PROBES)
def test_true_text_outscores_corruption_on_both_scores(file_name, true_text, corrupted):
    from src.acoustic import phoneme_scores, whisper_scores
    audio, _ = load_audio(DEV / file_name)
    w_true, w_bad = whisper_scores(audio, [true_text, corrupted])
    p_true, p_bad = phoneme_scores(audio, [true_text, corrupted])
    assert w_true > w_bad
    assert p_true > p_bad


@gpu
def test_acoustic_prefers_spoken_candidate_and_margins():
    from src.acoustic import acoustic_from_spoken, margins
    file_name, true_text, corrupted = PROBES[0]
    audio, _ = load_audio(DEV / file_name)
    scores = acoustic_from_spoken(audio, [[true_text], [corrupted, "f of c equals eight"]])
    assert scores[0] > scores[1]
    assert margins(scores) == [0.0, scores[0] - scores[1]]


@gpu
def test_batched_scores_match_single():
    from src.acoustic import phoneme_scores, whisper_scores
    file_name, true_text, corrupted = PROBES[1]
    audio, _ = load_audio(DEV / file_name)
    texts = [true_text, corrupted, "x"]
    for fn in (whisper_scores, phoneme_scores):
        batch = fn(audio, texts)
        single = [fn(audio, [t])[0] for t in texts]
        assert batch == pytest.approx(single, abs=2e-2)
