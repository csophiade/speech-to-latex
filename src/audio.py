"""Audio loading: every model call gets 16 kHz mono float32."""

import librosa
import numpy as np

from src.config import SAMPLE_RATE


def load_audio(path, sr=SAMPLE_RATE):
    """Load any audio file (wav/mp3/...) as mono float32 at `sr` Hz."""
    audio, _ = librosa.load(str(path), sr=sr, mono=True)
    return audio.astype(np.float32), sr


def to_16k_mono(audio, orig_sr):
    """Convert an in-memory array (e.g. from the demo microphone) to 16 kHz mono float32."""
    audio = np.asarray(audio, dtype=np.float32)
    if audio.ndim == 2:
        # Gradio gives (samples, channels)
        audio = audio.mean(axis=1) if audio.shape[1] <= 8 else audio.mean(axis=0)
    if np.abs(audio).max() > 1.0:
        audio = audio / 32768.0
    if orig_sr != SAMPLE_RATE:
        audio = librosa.resample(audio, orig_sr=orig_sr, target_sr=SAMPLE_RATE)
    return audio.astype(np.float32), SAMPLE_RATE


def split_on_silence(audio, sr=SAMPLE_RATE, min_gap_s=1.2, rel_thresh=0.3, min_speech_s=0.25, pad_s=0.15):
    """Split a recording into utterances at silences >= min_gap_s. Returns [(start, end)] in samples.

    Energy-based: frame RMS in dB; a frame is speech if it is above floor + rel_thresh * (peak - floor),
    with floor/peak the 10th/99th percentiles (so the threshold adapts to mic level and room noise).
    Speech islands shorter than min_speech_s (breaths, clicks) are treated as silence.
    """
    hop, win = sr // 100, sr // 40   # 10 ms hop, 25 ms window
    rms = librosa.feature.rms(y=np.asarray(audio, dtype=np.float32), frame_length=win, hop_length=hop)[0]
    db = 20 * np.log10(rms + 1e-8)
    floor, peak = np.percentile(db, 10), np.percentile(db, 99)
    speech = db > floor + rel_thresh * (peak - floor)

    def runs(mask):
        """[(start_frame, end_frame)] of consecutive True frames."""
        edges = np.flatnonzero(np.diff(np.concatenate([[0], mask.astype(np.int8), [0]])))
        return list(zip(edges[::2], edges[1::2]))

    for s, e in runs(speech):
        if (e - s) * hop < min_speech_s * sr:
            speech[s:e] = False
    for s, e in runs(~speech):   # fill short pauses inside a line
        if s > 0 and e < len(speech) and (e - s) * hop < min_gap_s * sr:
            speech[s:e] = True
    pad = int(pad_s * sr)
    return [(max(0, s * hop - pad), min(len(audio), e * hop + win + pad)) for s, e in runs(speech)]
