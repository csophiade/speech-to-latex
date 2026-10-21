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
