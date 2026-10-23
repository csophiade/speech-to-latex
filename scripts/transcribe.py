import librosa
import torch
from transformers import pipeline

audio_path = "data/initial_test_math.wav"

# Load Whisper on GPU
pipe = pipeline(
    "automatic-speech-recognition",
    model="openai/whisper-large-v3-turbo",
    device=0,
    dtype=torch.float16,
)

# Load WAV as 16 kHz mono
audio, sr = librosa.load(audio_path, sr=16000, mono=True)

# Transcribe
result = pipe(
    {"array": audio, "sampling_rate": sr},
    generate_kwargs={
        "language": "english",
        "task": "transcribe",
    },
)

print("\nTRANSCRIPTION:")
print(result["text"])