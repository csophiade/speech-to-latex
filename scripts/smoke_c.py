import torch, jiwer
from transformers import (WhisperForConditionalGeneration, WhisperProcessor,
                          Wav2Vec2ForCTC, Wav2Vec2Processor)
from sympy.parsing.latex import parse_latex
import numpy as np
import librosa

dev = "cuda" if torch.cuda.is_available() else "cpu"
print("device:", dev)

# SymPy LaTeX parsing
try:
    print(parse_latex(r"\frac{x^2+1}{2}"))
except Exception as e:
    print("antlr failed, trying lark:", e)
    print(parse_latex(r"\frac{x^2+1}{2}", backend="lark"))

# Whisper n-best
wp = WhisperProcessor.from_pretrained("openai/whisper-large-v3-turbo")
wm = WhisperForConditionalGeneration.from_pretrained(
    "openai/whisper-large-v3-turbo", dtype=torch.float16).to(dev)

audio_path = "data/initial_test_math.wav"  # replace with a real clip
audio, sr = librosa.load(audio_path, sr=16000, mono=True)
feats = wp(audio, sampling_rate=16000, return_tensors="pt").input_features.to(dev, torch.float16)
out = wm.generate(feats, num_beams=5, num_return_sequences=5, language="en")
print(wp.batch_decode(out, skip_special_tokens=True))

# Phoneme CTC score (needs espeak-ng)
pp = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-lv-60-espeak-cv-ft")
pm = Wav2Vec2ForCTC.from_pretrained("facebook/wav2vec2-lv-60-espeak-cv-ft").to(dev)
inp = pp(audio, sampling_rate=16000, return_tensors="pt").input_values.to(dev)
labels = pp.tokenizer("x squared plus one", return_tensors="pt").input_ids.to(dev)
loss = pm(inp, labels=labels).loss
print("ctc loss:", loss.item())