import torch
from transformers import (
    WhisperProcessor,
    WhisperForConditionalGeneration,
    Wav2Vec2Processor,
    Wav2Vec2ForCTC,
)
from sympy.parsing.latex import parse_latex

assert torch.cuda.is_available(), "CUDA unavailable"
device = "cuda"

print("GPU:", torch.cuda.get_device_name(0))

# 1. Load Whisper
whisper_id = "openai/whisper-large-v3-turbo"

wp = WhisperProcessor.from_pretrained(whisper_id)
wm = WhisperForConditionalGeneration.from_pretrained(
    whisper_id,
    torch_dtype=torch.float16,
).to(device)

print("Whisper loaded successfully")

# 2. Load Wav2Vec2
ctc_id = "facebook/wav2vec2-lv-60-espeak-cv-ft"

pp = Wav2Vec2Processor.from_pretrained(ctc_id)
pm = Wav2Vec2ForCTC.from_pretrained(ctc_id).to(device)

print("Wav2Vec2 loaded successfully")

# 3. Test LaTeX parsing
expr = parse_latex(r"\frac{x^2+1}{2}")
print("Parsed expression:", expr)

print("ALL MODEL-LOADING CHECKS PASSED")