"""End-to-end smoke test: Whisper -> LLM -> LaTeX on 50 Speech2Latex clips.

Usage: .venv/bin/python scripts/smoke_e2e.py [n_clips] [llm_model]
Writes data/smoke/e2e.jsonl (one row per clip).
"""

import json
import random
import re
import sys
from pathlib import Path

import jiwer
import librosa
import torch
from dotenv import dotenv_values
from openai import OpenAI
from sympy.parsing.latex import parse_latex
from transformers import pipeline

ROOT = Path(__file__).resolve().parents[1]
SPLIT = ROOT / "data/speech2latex/en_test_500"
N = int(sys.argv[1]) if len(sys.argv) > 1 else 50
LLM = sys.argv[2] if len(sys.argv) > 2 else "gpt-4.1-mini"

SYSTEM = ("You convert spoken math, as transcribed by a speech recognizer, into LaTeX. "
          "The transcript may contain recognition errors; use mathematical context to fix them. "
          "Reply with only the LaTeX expression: no $ delimiters, no explanation, no code fences.")


def norm(s):
    return re.sub(r"\s+", "", s.replace("$", ""))


def sympy_ok(s):
    try:
        parse_latex(s)
        return True
    except Exception:
        return False


rows = [json.loads(l) for l in open(SPLIT / "metadata.jsonl")]
rows = random.Random(0).sample(rows, N)

asr = pipeline("automatic-speech-recognition", model="openai/whisper-large-v3-turbo",
               device=0, dtype=torch.float16)
llm = OpenAI(api_key=dotenv_values(ROOT / ".env")["OPENAI_API_KEY"])

out = []
for i, r in enumerate(rows):
    audio, sr = librosa.load(SPLIT / r["file_name"], sr=16000, mono=True)
    text = asr({"array": audio, "sampling_rate": sr},
               generate_kwargs={"language": "english", "task": "transcribe"})["text"].strip()
    resp = llm.chat.completions.create(
        model=LLM, temperature=0,
        messages=[{"role": "system", "content": SYSTEM}, {"role": "user", "content": text}])
    pred = resp.choices[0].message.content.strip().strip("$").strip()
    out.append({"file_name": r["file_name"], "is_tts": r["is_tts"], "reference": r["sentence"],
                "whisper": text, "spoken_ref": r["pronunciation"], "pred": pred,
                "exact": norm(pred) == norm(r["sentence"]),
                "sympy_ok": sympy_ok(pred)})
    print(f"[{i+1:2d}/{N}] {'OK ' if out[-1]['exact'] else 'XX '} {text!r} -> {pred!r}  (ref {r['sentence']!r})",
          flush=True)

dest = ROOT / "data/smoke"
dest.mkdir(exist_ok=True)
with open(dest / "e2e.jsonl", "w") as f:
    for o in out:
        f.write(json.dumps(o, ensure_ascii=False) + "\n")

wer = jiwer.wer([o["spoken_ref"] for o in out], [o["whisper"] for o in out])
print(f"\nclips={N} llm={LLM}")
print(f"Whisper WER vs. intended speech: {wer:.3f}")
for name, sub in [("all", out), ("human", [o for o in out if not o["is_tts"]]),
                  ("tts", [o for o in out if o["is_tts"]])]:
    if sub:
        print(f"{name:6s} n={len(sub):2d} exact-match={sum(o['exact'] for o in sub)/len(sub):.2f} "
              f"sympy-parses={sum(o['sympy_ok'] for o in sub)/len(sub):.2f}")
