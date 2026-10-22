"""CosyVoice2 batch worker. Runs under /workspace/tts-venv, NOT the project venv.

Usage: tts_worker.py items.jsonl
Each line: {"text": ..., "voice": "v1"|"v2", "out_path": ...}. Writes 16 kHz mono wav per item,
prints one JSON line per item ({"out_path", "audio_s", "compute_s"} or {"out_path", "error"}).
"""

import json
import os
import sys
import time

COSYVOICE_DIR = os.environ.get("COSYVOICE_DIR", "/workspace/models/CosyVoice")
MODEL_DIR = os.environ.get("COSYVOICE_MODEL", "/workspace/models/CosyVoice2-0.5B")
REF_DIR = os.environ.get("TTS_REF_DIR", "/workspace/models/tts_refs")
VOICES = ("v1", "v2")  # v1 = LibriSpeech 1272 (male), v2 = LibriSpeech 1993 (female)
OUT_SR = 16000

sys.path[:0] = [COSYVOICE_DIR, os.path.join(COSYVOICE_DIR, "third_party/Matcha-TTS")]


def main(jsonl_path):
    import librosa
    import soundfile as sf
    import torch
    from cosyvoice.cli.cosyvoice import CosyVoice2

    tts = CosyVoice2(MODEL_DIR, fp16=True)
    for v in VOICES:  # embed each reference clip once
        ref_text = open(os.path.join(REF_DIR, f"{v}.txt")).read().strip()
        tts.add_zero_shot_spk(ref_text, os.path.join(REF_DIR, f"{v}.wav"), v)

    with open(jsonl_path) as f:
        items = [json.loads(line) for line in f if line.strip()]
    for it in items:
        try:
            t0 = time.time()
            # a lowercase leading letter ("x squared ...") gets swallowed; sentence-case fixes it
            text = it["text"].strip()
            text = text[:1].upper() + text[1:]
            chunks = [o["tts_speech"] for o in tts.inference_zero_shot(
                text, "", "", zero_shot_spk_id=it["voice"], stream=False)]
            wav = torch.cat(chunks, dim=1).squeeze(0).float().cpu().numpy()
            wav = librosa.resample(wav, orig_sr=tts.sample_rate, target_sr=OUT_SR)
            wav = 0.9 * wav / max(float(abs(wav).max()), 1e-6)  # peak-normalise (refs differ in level)
            os.makedirs(os.path.dirname(os.path.abspath(it["out_path"])), exist_ok=True)
            sf.write(it["out_path"], wav, OUT_SR, subtype="PCM_16")
            res = {"out_path": it["out_path"], "audio_s": len(wav) / OUT_SR,
                   "compute_s": time.time() - t0}
        except Exception as e:  # one bad line must not kill the batch
            res = {"out_path": it.get("out_path"), "error": f"{type(e).__name__}: {e}"}
        print("RESULT " + json.dumps(res), flush=True)
    if torch.cuda.is_available():
        print(f"peak GPU mem {torch.cuda.max_memory_allocated() / 2**30:.2f} GiB", file=sys.stderr)


if __name__ == "__main__":
    main(sys.argv[1])
