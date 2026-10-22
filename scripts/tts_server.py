"""Persistent CosyVoice2 server for the demo's spoken explanations. Runs under /workspace/tts-venv.

Usage: /workspace/tts-venv/bin/python scripts/tts_server.py [--socket PATH]
Loads the model and the voice prompts once, then serves a Unix socket: one JSON request per line,
{"text", "voice", "out_path"} -> one JSON reply line {"out_path", "audio_s", "compute_s"} or
{"error"}. {"ping": true} -> {"ok": true}. Requests are served one at a time (one GPU model).
Same synthesis as scripts/tts_worker.py (16 kHz mono, peak-normalised); client: src/tts_live.py.
"""

import argparse
import json
import os
import socket
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from tts_worker import COSYVOICE_DIR, MODEL_DIR, OUT_SR, REF_DIRS  # noqa: E402

DEFAULT_SOCKET = "/tmp/speech-to-latex-tts.sock"
sys.path[:0] = [COSYVOICE_DIR, os.path.join(COSYVOICE_DIR, "third_party/Matcha-TTS")]


def load(voices):
    from cosyvoice.cli.cosyvoice import CosyVoice2
    tts = CosyVoice2(MODEL_DIR, fp16=True)
    for v in voices:
        ref_text = open(os.path.join(REF_DIRS[v], f"{v}.txt")).read().strip()
        tts.add_zero_shot_spk(ref_text, os.path.join(REF_DIRS[v], f"{v}.wav"), v)
    return tts


def synth(tts, req):
    """One request -> reply dict (never raises)."""
    import librosa
    import soundfile as sf
    import torch
    try:
        t0 = time.time()
        text = req["text"].strip()
        text = text[:1].upper() + text[1:]
        chunks = [o["tts_speech"] for o in tts.inference_zero_shot(
            text, "", "", zero_shot_spk_id=req.get("voice", "v1"), stream=False)]
        wav = torch.cat(chunks, dim=1).squeeze(0).float().cpu().numpy()
        wav = librosa.resample(wav, orig_sr=tts.sample_rate, target_sr=OUT_SR)
        wav = 0.9 * wav / max(float(abs(wav).max()), 1e-6)
        os.makedirs(os.path.dirname(os.path.abspath(req["out_path"])), exist_ok=True)
        sf.write(req["out_path"], wav, OUT_SR, subtype="PCM_16")
        return {"out_path": req["out_path"], "audio_s": len(wav) / OUT_SR, "compute_s": time.time() - t0}
    except Exception as e:
        return {"out_path": req.get("out_path"), "error": f"{type(e).__name__}: {e}"}


def serve(tts, path):
    if os.path.exists(path):
        os.unlink(path)
    srv = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    srv.bind(path)
    srv.listen(8)
    print(f"tts_server ready on {path}", flush=True)
    while True:
        conn, _ = srv.accept()
        with conn, conn.makefile("rw") as f:
            for line in f:
                try:
                    req = json.loads(line)
                except json.JSONDecodeError as e:
                    rep = {"error": f"bad request: {e}"}
                else:
                    rep = {"ok": True} if req.get("ping") else synth(tts, req)
                f.write(json.dumps(rep) + "\n")
                f.flush()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--socket", default=DEFAULT_SOCKET)
    ap.add_argument("--voices", nargs="+", default=["v1"])
    a = ap.parse_args()
    serve(load(a.voices), a.socket)


if __name__ == "__main__":
    main()
