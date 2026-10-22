"""Text-to-speech via CosyVoice2-0.5B, run in a separate venv (/workspace/tts-venv).

The TTS stack pins torch/transformers versions that clash with ours, so we shell out to
scripts/tts_worker.py. Voices: "v1" (male) and "v2" (female), zero-shot cloned from
LibriSpeech dev-clean reference clips (CC BY 4.0) in /workspace/models/tts_refs.
"""

import json
import subprocess
import tempfile
from pathlib import Path

TTS_PYTHON = "/workspace/tts-venv/bin/python"
WORKER = Path(__file__).resolve().parent.parent / "scripts" / "tts_worker.py"
VOICES = ("v1", "v2")


def synth_batch(items):
    """items: [{"text", "voice", "out_path"}]. Loads the model once; returns per-item result dicts
    ({"out_path", "audio_s", "compute_s"} or {"out_path", "error"}) in input order."""
    for it in items:
        if it["voice"] not in VOICES:
            raise ValueError(f"unknown voice {it['voice']!r}; use one of {VOICES}")
    with tempfile.NamedTemporaryFile("w", suffix=".jsonl", delete=False) as f:
        for it in items:
            f.write(json.dumps({"text": it["text"], "voice": it["voice"],
                                "out_path": str(it["out_path"])}) + "\n")
    proc = subprocess.run([TTS_PYTHON, "-W", "ignore", str(WORKER), f.name],
                          capture_output=True, text=True)
    Path(f.name).unlink(missing_ok=True)
    results = [json.loads(line[7:]) for line in proc.stdout.splitlines() if line.startswith("RESULT ")]
    if proc.returncode != 0 or len(results) != len(items):
        raise RuntimeError(f"tts worker failed (rc={proc.returncode}):\n{proc.stderr[-2000:]}")
    return results


def synth(text, voice, out_path):
    """Synthesise one line to a 16 kHz mono wav. Pays model load each call; prefer synth_batch."""
    res = synth_batch([{"text": text, "voice": voice, "out_path": out_path}])[0]
    if "error" in res:
        raise RuntimeError(res["error"])
    return res
