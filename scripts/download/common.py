"""Shared helpers for the dataset download scripts."""

import io
import json
import os
from pathlib import Path

import soundfile as sf
from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "data"
SEED = 42

# Picks up HF_TOKEN from .env (all datasets are public; the token just avoids rate limits)
load_dotenv(PROJECT_ROOT / ".env")


def write_audiofolder(rows, out_dir, audio_key, name_fn):
    """Write rows as an HF "audiofolder": audio/<name> files plus metadata.jsonl.

    Audio bytes are written as-is (no re-encoding). Load back with
    load_dataset("audiofolder", data_dir=out_dir).
    """
    out_dir = Path(out_dir)
    (out_dir / "audio").mkdir(parents=True, exist_ok=True)
    with open(out_dir / "metadata.jsonl", "w") as meta:
        for i, row in enumerate(rows):
            audio = row.pop(audio_key)
            name = name_fn(i, row, audio)
            (out_dir / "audio" / name).write_bytes(audio["bytes"])
            info = sf.info(io.BytesIO(audio["bytes"]))
            row = {"file_name": f"audio/{name}", **row,
                   "duration_s": round(info.duration, 3), "sample_rate": info.samplerate}
            meta.write(json.dumps(row, ensure_ascii=False) + "\n")
    return out_dir


def write_jsonl(rows, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    return path


def dir_size_mb(path):
    return sum(p.stat().st_size for p in Path(path).rglob("*") if p.is_file()) / 1e6


def hf_token():
    return os.environ.get("HF_TOKEN")
