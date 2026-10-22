"""Re-render synthesised clips that Whisper does not hear as their text; keep the best take.

Usage: .venv/bin/python scripts/tts_redo.py [--rounds 3] [--max-wer 0.0] [--voices v3]

Scores every clip in data/derivations/audio/manifest.jsonl (Whisper best-1 vs the spoken text,
same normalisation as scripts/tts_qc.py; cached in qc.jsonl), then for up to --rounds rounds
re-synthesises every clip with WER > --max-wer and keeps whichever take scores lower. CosyVoice
samples, so a fresh take usually fixes dropped words. Finally writes each clip's qc_wer and
qc_asr into the manifest, so later phases can exclude clips whose audio does not match the label.
"""

import argparse
import json
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from synthesize_derivations import MANIFEST, tts_text  # noqa: E402
from tts_qc import score  # noqa: E402

from src.config import DATA_DIR  # noqa: E402
from src.tts import synth_batch  # noqa: E402

QC = MANIFEST.with_name("qc.jsonl")


def key(r):
    return r["id"], r["voice"], r["line"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rounds", type=int, default=3)
    ap.add_argument("--max-wer", type=float, default=0.0)
    ap.add_argument("--voices", help="comma-separated voices to score and re-render (default: all)")
    args = ap.parse_args()
    want = set(args.voices.split(",")) if args.voices else None

    rows = {key(r): r for r in map(json.loads, MANIFEST.read_text().splitlines())}
    errors = {it["id"]: it["error"] for it in json.loads((DATA_DIR / "derivations" / "derivations.json").read_text())}
    qc = {key(q): q for q in map(json.loads, QC.read_text().splitlines())} if QC.exists() else {}
    for k, r in rows.items():
        if k not in qc and (want is None or r["voice"] in want):
            s = score(r, errors)
            qc[k] = {"id": r["id"], "voice": r["voice"], "line": r["line"], "wer": s["wer"], "asr": s["asr"], "takes": 1}
    QC.write_text("".join(json.dumps(q) + "\n" for q in qc.values()))
    mine = {k: q for k, q in qc.items() if want is None or k[1] in want}
    print("bad before:", sum(q["wer"] > args.max_wer for q in mine.values()), "of", len(mine))

    for rnd in range(1, args.rounds + 1):
        bad = [rows[k] for k, q in mine.items() if q["wer"] > args.max_wer]
        if not bad:
            break
        for r in bad:   # keep the current take aside until the new one is scored
            wav = DATA_DIR.parent / r["wav"]
            shutil.move(wav, wav.with_suffix(".prev.wav"))
        synth_batch([{"text": tts_text(r["spoken"]), "voice": r["voice"], "out_path": DATA_DIR.parent / r["wav"]}
                     for r in bad])
        fixed = 0
        for r in bad:
            k, wav = key(r), DATA_DIR.parent / r["wav"]
            prev = wav.with_suffix(".prev.wav")
            new = score(r, errors) if wav.exists() else None
            if new is not None and new["wer"] < qc[k]["wer"]:
                qc[k] |= {"wer": new["wer"], "asr": new["asr"]}
                prev.unlink()
            else:
                shutil.move(prev, wav)
            qc[k]["takes"] += 1
            fixed += qc[k]["wer"] <= args.max_wer
        QC.write_text("".join(json.dumps(q) + "\n" for q in qc.values()))
        print(f"round {rnd}: re-rendered {len(bad)}, now clean {fixed}")

    for k, q in qc.items():
        rows[k] |= {"qc_wer": round(q["wer"], 4), "qc_asr": q["asr"]}
    MANIFEST.write_text("".join(json.dumps(rows[k]) + "\n" for k in sorted(rows)))
    left = [q for q in mine.values() if q["wer"] > args.max_wer]
    print(f"still WER > {args.max_wer}: {len(left)}; WER > 0.15: {sum(q['wer'] > 0.15 for q in left)}")
    for q in sorted(left, key=lambda q: -q["wer"])[:30]:
        print(f"  {q['wer']:.2f} {q['voice']} {q['id']}:{q['line']} -> {q['asr']}")


if __name__ == "__main__":
    main()
