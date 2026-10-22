"""Phase 3.5 QC: Whisper best-1 on random synthesised clips vs the verbalised text.

Usage: .venv/bin/python -W ignore scripts/tts_qc.py [--n 20]
Writes results/reports/tts_qc.md. A clip is "garbage" if its normalised WER exceeds 0.5
(digits/symbols are spelled out on both sides first; see src/spoken_norm.py).
"""

import argparse
import json
import random
import sys
from pathlib import Path

import jiwer

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.asr import transcribe_nbest  # noqa: E402
from src.audio import load_audio  # noqa: E402
from src.config import DATA_DIR, RESULTS_DIR, ROOT, SEED  # noqa: E402
from src.spoken_norm import normalise_spoken  # noqa: E402

MANIFEST = DATA_DIR / "derivations" / "audio" / "manifest.jsonl"
REPORT = RESULTS_DIR / "reports" / "tts_qc.md"
GARBAGE_WER = 0.5
NOTES = [
    "## Notes", "",
    "- First pass (plain spoken text sent to TTS): 0/20 garbage in the sample, but over all 200 clips "
    "3 were garbage and a leading \"x\" was swallowed in 20 (v1 4, v2 16), e.g. \"x times the quantity ...\" "
    "-> \"times the quantity ...\". The verbalisation was faithful; the TTS front end was the problem.",
    "- Fix: scripts/synthesize_derivations.tts_text spells a bare x as \"ex\" for CosyVoice only (manifest "
    "keeps the plain text). On the 12 worst lines this recovered the first word 12/12 (vs 5/12 re-rendered "
    "unchanged). All clips were re-synthesised with the fix; the numbers above are after it.",
    "- Remaining dropped last words are random, not systematic: re-rendering the 10 affected lines "
    "unchanged fixed 10/10 (adding a final period fixed 9/10). For the full run, transcribe every clip "
    "(--scan-all), delete clips with WER > 0 and rerun the resumable synthesis once.",
]


def score(r, errors):
    audio, _ = load_audio(ROOT / r["wav"])
    hyp = transcribe_nbest(audio)[0][0]
    ref, words = normalise_spoken(r["spoken"]), normalise_spoken(hyp)
    wer = jiwer.wer(ref, words or "<empty>")
    err = errors[r["id"]]
    return {**r, "asr": hyp, "wer": wer, "garbage": wer > GARBAGE_WER,
            "first_lost": words.split()[:1] != ref.split()[:1], "last_lost": words.split()[-1:] != ref.split()[-1:],
            "err_line": bool(err and err["line"] == r["line"])}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=20)
    ap.add_argument("--scan-all", action="store_true", help="also transcribe every clip, add per-voice stats")
    args = ap.parse_args()
    rows = [json.loads(line) for line in MANIFEST.read_text().splitlines() if line.strip()]
    errors = {it["id"]: it["error"] for it in json.loads((DATA_DIR / "derivations" / "derivations.json").read_text())}
    sample = random.Random(SEED).sample(rows, min(args.n, len(rows)))
    out = [score(r, errors) for r in sample]
    n_bad = sum(o["garbage"] for o in out)
    lines = [
        "# TTS QC (Phase 3.5)", "",
        f"{len(out)} random clips (seed {SEED}) from {len(rows)} synthesised lines "
        f"({len({r['id'] for r in rows})} derivations, voices {sorted({r['voice'] for r in rows})}).",
        f"Whisper best-1 vs the LLM verbalisation, both normalised (digits and symbols spelled out, "
        f"filler words dropped). Garbage = WER > {GARBAGE_WER}.", "",
        f"**Garbage: {n_bad} / {len(out)}** (threshold for fixing the verbalisation prompt: > 4). "
        f"Mean WER {sum(o['wer'] for o in out) / len(out):.3f}.", "",
        "| # | clip | voice | LaTeX | spoken (LLM) | Whisper best-1 | WER | garbage |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for i, o in enumerate(out, 1):
        tag = " (deliberate error)" if o["err_line"] else ""
        lines.append(f"| {i} | {o['id']} line {o['line']}{tag} | {o['voice']} | `{o['latex']}` | {o['spoken']} | "
                     f"{o['asr']} | {o['wer']:.2f} | {'YES' if o['garbage'] else ''} |")
    if args.scan_all:
        scored = [score(r, errors) for r in rows]
        lines += ["", f"## All {len(scored)} clips", "",
                  "| voice | clips | mean WER | garbage | WER > 0 | first word lost | last word lost |",
                  "|---|---|---|---|---|---|---|"]
        for v in sorted({o["voice"] for o in scored}):
            o = [x for x in scored if x["voice"] == v]
            lines.append(f"| {v} | {len(o)} | {sum(x['wer'] for x in o) / len(o):.3f} | "
                         f"{sum(x['garbage'] for x in o)} | {sum(x['wer'] > 0 for x in o)} | "
                         f"{sum(x['first_lost'] for x in o)} | {sum(x['last_lost'] for x in o)} |")
        worst = sorted((x for x in scored if x["wer"] > 0.15), key=lambda x: -x["wer"])
        lines += ["", "Clips with WER > 0.15:", ""] + [
            f"- {x['id']} line {x['line']} {x['voice']} (WER {x['wer']:.2f}): \"{x['spoken']}\" -> \"{x['asr']}\""
            for x in worst]
    lines += [""] + NOTES
    REPORT.parent.mkdir(parents=True, exist_ok=True)
    REPORT.write_text("\n".join(lines) + "\n")
    print(f"garbage {n_bad}/{len(out)} -> {REPORT}")
    for o in out:
        print(f"{o['wer']:.2f} {o['voice']} {o['id']}:{o['line']} | {o['spoken']} | {o['asr']}")


if __name__ == "__main__":
    main()
