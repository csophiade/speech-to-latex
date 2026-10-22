"""Phase 3.5: verbalise every derivation line with the LLM, synthesise it with both TTS voices.

Usage: .venv/bin/python -W ignore scripts/synthesize_derivations.py [--limit N] [--ids a,b] [--voices v1,v2]
       .venv/bin/python -W ignore scripts/synthesize_derivations.py --accent-subset --voices v3
--accent-subset restricts to data/derivations/accent_subset.json (created on first use: 50 calib + 50 test
derivations, stratified by type and error presence, seed 42).
Writes data/derivations/audio/<voice>/<id>/line_<k>.wav and data/derivations/audio/manifest.jsonl
({id, voice, line, latex, spoken, wav, duration_s}). Resumable: existing wavs are skipped.
"""

import argparse
import json
import random
import re
import sys
import threading
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import soundfile as sf

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.candidates import llm_call  # noqa: E402
from src.config import DATA_DIR  # noqa: E402
from src.tts import VOICES, synth_batch  # noqa: E402

DERIVS = DATA_DIR / "derivations" / "derivations.json"
AUDIO_DIR = DATA_DIR / "derivations" / "audio"
MANIFEST = AUDIO_DIR / "manifest.jsonl"
ACCENT_SUBSET = DATA_DIR / "derivations" / "accent_subset.json"
_LOCK = threading.Lock()
CHUNK = 150   # lines per TTS worker call: ~65 s model load amortised, progress saved per chunk

# Each line is verbalised on its own (no derivation context), so the LLM cannot "fix" a deliberate
# error by looking at the neighbouring lines. Style matches data/script.md (the human recordings).
SYSTEM = """You turn one line of LaTeX maths into the words a student would say when dictating it aloud.
Rules:
- Read exactly what is written. The line may be mathematically wrong on purpose: never correct, simplify, reorder or complete it.
- Numbers as words ("twenty four", "negative three"). A leading minus on a number or term is "negative"; subtraction is "minus".
- Fractions: "a over b" ("thirteen over four"); for a compound numerator/denominator say "the quantity ... , over the quantity ...".
- Brackets: "times the quantity x plus four" or "the quantity x plus three, squared", so the grouping is audible.
- x^2 is "x squared", \\sqrt{D} is "the square root of D", \\pm is "plus or minus", \\cdot is "times".
- Relations: "equals", "is less than", "is greater than", "is less than or equal to", "is greater than or equal to".
- \\text{ or } is "or". No LaTeX, no symbols, no digits, no narration such as "next" or "so".
Return JSON: {"spoken": "<the words>"}"""

EXAMPLES = [
    ("6(x + 4) = 40", "six times the quantity x plus four equals forty"),
    ("-4x + 6 = 19", "negative four x plus six equals nineteen"),
    ("x = -\\frac{13}{4}", "x equals negative thirteen over four"),
    ("x \\cdot (2x - 3) = 5x + 1", "x times the quantity two x minus three equals five x plus one"),
    ("x \\geq \\frac{10}{7}", "x is greater than or equal to ten over seven"),
]


def verbalise(latex):
    """LLM reading of one LaTeX line (disk-cached via llm_call)."""
    messages = [{"role": "system", "content": SYSTEM}]
    for src, spoken in EXAMPLES:
        messages += [{"role": "user", "content": src},
                     {"role": "assistant", "content": json.dumps({"spoken": spoken})}]
    messages.append({"role": "user", "content": latex})
    text, _ = llm_call(messages)
    try:
        spoken = json.loads(text)["spoken"].strip()
    except (json.JSONDecodeError, KeyError, TypeError, AttributeError):
        spoken = ""
    if not spoken:
        raise RuntimeError(f"verbalisation failed for {latex!r}: {text!r}")
    return spoken


def tts_text(spoken):
    """Text sent to CosyVoice. A bare "x" is often swallowed (16/100 v2 lines lost a leading "x" in
    QC); spelling it "ex" fixed 12/12 of those lines. The manifest keeps the plain spoken text."""
    return re.sub(r"\bx\b", "ex", spoken)


def wav_path(voice, deriv_id, k):
    return AUDIO_DIR / voice / deriv_id / f"line_{k}.wav"


def load_manifest():
    if not MANIFEST.exists():
        return {}
    rows = [json.loads(line) for line in MANIFEST.read_text().splitlines() if line.strip()]
    return {(r["id"], r["voice"], r["line"]): r for r in rows}


def save_manifest(rows):
    MANIFEST.parent.mkdir(parents=True, exist_ok=True)
    tmp = MANIFEST.with_suffix(".tmp")
    with _LOCK:   # both voice threads save
        tmp.write_text("".join(json.dumps(r) + "\n" for r in sorted(rows.values(),
                       key=lambda r: (r["id"], r["voice"], r["line"]))))
        tmp.replace(MANIFEST)


def select(items, ids=None, limit=None):
    if ids:
        by_id = {it["id"]: it for it in items}
        missing = [i for i in ids if i not in by_id]
        if missing:
            raise SystemExit(f"unknown ids: {missing}")
        items = [by_id[i] for i in ids]
    return items[:limit] if limit else items


def accent_subset(items, per_split=50, seed=42):
    """Ids of a split-balanced subset, stratified by (type, has error) with proportional allocation."""
    rng = random.Random(seed)
    ids = []
    for split in ("calib", "test"):
        pool = [it for it in items if it["split"] == split]
        strata = {}
        for it in pool:
            strata.setdefault((it["type"], it["error"] is not None), []).append(it["id"])
        for key in sorted(strata):
            ids += rng.sample(sorted(strata[key]), round(per_split * len(strata[key]) / len(pool)))
    return ids


def load_accent_subset(items):
    """The saved accent subset; created (and saved) on first use so it never changes afterwards."""
    if not ACCENT_SUBSET.exists():
        ACCENT_SUBSET.write_text(json.dumps({"seed": 42, "ids": accent_subset(items)}, indent=1) + "\n")
    return json.loads(ACCENT_SUBSET.read_text())["ids"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int)
    ap.add_argument("--ids", help="comma-separated derivation ids")
    ap.add_argument("--accent-subset", action="store_true", help="only the derivations in accent_subset.json")
    ap.add_argument("--voices", default="v1,v2", help=f"any of {','.join(VOICES)}")
    args = ap.parse_args()
    voices = args.voices.split(",")
    items = json.loads(DERIVS.read_text())
    ids = load_accent_subset(items) if args.accent_subset else args.ids and args.ids.split(",")
    items = select(items, ids, args.limit)

    lines = [(it["id"], k, latex) for it in items for k, latex in enumerate(it["lines"])]
    with ThreadPoolExecutor(8) as pool:
        spoken = dict(zip([(i, k) for i, k, _ in lines], pool.map(verbalise, [l for *_, l in lines])))
    print(f"{len(items)} derivations, {len(lines)} lines verbalised")

    rows = load_manifest()
    todo = {v: [] for v in voices}
    for i, k, latex in lines:
        for v in voices:
            path = wav_path(v, i, k)
            row = {"id": i, "voice": v, "line": k, "latex": latex, "spoken": spoken[i, k],
                   "wav": str(path.relative_to(DATA_DIR.parent))}
            if path.exists():   # resume: keep the wav, refresh the manifest row
                rows[i, v, k] = {**row, "duration_s": round(sf.info(str(path)).duration, 3)}
            else:
                todo[v].append(row)
    save_manifest(rows)
    print("to synthesise:", {v: len(t) for v, t in todo.items()})

    def run_voice(v):
        # one worker per voice, in parallel (~3.4 GiB GPU each)
        stats = [0.0, 0.0, 0]
        for start in range(0, len(todo[v]), CHUNK):
            chunk = todo[v][start:start + CHUNK]
            res = synth_batch([{"text": tts_text(r["spoken"]), "voice": v,
                                "out_path": DATA_DIR.parent / r["wav"]} for r in chunk])
            for r, out in zip(chunk, res):
                if "error" in out:
                    stats[2] += 1
                    print(f"FAILED {r['id']} {v} line {r['line']}: {out['error']}")
                    continue
                rows[r["id"], v, r["line"]] = {**r, "duration_s": round(out["audio_s"], 3)}
                stats[0] += out["audio_s"]
                stats[1] += out["compute_s"]
            save_manifest(rows)   # progress survives an interrupted run
            print(f"{v}: {min(start + CHUNK, len(todo[v]))}/{len(todo[v])} lines", flush=True)
        return stats

    with ThreadPoolExecutor(len(voices)) as pool:
        for v, (audio_s, compute_s, failed) in zip(voices, pool.map(run_voice, voices)):
            rtf = compute_s / audio_s if audio_s else float("nan")
            print(f"{v}: {audio_s:.0f} s audio, {compute_s:.0f} s compute (RTF {rtf:.2f}), {failed} failed")
    print(f"manifest: {len(rows)} rows -> {MANIFEST}")


if __name__ == "__main__":
    main()
