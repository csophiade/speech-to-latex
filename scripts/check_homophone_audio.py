"""Check that the homophone error lines are verbalised, and heard, with the SAID number.

Usage: .venv/bin/python -W ignore scripts/check_homophone_audio.py [--text-only] [--items-file F]
--text-only: check the LLM verbalisations in the manifest (before/without audio QC).
Otherwise also checks Whisper best-1 (qc_asr, written by scripts/tts_redo.py) on every error-line clip:
"heard said" = the said number word is present and the correct one absent (unless the correct number
also occurs elsewhere in the said line). Prints counts per voice and lists failures.
"""

import argparse
import json
import re
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.config import DATA_DIR  # noqa: E402
from src.homophone import swap_sites  # noqa: E402
from src.spoken_norm import normalise_spoken, number_words  # noqa: E402

HOM = DATA_DIR / "derivations" / "homophone.json"
MANIFEST = DATA_DIR / "derivations" / "audio" / "manifest.jsonl"
_UNITS = r"(?: (?:one|two|three|four|five|six|seven|eight|nine))?"


def count_word(text, n):
    """Occurrences of number n in normalised text (a -ty word followed by a unit is another number)."""
    w = number_words(n)
    tail = r"(?! (?:one|two|three|four|five|six|seven|eight|nine)\b)" if n % 10 == 0 else ""
    return len(re.findall(rf"(?<![a-z])(?<!hundred ){w}(?![a-z]){tail}", normalise_spoken(text)))


def hears_said(text, item):
    """(said heard, correct heard where it should not be)."""
    h, line = item["homophone"], item["lines"][-1]
    want_said = sum(v == h["said"] for *_, v in swap_sites(line))
    want_correct = sum(v == h["correct"] for *_, v in swap_sites(line))
    return count_word(text, h["said"]) >= want_said, count_word(text, h["correct"]) > want_correct


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--text-only", action="store_true")
    ap.add_argument("--items-file", type=Path, default=HOM, help="e.g. data/derivations/homophone_fresh.json")
    args = ap.parse_args()
    items = {it["id"]: it for it in json.loads(args.items_file.read_text())}
    rows = [r for r in map(json.loads, MANIFEST.read_text().splitlines())
            if r["id"] in items and r["line"] == items[r["id"]]["error"]["line"]]
    stats, bad = Counter(), []
    for r in rows:
        it = items[r["id"]]
        for field in ("spoken",) + (() if args.text_only else ("qc_asr",)):
            if field not in r:
                stats[r["voice"], field, "missing"] += 1
                continue
            ok, wrong = hears_said(r[field], it)
            tag = "said" if ok and not wrong else ("correct" if wrong else "neither")
            stats[r["voice"], field, tag] += 1
            if tag != "said":
                bad.append((r["voice"], field, r["id"], it["homophone"]["said"], it["homophone"]["correct"], r[field]))
    print(f"{len(rows)} error-line clips ({len(items)} items)")
    for k in sorted(stats):
        print(" ", *k, stats[k])
    for b in bad:
        print("  FAIL", *b)


if __name__ == "__main__":
    main()
