"""Canned demo derivations (Phase 5): demo/examples/<name>/line_<k>.wav + example.json.

Synthetic examples are read by voice v1 (src/tts, CosyVoice2, batch); recorded examples copy the
owner's line clips from data/recordings/lines. example.json holds, per line, what was SAID (LaTeX
and spoken text) and what the demo is meant to show; how each line actually resolves is recorded
by scripts/run_demo_examples.py, not here.

Usage: .venv/bin/python scripts/make_demo_examples.py [--only NAME ...] [--force]
"""

import argparse
import json
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src import config  # noqa: E402

OUT = config.ROOT / "demo" / "examples"
RECS = config.DATA_DIR / "recordings" / "manifest.jsonl"

# (said LaTeX, spoken text, purpose)
SYNTHETIC = {
    "quadratic": {
        "title": "Quadratic: divide by x, then a planted fifteen/fifty",
        "lines": [
            ("x^3 + 5x^2 = 50x", "x cubed plus five x squared equals fifty x", "problem"),
            ("x^2 + 5x = 50", "x squared plus five x equals fifty",
             "divide-by-x: expected red flag 'dividing by x may lose the solution x = 0'"),
            ("x^2 + 5x - 15 = 0", "x squared plus five x minus fifteen equals zero",
             "planted fifteen/fifty: the speaker says fifteen; fifty makes the step valid"),
            ("(x + 10)(x - 5) = 0", "the quantity x plus ten times the quantity x minus five equals zero", "valid"),
            ("x = -10 \\text{ or } x = 5", "x equals negative ten or x equals five", "valid"),
        ],
    },
    # ask-back: the said line has two confusable numbers and EACH single swap gives a valid,
    # non-equivalent step, so both readings are valid; it fires only if both are within delta_hom
    "askback": {
        "title": "Ask-back: fifty times the quantity ... equals fifteen times fifty",
        "lines": [
            ("15x^2 + 75x = 750", "fifteen x squared plus seventy five x equals seven hundred fifty", "problem"),
            ("50(x^2 + 5x) = 15 \\cdot 50",
             "fifty times the quantity x squared plus five x equals fifteen times fifty",
             "ask-back attempt: 15(x^2 + 5x) = 15 * 50 and 50(x^2 + 5x) = 50 * 50 are both valid"),
            ("x^2 + 5x = 50", "x squared plus five x equals fifty", "valid"),
        ],
    },
}

# ask-back attempts that did NOT fire with voice v1 (both swaps scored Δ > 0.9; see demo/script.md);
# build with --only
ATTEMPTS = {
    "askback_16_60": {
        "title": "Ask-back attempt: sixty x squared equals sixteen",
        "lines": [
            ("16x^2 - 16 = 0", "sixteen x squared minus sixteen equals zero", "problem"),
            ("60x^2 = 16", "sixty x squared equals sixteen",
             "ask-back attempt: 16x^2 = 16 and 60x^2 = 60 are both valid one-swap readings"),
            ("x^2 = 1", "x squared equals one", "valid"),
            ("x = 1 \\text{ or } x = -1", "x equals one or x equals negative one", "valid"),
        ],
    },
    "askback_14_40": {
        "title": "Ask-back attempt: fourteen x squared equals forty",
        "lines": [
            ("14x^2 = 14", "fourteen x squared equals fourteen", "problem"),
            ("14x^2 = 40", "fourteen x squared equals forty",
             "ask-back attempt: 14x^2 = 14 and 40x^2 = 40 are both valid one-swap readings"),
            ("x^2 = 1", "x squared equals one", "valid"),
        ],
    },
}

RECORDED = {
    "rec_clean": ("d033", "Owner's recording: clean quadratic"),
    "rec_error": ("d031", "Owner's recording: quadratic with a sign slip in the factoring"),
}


def write(name, meta):
    d = OUT / name
    d.mkdir(parents=True, exist_ok=True)
    (d / "example.json").write_text(json.dumps(dict(meta, name=name), indent=1, ensure_ascii=False) + "\n")


def synthetic(names, force):
    from src.tts import synth_batch
    items, metas = [], {}
    for name in names:
        ex = SYNTHETIC[name]
        lines = [{"k": k, "wav": f"line_{k}.wav", "said": s, "spoken": sp, "purpose": p}
                 for k, (s, sp, p) in enumerate(ex["lines"])]
        metas[name] = {"title": ex["title"], "source": "synthetic, voice v1 (CosyVoice2)", "lines": lines}
        for ln in lines:
            out = OUT / name / ln["wav"]
            if force or not out.exists():
                items.append({"text": ln["spoken"], "voice": "v1", "out_path": str(out)})
    if items:
        for r in synth_batch(items):
            print(r, flush=True)
    for name, meta in metas.items():
        write(name, meta)


def recorded(names):
    rows = [json.loads(line) for line in open(RECS)]
    for name in names:
        rid, title = RECORDED[name]
        mine = sorted((r for r in rows if r["id"] == rid), key=lambda r: r["line"])
        err = mine[0]["error"]
        lines = []
        for r in mine:
            (OUT / name).mkdir(parents=True, exist_ok=True)
            shutil.copy(r["wav"], OUT / name / f"line_{r['line']}.wav")
            purpose = "problem" if r["line"] == 0 else (
                f"deliberate error ({err['type']})" if err and r["line"] == err["line"] else "valid")
            lines.append({"k": r["line"], "wav": f"line_{r['line']}.wav", "said": r["latex"],
                          "spoken": r["spoken"], "purpose": purpose})
        write(name, {"title": title, "source": f"owner's recording ({rid})", "lines": lines})


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", nargs="+")
    ap.add_argument("--force", action="store_true")
    a = ap.parse_args()
    names = a.only or list(SYNTHETIC) + list(RECORDED)
    SYNTHETIC.update(ATTEMPTS)
    recorded([n for n in names if n in RECORDED])
    synthetic([n for n in names if n in SYNTHETIC], a.force)


if __name__ == "__main__":
    main()
