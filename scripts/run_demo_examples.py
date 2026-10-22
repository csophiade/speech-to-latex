"""Run every canned demo example through the live pipeline and record how each line ACTUALLY resolves.

Usage: .venv/bin/python -W ignore scripts/run_demo_examples.py [--only NAME ...]
Writes demo/examples/<name>/resolved.json (per line: status, heard, output, options, explanation,
suggestion, spoken explanation, top candidates, latency) and prints a summary table. Lines are run
sequentially against the pipeline's own accepted outputs, exactly as the demo does; an ask_back is
left unresolved (its output is ĉ), as if the user had not clicked yet.
"""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src import config  # noqa: E402
from src.audio import load_audio  # noqa: E402
from src.pipeline import LinePipeline  # noqa: E402

EXAMPLES = config.ROOT / "demo" / "examples"


def run_example(pipe, d):
    meta = json.loads((d / "example.json").read_text())
    history, out = [], []
    for ln in meta["lines"]:
        audio, sr = load_audio(d / ln["wav"])
        res = pipe.process_line(audio, sr, history)
        history.append(res)
        out.append({"k": ln["k"], "said": ln["said"], "purpose": ln["purpose"], **res})
    (d / "resolved.json").write_text(json.dumps(out, indent=1, ensure_ascii=False) + "\n")
    return meta, out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", nargs="+")
    a = ap.parse_args()
    pipe = LinePipeline()
    for d in sorted(EXAMPLES.iterdir()):
        if not (d / "example.json").exists() or (a.only and d.name not in a.only):
            continue
        meta, out = run_example(pipe, d)
        print(f"\n== {d.name}: {meta['title']}")
        for r in out:
            extra = f" options={r['options']}" if r["options"] else ""
            extra += f" suggestion={r['suggestion']['latex']} (Δ={r['suggestion']['delta']:.3f})" if r["suggestion"] else ""
            print(f"  {r['k']} [{r['status']:>10}] said={r['said']!r} heard={r['heard']!r} out={r['output']!r}"
                  f"{extra} | {r['latency_s']:.2f}s | {r['explanation'] or ''}")
            for c in r["candidates"]:
                print(f"       Δ={c['delta']:.3f} {c['status']:>9} hom={c['hom_class']} aug={c['augmented']} {c['latex']}")


if __name__ == "__main__":
    main()
