"""Paired comparison of two systems from their saved per-clip predictions.

Usage: .venv/bin/python scripts/compare_systems.py [--a C --b D] [--datasets s2l_test500 mathspeech]

For each dataset: CER(b) - CER(a) with a 95% paired bootstrap CI (1000 resamples, seed 42),
how often the systems pick different outputs, and every clip where exactly one of them is
an exact (normalised) match. Reads results/preds/<system>_<dataset>.jsonl (written by
scripts/run_ablation.py); writes results/compare_<a>_vs_<b>.json and
results/reports/compare_<a>_vs_<b>.md.
"""

import argparse
import json
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src import config  # noqa: E402
from src.eval import cer  # noqa: E402

N_BOOT = 1000


def load(system, dataset):
    path = config.RESULTS_DIR / "preds" / f"{system}_{dataset}.jsonl"
    return [json.loads(line) for line in open(path)]


def compare(a, b, dataset):
    pa, pb = load(a, dataset), load(b, dataset)
    assert [p["file_name"] for p in pa] == [p["file_name"] for p in pb], "prediction files are not aligned"
    refs = [p["reference"] for p in pa]
    ha, hb = [p["pred"] for p in pa], [p["pred"] for p in pb]

    rng = random.Random(config.SEED)
    diffs = []
    for _ in range(N_BOOT):
        idx = [rng.randrange(len(refs)) for _ in refs]
        r = [refs[i] for i in idx]
        diffs.append(cer(r, [hb[i] for i in idx]) - cer(r, [ha[i] for i in idx]))
    diffs.sort()

    def clip(x, y):
        return {"file_name": x["file_name"], "is_tts": x.get("is_tts"), "transcript": x["transcript"],
                "reference": x["reference"], a: x["pred"], b: y["pred"]}

    only_b = [clip(x, y) for x, y in zip(pa, pb) if y["nem"] and not x["nem"]]
    only_a = [clip(x, y) for x, y in zip(pa, pb) if x["nem"] and not y["nem"]]
    return {
        "dataset": dataset, "n": len(refs),
        f"cer_{a}": round(cer(refs, ha), 4), f"cer_{b}": round(cer(refs, hb), 4),
        "cer_diff_b_minus_a": round(cer(refs, hb) - cer(refs, ha), 4),
        "ci95": [round(diffs[int(0.025 * N_BOOT)], 4), round(diffs[int(0.975 * N_BOOT) - 1], 4)],
        "n_bootstrap": N_BOOT, "seed": config.SEED,
        "n_differ": sum(x != y for x, y in zip(ha, hb)),
        f"exact_only_{b}": only_b, f"exact_only_{a}": only_a,
    }


def write_report(path, a, b, results):
    esc = lambda s: (s or "").replace("|", "\\|").replace("\n", " ")
    lines = [f"# {a} vs {b}: paired comparison", "",
             f"CER difference = CER({b}) − CER({a}); negative favours {b}. 95% CI from a paired bootstrap "
             f"({N_BOOT} resamples, seed {config.SEED}). Exact match = normalised exact match (NEM).", "",
             f"| dataset | n | CER {a} | CER {b} | CER({b})−CER({a}) | 95% CI | outputs differ | exact only {b} | exact only {a} |",
             "|---|---|---|---|---|---|---|---|---|"]
    for r in results:
        lines.append(f"| {r['dataset']} | {r['n']} | {r[f'cer_{a}']} | {r[f'cer_{b}']} | {r['cer_diff_b_minus_a']:+} | "
                     f"[{r['ci95'][0]:+}, {r['ci95'][1]:+}] | {r['n_differ']} | {len(r[f'exact_only_{b}'])} | "
                     f"{len(r[f'exact_only_{a}'])} |")
    for r in results:
        for winner in (b, a):
            clips = r[f"exact_only_{winner}"]
            lines += ["", f"## {r['dataset']}: exact only with {winner} ({len(clips)} clips)", "",
                      f"| clip | Whisper best-1 | reference | {a} | {b} |", "|---|---|---|---|---|"]
            for c in clips:
                lines.append(f"| {c['file_name']} | {esc(c['transcript'])} | `{esc(c['reference'])}` | "
                             f"`{esc(c[a])}` | `{esc(c[b])}` |")
    path.write_text("\n".join(lines) + "\n")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--a", default="C")
    ap.add_argument("--b", default="D")
    ap.add_argument("--datasets", nargs="+", default=["s2l_test500", "mathspeech"])
    args = ap.parse_args()
    results = [compare(args.a, args.b, d) for d in args.datasets]
    stem = f"compare_{args.a}_vs_{args.b}"
    (config.RESULTS_DIR / f"{stem}.json").write_text(json.dumps(results, indent=2, ensure_ascii=False) + "\n")
    write_report(config.RESULTS_DIR / "reports" / f"{stem}.md", args.a, args.b, results)
    for r in results:
        print(r["dataset"], r["cer_diff_b_minus_a"], r["ci95"], "differ", r["n_differ"],
              f"exact only {args.b}", len(r[f"exact_only_{args.b}"]), f"exact only {args.a}", len(r[f"exact_only_{args.a}"]))


if __name__ == "__main__":
    main()
