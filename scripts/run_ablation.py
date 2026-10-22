"""Full single-equation ablation: Systems A, B, C, D on en_test_500 and MathSpeech.

Usage: .venv/bin/python scripts/run_ablation.py [--datasets s2l_test500 mathspeech]
λ and β come from results/tuned.json (written by scripts/tune_cd.py on en_dev).

Pools (n-best transcripts -> LaTeX candidates -> spoken forms -> raw acoustic scores) are
cached per dataset under data/cache/pools/, with the wall-clock time of every original call,
so selection and metrics can be recomputed for free. Appends to results/results.json and
writes results/figures/ablation_bar.png.

Phase 2 numbers are PRE-AUGMENTATION and frozen: pools here are built with augment=False
(confusable-number augmentation, src/augment.py, belongs to the Phase 4 derivation system),
so the cached pools, tuned.json and the single-equation ablation stay as they were.
"""

import argparse
import json
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from run_baseline import REF_KEY, asr_nbest, load_rows, write_report  # noqa: E402

from src import config  # noqa: E402
from src.eval import append_result, nem, summarise  # noqa: E402
from src.rerank import build_pool, fill_levels, score_pool, select  # noqa: E402

SYSTEMS = ["A", "B", "C", "D"]
TUNED = config.RESULTS_DIR / "tuned.json"


def pool_path(dataset, llm):
    # v2: plain + explicit verbalisations (v1 pools used free-form verbalisations)
    return config.DATA_DIR / "cache" / "pools" / "v2" / llm / f"{dataset}.jsonl"


def ensure_pools(dataset, llm=config.LLM_MODEL, limit=None, clip_workers=6):
    """{file_name: {"nbest", "pool", "timing"}} for every clip, building what is missing."""
    rows = load_rows(dataset, limit)
    asr = asr_nbest(dataset, rows)
    path = pool_path(dataset, llm)
    path.parent.mkdir(parents=True, exist_ok=True)
    cache = {}
    if path.exists():
        for line in open(path):
            r = json.loads(line)
            cache[r["file_name"]] = r
    todo = [r for r in rows if r["file_name"] not in cache]
    if todo:
        import numpy as np

        from src.audio import load_audio
        # LLM stage (network-bound): many clips at once
        with ThreadPoolExecutor(clip_workers) as ex:
            built = list(tqdm(ex.map(lambda r: build_pool(asr[r["file_name"]]["nbest"], llm, augment=False), todo),
                              total=len(todo), desc=f"LLM pools {dataset}"))
        # GPU stage: one clip at a time, after an untimed warm-up so model loading isn't counted
        score_pool(np.zeros(config.SAMPLE_RATE, dtype=np.float32),
                   [{"spoken": ["warm up"], "latex": "x", "level": "clean"}])
        with open(path, "a") as f:
            for r, (pool, timing) in tqdm(list(zip(todo, built)), desc=f"Acoustic {dataset}"):
                audio, _ = load_audio(config.DATASETS[dataset] / r["file_name"])
                fill_levels(pool)   # main thread: parse timeouts apply
                timing.update(score_pool(audio, pool))
                timing["asr_s"] = asr[r["file_name"]]["seconds"]
                entry = {"file_name": r["file_name"], "nbest": asr[r["file_name"]]["nbest"],
                         "pool": pool, "timing": timing}
                f.write(json.dumps(entry, ensure_ascii=False) + "\n")
                cache[r["file_name"]] = entry
    for entry in cache.values():   # levels from older parser versions may be stale
        fill_levels(entry["pool"])
    return rows, cache


def latency(system, timing, select_s):
    t = timing["asr_s"] + select_s
    if system in ("A", "B"):
        return t + timing["llm_top_s"]
    if system == "C":
        return t + timing["llm_latex_s"]
    return t + timing["llm_latex_s"] + timing["llm_spoken_s"] + timing["phoneme_s"]


def evaluate(system, dataset, rows, pools, lam=1.0, gamma=0.25, eps=0.05):
    """Per-clip predictions for one system (no logging)."""
    preds = []
    for r in rows:
        entry = pools[r["file_name"]]
        pool = entry["pool"]
        start = time.perf_counter()
        i = select(system, pool, lam, gamma, eps)
        select_s = time.perf_counter() - start
        preds.append({
            "file_name": r["file_name"], "reference": r[REF_KEY[dataset]],
            "transcript": entry["nbest"][0][0] if entry["nbest"] else "",
            "pred": pool[i]["latex"] if i is not None else "",
            "level": pool[i]["level"] if i is not None else "none",
            "src_rank": pool[i]["src_rank"] if i is not None else None,
            "pool_size": len(pool), "is_tts": r.get("is_tts"),
            "oracle_nem": any(nem(r[REF_KEY[dataset]], p["latex"]) for p in pool),
            "latency_s": latency(system, entry["timing"], select_s),
        })
    return preds


def score(preds):
    metrics, verdicts = summarise([p["reference"] for p in preds], [p["pred"] for p in preds],
                                  [p["latency_s"] for p in preds])
    for p, v in zip(preds, verdicts):
        p["symeq"], p["nem"] = v, nem(p["reference"], p["pred"])
    n = len(preds)
    metrics["chosen_parses"] = round(sum(p["level"] != "none" for p in preds) / n, 4)
    metrics["chosen_from_beam_gt1"] = round(sum((p["src_rank"] or 0) > 0 for p in preds) / n, 4)
    metrics["pool_size_mean"] = round(sum(p["pool_size"] for p in preds) / n, 2)
    metrics["oracle_nem"] = round(sum(p["oracle_nem"] for p in preds) / n, 4)
    if any(p["is_tts"] is not None for p in preds):
        for name, flag in [("human", 0), ("tts", 1)]:
            sub = [p for p in preds if p["is_tts"] == flag]
            metrics[f"n_{name}"] = len(sub)
            metrics[f"cer_{name}"] = summarise([p["reference"] for p in sub], [p["pred"] for p in sub],
                                               [p["latency_s"] for p in sub])[0]["cer"]
    return metrics


def plot(rows, path):
    """Grouped bars: CER and SymEq on the parseable subset, with the SymEq coverage in the caption."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    datasets = list(dict.fromkeys(r["dataset"] for r in rows))
    metric = {(r["system"], r["dataset"]): r["metrics"] for r in rows}
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.6))
    width = 0.8 / len(SYSTEMS)
    for ax, key, title in [(axes[0], "cer", "CER (lower is better)"),
                           (axes[1], "symeq_on_parseable", "SymEq on parseable subset (higher is better)")]:
        for k, system in enumerate(SYSTEMS):
            vals = [metric[(system, d)][key] for d in datasets]
            xs = [i + (k - (len(SYSTEMS) - 1) / 2) * width for i in range(len(datasets))]
            ax.bar(xs, vals, width, label=system)
            for x, v in zip(xs, vals):
                ax.text(x, v, f"{v:.3f}", ha="center", va="bottom", fontsize=7)
        ns = {d: metric[(SYSTEMS[0], d)]["n"] for d in datasets}
        ax.set_xticks(range(len(datasets)), [f"{d}\n(n={ns[d]})" for d in datasets])
        ax.set_title(title, fontsize=10)
        ax.margins(y=0.12)
    axes[1].legend(title="System", loc="upper left", fontsize=8)
    coverage = ", ".join(f"{d} {metric[(SYSTEMS[0], d)]['symeq_coverage']:.2%}" for d in datasets)
    fig.text(0.5, 0.01, "SymEq is computed only on clips whose reference LaTeX parses with SymPy "
             f"(coverage: {coverage}); an unparseable prediction counts as wrong. CER uses all clips.",
             ha="center", fontsize=8, wrap=True)
    fig.tight_layout(rect=(0, 0.06, 1, 1))
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150)


def latest_ablation_rows(datasets):
    """Most recent logged ablation row per (system, dataset), current pool version only."""
    rows = [r for r in json.loads(config.RESULTS_JSON.read_text())
            if r.get("ablation") and r.get("pool_version") == "v2" and r["dataset"] in datasets]
    latest = {(r["system"], r["dataset"]): r for r in rows}
    return list(latest.values())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--datasets", nargs="+", default=["s2l_test500", "mathspeech"])
    ap.add_argument("--llm", default=config.LLM_MODEL)
    ap.add_argument("--limit", type=int, help="debug: first N clips, nothing logged")
    ap.add_argument("--plot-only", action="store_true", help="redraw the figure from results.json")
    args = ap.parse_args()
    if args.plot_only:
        plot(latest_ablation_rows(args.datasets), config.FIGURES_DIR / "ablation_bar.png")
        return

    tuned = json.loads(TUNED.read_text())
    params = {"A": {}, "B": {}, "C": {"lam": tuned["C"]["lam"]},
              "D": {"lam": tuned["D"]["lam"], "gamma": tuned["D"]["gamma"], "eps": tuned["D"]["eps"]}}
    logged = []
    for dataset in args.datasets:
        rows, pools = ensure_pools(dataset, args.llm, args.limit)
        for system in SYSTEMS:
            preds = evaluate(system, dataset, rows, pools, **params[system])
            metrics = score(preds)
            print(system, dataset, json.dumps(metrics))
            out = config.RESULTS_DIR
            (out / "preds").mkdir(parents=True, exist_ok=True)
            (out / "reports").mkdir(parents=True, exist_ok=True)
            with open(out / "preds" / f"{system}_{dataset}.jsonl", "w") as f:
                for p in preds:
                    f.write(json.dumps(p, ensure_ascii=False) + "\n")
            write_report(out / "reports" / f"{system}_{dataset}.md", system, dataset, args.llm, metrics,
                         [{**p, "candidates": None} for p in preds])
            row = {"system": system, "dataset": dataset, "metrics": metrics, "n": metrics["n"]}
            if not args.limit:
                append_result(system, dataset, metrics, llm=args.llm, asr=config.WHISPER_MODEL,
                              seed=config.SEED, ablation=True, pool_version="v2", **params[system])
            logged.append(row)
    if not args.limit:
        plot(logged, config.FIGURES_DIR / "ablation_bar.png")


if __name__ == "__main__":
    main()
