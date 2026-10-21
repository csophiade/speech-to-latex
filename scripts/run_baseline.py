"""Systems A and B on one dataset.

A: Whisper (beam 5, best-1) -> LLM -> first LaTeX candidate.
B: as A, but take the first candidate that SymPy parses; if none parse, A's output marked `unverified`.

Usage: .venv/bin/python scripts/run_baseline.py --system A --dataset s2l_test500
Appends to results/results.json; writes results/preds/<system>_<dataset>.jsonl and
results/reports/<system>_<dataset>.md (10 random outputs vs ground truth).

ASR n-best and LLM responses are cached under data/cache/, together with the wall-clock
time of the original (uncached) call, so latency stays honest on re-runs.
"""

import argparse
import json
import random
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src import config  # noqa: E402
from src.candidates import to_latex_timed  # noqa: E402
from src.eval import append_result, latency, nem, summarise  # noqa: E402
from src.mathparse import parses  # noqa: E402

REF_KEY = {"s2l_test500": "sentence", "s2l_dev": "sentence", "s2l_test": "sentence",
           "mathspeech": "LaTeX"}


def load_rows(dataset, limit=None):
    folder = config.DATASETS[dataset]
    rows = [json.loads(l) for l in open(folder / "metadata.jsonl")]
    return rows[:limit] if limit else rows


def asr_nbest(dataset, rows):
    """{file_name: {"nbest": [[text, logprob], ...], "seconds": s}}, cached on disk."""
    slug = config.WHISPER_MODEL.split("/")[-1]
    path = config.DATA_DIR / "cache" / "asr" / f"{slug}_k{config.K_BEAMS}" / f"{dataset}.jsonl"
    path.parent.mkdir(parents=True, exist_ok=True)
    cache = {}
    if path.exists():
        for line in open(path):
            r = json.loads(line)
            cache[r["file_name"]] = r
    todo = [r for r in rows if r["file_name"] not in cache]
    if todo:
        import numpy as np

        from src.asr import transcribe_nbest
        from src.audio import load_audio
        transcribe_nbest(np.zeros(config.SAMPLE_RATE, dtype=np.float32))   # warm-up, untimed
        folder = config.DATASETS[dataset]
        with open(path, "a") as f:
            for r in tqdm(todo, desc=f"ASR {dataset}"):
                with latency() as t:
                    audio, _ = load_audio(folder / r["file_name"])
                    nbest = transcribe_nbest(audio)
                entry = {"file_name": r["file_name"], "nbest": nbest, "seconds": t.seconds}
                f.write(json.dumps(entry, ensure_ascii=False) + "\n")
                cache[r["file_name"]] = entry
    return cache


def run(system, dataset, llm, limit=None, workers=8):
    rows = load_rows(dataset, limit)
    asr = asr_nbest(dataset, rows)
    transcripts = [asr[r["file_name"]]["nbest"][0][0] for r in rows]

    with ThreadPoolExecutor(workers) as pool:
        llm_out = list(tqdm(pool.map(lambda t: to_latex_timed(t, model=llm), transcripts),
                            total=len(rows), desc=f"LLM {dataset}"))

    preds = []
    for r, transcript, (cands, llm_s) in zip(rows, transcripts, llm_out):
        with latency() as t:
            first = cands[0] if cands else ""
            first_parses = parses(first)
            pred, status = first, None
            if system == "B":
                pred = next((c for c in cands if parses(c)), None)
                status = "verified" if pred is not None else "unverified"
                pred = pred if pred is not None else first
        preds.append({
            "file_name": r["file_name"], "reference": r[REF_KEY[dataset]], "transcript": transcript,
            "candidates": cands, "pred": pred, "status": status, "first_parses": first_parses,
            "is_tts": r.get("is_tts"),
            "latency_s": asr[r["file_name"]]["seconds"] + llm_s + t.seconds,
        })

    metrics, verdicts = summarise([p["reference"] for p in preds], [p["pred"] for p in preds],
                                  [p["latency_s"] for p in preds])
    for p, v in zip(preds, verdicts):
        p["symeq"], p["nem"] = v, nem(p["reference"], p["pred"])
    n = len(preds)
    metrics["parse_fail_first_candidate"] = round(sum(not p["first_parses"] for p in preds) / n, 4)
    metrics["output_unparseable"] = round(sum(not parses(p["pred"]) for p in preds) / n, 4)
    metrics["no_candidates"] = sum(not p["candidates"] for p in preds)
    if system == "B":
        metrics["unverified_rate"] = round(sum(p["status"] == "unverified" for p in preds) / n, 4)
    if any(p["is_tts"] is not None for p in preds):
        for name, flag in [("human", 0), ("tts", 1)]:
            sub = [p for p in preds if p["is_tts"] == flag]
            metrics[f"symeq_{name}"] = round(sum(p["symeq"] == "equal" for p in sub) / len(sub), 4)
            metrics[f"n_{name}"] = len(sub)

    out = config.RESULTS_DIR
    (out / "preds").mkdir(parents=True, exist_ok=True)
    (out / "reports").mkdir(parents=True, exist_ok=True)
    with open(out / "preds" / f"{system}_{dataset}.jsonl", "w") as f:
        for p in preds:
            f.write(json.dumps(p, ensure_ascii=False) + "\n")
    write_report(out / "reports" / f"{system}_{dataset}.md", system, dataset, llm, metrics, preds)
    if not limit:
        append_result(system, dataset, metrics, llm=llm, asr=config.WHISPER_MODEL, seed=config.SEED)
    return metrics


def write_report(path, system, dataset, llm, metrics, preds):
    sample = random.Random(config.SEED).sample(preds, min(10, len(preds)))
    esc = lambda s: (s or "").replace("|", "\\|").replace("\n", " ")
    lines = [f"# System {system} on {dataset} (LLM: {llm}, n={metrics['n']})", "",
             "```json", json.dumps(metrics, indent=2), "```", "",
             "10 random outputs (seed 42):", "",
             "| # | Whisper best-1 | Prediction | Ground truth | NEM | SymEq |",
             "|---|---|---|---|---|---|"]
    for i, p in enumerate(sample, 1):
        lines.append(f"| {i} | {esc(p['transcript'])} | `{esc(p['pred'])}` | `{esc(p['reference'])}` "
                     f"| {'✓' if p['nem'] else '✗'} | {p['symeq']} |")
    path.write_text("\n".join(lines) + "\n")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--system", choices=["A", "B"], required=True)
    ap.add_argument("--dataset", choices=list(REF_KEY), required=True)
    ap.add_argument("--llm", default=config.LLM_MODEL)
    ap.add_argument("--limit", type=int, help="debug: first N clips only, not logged to results.json")
    ap.add_argument("--workers", type=int, default=8)
    args = ap.parse_args()
    print(json.dumps(run(args.system, args.dataset, args.llm, args.limit, args.workers), indent=2))
