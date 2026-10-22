"""Derivation mode (Phase 4): System D + checker + decision rule on every derivation line clip.

Usage:
  .venv/bin/python scripts/run_derivations.py stage1 [--limit N]     # GPU + LLM, resumable
  .venv/bin/python scripts/run_derivations.py stage2 --delta 0.0     # CPU, per-line records
  .venv/bin/python scripts/run_derivations.py augment [--limit N]    # migrate cached pools (GPU)

Stage 1 (per clip, cached as one JSON line in data/cache/derivations/<llm>/stage1.jsonl):
Whisper 5-best -> build_pool(context_vars) -> score_pool. Clips are sources v1/v2 (synthetic
voices, data/derivations) and rec (the owner's recordings), in priority order rec, synthetic
calib, synthetic test. LLM work runs across clips in threads; GPU work stays on the main thread.

context_vars: a real system would pass the variables of its own previous output line, which
would make stage 1 depend on the decisions of stage 2. The variable set is fixed per problem
and is equally available from line 0's own output, so we use the variables of the ground-truth
problem statement (line 0) for lines >= 1 and none for line 0. This keeps stage 1 independent
of δ and of every decision.

Augmentation (src/augment.py, config.AUGMENT_CONFUSABLE): new stage-1 rows are built with
confusable-number variants (augmented=true). The "augment" stage migrates rows cached before
that: it adds the variants to the stored pool, scores only their new spoken forms on the GPU
and APPENDS the updated row (read_cache keeps the last row per key, so the run is resumable
and the pre-augmentation row stays in the file). Sources: --voices (default v1 v2) plus rec;
v3 (accented) is not migrated and keeps its pre-augmentation pools. Timing: the variants'
GPU seconds are added to whisper_s/phoneme_s (a live system scores the whole pool in one
batch, so this slightly overstates latency); the pre-augmentation values are kept as
*_pre_aug and the added part as aug_*.

Fresh confirmatory homophone set (results/reports/prereg_homophone.md): pass --items-file
data/derivations/homophone_fresh.json to stage1; stage 2 runs via scripts/run_homfresh.py with the
class-conditional rule (delta_hom for homophone-class candidates, decide.tag_homophone).

Stage 2 (sequential per derivation, derivations in parallel processes): frozen System D scores,
Delta(c) = D(ĉ) - D(c), checker status of every candidate against the system's own previous
output, decide() at δ. Writes data/cache/derivations/<llm>/records_<delta>.jsonl: the line
record format of src/calibrate.py plus the decision fields and latency parts.
"""

import argparse
import json
import sys
import time
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from pathlib import Path

from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src import config  # noqa: E402

TUNED = config.RESULTS_DIR / "tuned.json"
DERIVS = config.DATA_DIR / "derivations" / "derivations.json"
HOMOPHONE = config.DATA_DIR / "derivations" / "homophone.json"   # stress set (hom-*), v1/v2 only
SYNTH = config.DATA_DIR / "derivations" / "audio" / "manifest.jsonl"
RECS = config.DATA_DIR / "recordings" / "manifest.jsonl"
VOICES = ("v1", "v2")   # clean synthetic voices; add "v3" (accented subset) with --voices


def cache_dir(llm):
    return config.DATA_DIR / "cache" / "derivations" / llm


def clip_key(c):
    return f'{c["source"]}/{c["id"]}/{c["line"]}'


# ---------- clips and ground truth ----------

def labels(n_lines, error_line):
    """Ground-truth step label per line: given / valid / error / follows_error."""
    out = []
    for i in range(n_lines):
        if i == 0:
            out.append("given")
        elif error_line is None or i < error_line:
            out.append("valid")
        else:
            out.append("error" if i == error_line else "follows_error")
    return out


def problem_vars(latex):
    """Variable names in a problem statement (line 0), or None if it cannot be read."""
    from src.checker import read_line
    try:
        r = read_line(latex)
    except Exception:
        return None
    rel = r["rel"] if r else None
    if rel is None:
        return None
    return sorted(str(s) for s in rel.free_symbols) or None


def load_clips(voices=VOICES, homophone=False, first_pass=True):
    """Every line clip (dicts: source, split, set, id, line, n_lines, said, wav, qc_wer, label, line0), priority order.
    voices: synthetic voices to include (the manifest also holds the accented subset, v3, once synthesised).
    homophone: also the homophone stress set (data/derivations/homophone.json, set "homophone"), or a
    Path: that derivations file instead (e.g. the fresh confirmatory set);
    first_pass=False: only the homophone set (no recordings, no first-pass derivations)."""
    derivs = {d["id"]: dict(d, set="first-pass") for d in json.loads(DERIVS.read_text())} if first_pass else {}
    hom_file = homophone if isinstance(homophone, Path) else HOMOPHONE if homophone else None
    if hom_file is not None and hom_file.exists():
        derivs.update({d["id"]: d for d in json.loads(hom_file.read_text())})
    clips = []
    for r in map(json.loads, open(RECS)) if first_pass else []:
        err = r["error"]["line"] if r["error"] else None
        clips.append({"source": "rec", "split": r["split"], "set": "first-pass", "id": r["id"], "line": r["line"],
                      "n_lines": r["n_lines"], "said": r["latex"], "wav": r["wav"], "qc_wer": r["wer"],
                      "label": labels(r["n_lines"], err)[r["line"]], "error_line": err})
    recs0 = {c["id"]: c["said"] for c in clips if c["line"] == 0}
    for c in clips:
        c["line0"] = recs0[c["id"]]
    synth = []
    for r in map(json.loads, open(SYNTH)):
        if r["voice"] not in voices or r["id"] not in derivs:   # hom-* only with homophone=True
            continue
        d = derivs[r["id"]]
        err = d["error"]["line"] if d["error"] else None
        synth.append({"source": r["voice"], "split": d["split"], "set": d["set"], "id": r["id"], "line": r["line"],
                      "n_lines": len(d["lines"]), "said": r["latex"], "wav": r["wav"], "qc_wer": r.get("qc_wer"),
                      "label": labels(len(d["lines"]), err)[r["line"]], "error_line": err,
                      "line0": d["lines"][0]})
    synth.sort(key=lambda c: (c["split"] != "calib", c["id"], c["source"], c["line"]))
    return clips + synth


# ---------- stage 1 ----------

def read_cache(path):
    done = {}
    if path.exists():
        for line in open(path):
            try:
                r = json.loads(line)
            except json.JSONDecodeError:   # torn last line from a kill
                continue
            done[r["key"]] = r
    return done


def stage1(llm, limit=None, clip_workers=12, chunk=48, voices=VOICES, homophone=False):
    """ASR -> pool -> acoustic scores for every clip not yet cached."""
    import numpy as np

    from src.asr import transcribe_nbest
    from src.audio import load_audio
    from src.eval import latency
    from src.rerank import build_pool, score_pool

    out = cache_dir(llm) / "stage1.jsonl"
    out.parent.mkdir(parents=True, exist_ok=True)
    done = read_cache(out)
    clips = [c for c in load_clips(voices, homophone) if clip_key(c) not in done][:limit]
    print(f"stage1: {len(done)} cached, {len(clips)} to do", flush=True)
    if not clips:
        return
    vars_of = {}
    zero = np.zeros(config.SAMPLE_RATE, dtype=np.float32)
    transcribe_nbest(zero)   # warm-ups, untimed
    score_pool(zero, [{"spoken": ["warm up"], "latex": "x", "level": "clean"}])
    bar = tqdm(total=len(clips), desc="stage1", mininterval=30)
    with ThreadPoolExecutor(clip_workers) as ex, open(out, "a") as f:
        pending = []
        for s in range(0, len(clips) + chunk, chunk):
            batch = []
            for c in clips[s:s + chunk]:   # GPU: ASR on the main thread, LLM pools in the background
                path = Path(c["wav"]) if Path(c["wav"]).is_absolute() else config.ROOT / c["wav"]
                audio, _ = load_audio(path)
                with latency() as t:
                    nbest = transcribe_nbest(audio)
                if c["line0"] not in vars_of:
                    vars_of[c["line0"]] = problem_vars(c["line0"])
                ctx = vars_of[c["line0"]] if c["line"] > 0 else None
                batch.append((c, audio, nbest, t.seconds, ctx, ex.submit(build_pool, nbest, llm, 8, ctx)))
            for item in pending:   # the previous chunk's pools: acoustic scores while this chunk's LLM runs
                _finish(f, *item)
                bar.update(1)
            pending = batch


def _finish(f, c, audio, nbest, asr_s, ctx, fut):
    """Score one clip's pool on the GPU and append it to the cache (a failure stays uncached)."""
    from src.rerank import fill_levels, score_pool
    try:
        pool, timing = fut.result()
    except Exception as e:
        print(f"FAILED {clip_key(c)}: {type(e).__name__}: {e}", flush=True)
        return
    fill_levels(pool)   # main thread: parse timeouts apply
    timing.update(score_pool(audio, pool))
    timing["asr_s"] = asr_s
    f.write(json.dumps({"key": clip_key(c), "context_vars": ctx, "nbest": nbest, "pool": pool,
                        "timing": timing, "augmented": config.AUGMENT_CONFUSABLE,
                        "n_aug": sum(1 for p in pool if p.get("augmented"))}, ensure_ascii=False) + "\n")
    f.flush()


# ---------- augmentation migration ----------

def augment_stage(llm, limit=None, voices=VOICES):
    """Add confusable-number variants (+ raw scores of their spoken forms) to cached rows."""
    import numpy as np

    from src.audio import load_audio
    from src.rerank import add_variants, score_pool

    out = cache_dir(llm) / "stage1.jsonl"
    done = read_cache(out)
    wav = {clip_key(c): c["wav"] for c in load_clips(voices)}
    todo = [r for k, r in done.items() if k in wav and not r.get("augmented")][:limit]
    skipped = sum(1 for k in done if k not in wav)
    print(f"augment: {len(done)} cached rows, {len(todo)} to migrate, "
          f"{len(done) - len(todo) - skipped} already augmented, {skipped} outside --voices/rec (left as is)",
          flush=True)
    if not todo:
        return
    score_pool(np.zeros(config.SAMPLE_RATE, dtype=np.float32),
               [{"spoken": ["warm up"], "latex": "x", "level": "clean"}])   # model load, untimed
    start, n_new, n_forms, n_before = time.perf_counter(), 0, 0, 0
    with open(out, "a") as f:
        for r in tqdm(todo, desc="augment", mininterval=30):
            pool = r["pool"]
            n_before += len(pool)
            new = add_variants(pool)   # main thread: levels with parse timeouts
            t = r["timing"]
            if new:
                path = Path(wav[r["key"]])
                audio, _ = load_audio(path if path.is_absolute() else config.ROOT / path)
                s = score_pool(audio, pool, only_missing=True)
                for name in ("whisper_s", "phoneme_s"):
                    t[name + "_pre_aug"] = t.get(name, 0.0)
                    t["aug_" + name] = s[name]
                    t[name] = t[name + "_pre_aug"] + s[name]
            r["augmented"], r["n_aug"] = True, len(new)
            n_new += len(new)
            n_forms += sum(len(p["spoken"]) for p in new)
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
            f.flush()
    secs = time.perf_counter() - start
    print(f"augment: {len(todo)} rows in {secs:.0f}s ({len(todo) / secs:.1f} rows/s); "
          f"pool size {n_before} -> {n_before + n_new} (+{n_new} candidates, {n_forms} new spoken forms)",
          flush=True)


# ---------- stage 2 ----------

def frozen_d():
    t = json.loads(TUNED.read_text())["D"]
    return {"lam": t["lam"], "gamma": t["gamma"], "eps": t["eps"]}


def line_candidates(pool, prev_out, line_index, params):
    """Pool candidates with delta, checker status and explanation, ĉ first; plus checker seconds."""
    from src.checker import check_step
    from src.rerank import select, system_scores

    if not pool:
        return [], 0.0
    scores = system_scores("D", pool, **params)
    hat = select("D", pool, **params)
    order = [hat] + [i for i in range(len(pool)) if i != hat]
    cands, check_s = [], 0.0
    for i in order:
        c = {"latex": pool[i]["latex"], "delta": max(0.0, scores[hat] - scores[i]),
             "augmented": bool(pool[i].get("augmented"))}
        if line_index == 0:
            c.update(status="given", explanation=None)
        else:
            start = time.perf_counter()
            r = check_step(prev_out, c["latex"])
            check_s += time.perf_counter() - start
            c.update(status=r["status"], explanation=r["explanation"])
        cands.append(c)
    return cands, check_s


def run_derivation(args):
    """Sequential decisions over one derivation's lines from one source -> list of records."""
    lines, entries, delta, params, *rest = args
    delta_hom = rest[0] if rest else None
    from src.decide import classes, decide, is_valid, tag_homophone
    from src.eval import symeq

    records, prev_out = [], None
    for c, entry in zip(lines, entries):
        if entry is None:   # missing clip: the derivation stops being checkable after it
            records.append({**_meta(c), "missing": True})
            prev_out = None
            continue
        cands, check_s = line_candidates(entry["pool"], prev_out, c["line"], params)
        if not cands:
            records.append({**_meta(c), "missing": False, "empty_pool": True, "status": "unchecked",
                            "output": "", "heard": "", "explanation": "no candidates",
                            "c_hat": {"latex": "", "status": "unchecked", "eq_said": False}, "cands": []})
            continue
        if c["line"] > 0 and prev_out is None:
            for x in cands:
                x.update(status="unchecked", explanation="no earlier line to compare with")
        cls = classes([x["latex"] for x in cands])
        for x, k in zip(cands, cls):
            x["cls"] = k
        if delta_hom is not None:
            cands = tag_homophone(cands)
        start = time.perf_counter()
        d = decide(cands, delta, c["line"], delta_hom=delta_hom)
        decide_s = time.perf_counter() - start
        eq = [symeq(c["said"], x["latex"]) == "equal" for x in cands]
        t = entry["timing"]
        lat = {"asr_s": t["asr_s"], "llm_latex_s": t["llm_latex_s"], "llm_spoken_s": t["llm_spoken_s"],
               "phoneme_s": t["phoneme_s"], "checker_s": check_s, "decide_s": decide_s}
        records.append({
            **_meta(c), "missing": False,
            "c_hat": {"latex": cands[0]["latex"], "status": cands[0]["status"], "eq_said": eq[0],
                      "explanation": cands[0]["explanation"]},
            "cands": [{"latex": x["latex"], "delta": x["delta"], "status": x["status"],
                       "valid": is_valid(x["status"]), "eq_said": e, "cls": x["cls"],
                       "augmented": x.get("augmented", False),
                       **({"hom_class": x["hom_class"]} if "hom_class" in x else {})}
                      for x, e in zip(cands, eq)],
            **d, "out_eq_said": symeq(c["said"], d["output"]) == "equal",
            "prev_out": prev_out, "latency": lat, "latency_s": sum(lat.values()),
            "transcript": entry["nbest"][0][0] if entry["nbest"] else ""})
        prev_out = d["output"]
    return records


def _meta(c):
    return {k: c.get(k) for k in ("source", "split", "set", "id", "line", "label", "said", "qc_wer")}


def strip_augmentation(entry):
    """The pre-augmentation view of a stage-1 row: confusable-number variants removed, pre-aug GPU timing.
    Raw scores are per spoken form and System D z-normalises over the pool at selection time, so this is
    exactly the pool the system had before augmentation (checked against the first-pass records)."""
    if entry is None:
        return None
    t = dict(entry["timing"])
    for name in ("whisper_s", "phoneme_s"):
        if name + "_pre_aug" in t:
            t[name] = t[name + "_pre_aug"]
        elif "aug_" + name in t:
            t[name] -= t["aug_" + name]
    return dict(entry, pool=[p for p in entry["pool"] if not p.get("augmented")], timing=t, augmented=False)


def run_records(llm, delta, voices=VOICES, splits=None, workers=16, homophone=False, first_pass=True,
                augmented=True, delta_hom=None):
    """Per-line records at δ (list) for every derivation whose stage-1 clips exist; splits filters.
    homophone / first_pass: which sets to include (see load_clips). augmented=False: strip the
    confusable-number variants from every pool (strip_augmentation), i.e. the pre-augmentation system.
    delta_hom: class-conditional threshold for homophone-class candidates (None: single δ, old rule)."""
    done = read_cache(cache_dir(llm) / "stage1.jsonl")
    params = frozen_d()
    groups = {}
    for c in load_clips(voices, homophone, first_pass):
        if splits is None or c["split"] in splits:
            groups.setdefault((c["source"], c["id"]), []).append(c)
    jobs = []
    for lines in groups.values():
        lines.sort(key=lambda c: c["line"])
        entries = [done.get(clip_key(c)) for c in lines]
        if not augmented:
            entries = [strip_augmentation(e) for e in entries]
        if entries[0] is None or all(e is None for e in entries):
            continue
        jobs.append((lines, entries, delta, params) + (() if delta_hom is None else (delta_hom,)))
    with ProcessPoolExecutor(workers) as ex:
        results = list(tqdm(ex.map(run_derivation, jobs, chunksize=4), total=len(jobs), desc="stage2",
                            mininterval=30))
    return [r for recs in results for r in recs], len(jobs)


def stage2(llm, delta, workers=16, limit=None, voices=VOICES):
    """Write records_<δ>[_voices].jsonl (limit: first N records, debug)."""
    records, n_jobs = run_records(llm, delta, voices, workers=workers)
    tag = "" if tuple(voices) == VOICES else "_" + "-".join(voices)
    out = cache_dir(llm) / f"records_{delta:g}{tag}.jsonl"
    with open(out, "w") as f:
        for r in records[:limit]:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"stage2: {n_jobs} derivations, {len(records[:limit])} line records -> {out}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("stage", choices=["stage1", "stage2", "augment"])
    ap.add_argument("--llm", default=config.LLM_MODEL)
    ap.add_argument("--delta", type=float, default=0.0)
    ap.add_argument("--limit", type=int)
    ap.add_argument("--workers", type=int, default=12)
    ap.add_argument("--voices", nargs="+", default=list(VOICES), help="synthetic voices (recordings always)")
    ap.add_argument("--homophone", action="store_true", help="stage1: also the homophone stress set (hom-*)")
    ap.add_argument("--items-file", type=Path, help="stage1: this homophone-schema file instead (homophone_fresh.json)")
    args = ap.parse_args()
    if args.stage == "stage1":
        stage1(args.llm, args.limit, args.workers, voices=tuple(args.voices),
               homophone=args.items_file or args.homophone)
    elif args.stage == "augment":
        augment_stage(args.llm, args.limit, tuple(args.voices))
    else:
        stage2(args.llm, args.delta, args.workers, args.limit, tuple(args.voices))


if __name__ == "__main__":
    main()
