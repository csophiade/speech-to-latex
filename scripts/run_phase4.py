"""Phase 4.2-4.4 (second pass): pooled δ calibration with the homophone stress set, the test-split
decision passes, ablation, margins and figures. The shipped system uses confusable-number pool
augmentation (src/augment.py); Phase 2 params (results/tuned.json) stay frozen.

Usage (each step caches its output under data/cache/derivations/<llm>/phase4/, rerun independently):
  .venv/bin/python scripts/run_phase4.py calibrate   # calib split only -> calibration.json, calib_records.jsonl
  .venv/bin/python scripts/run_phase4.py run         # sequential passes, every split, v1/v2/rec + homophone:
                                                     #   records.jsonl          augmented, calibrated δ (headline)
                                                     #   records_aug_inf.jsonl  augmented, δ = +inf (no gate)
                                                     #   records_noaug_inf.jsonl variants stripped, δ = +inf
                                                     #                           (= the first-pass system)
  .venv/bin/python scripts/run_phase4.py ablation    # A/B/C/D plain selection + checker on test lines
  .venv/bin/python scripts/run_phase4.py report [--log]   # metrics, margins.json, figure (--log: results.json)

Needs stage 1 of scripts/run_derivations.py (augmented rows, plus `stage1 --homophone`).

Calibration set: spoken-error line items (label "error") of the calib split, pooled over
  first-pass derivations: v1, v2 and the owner's recordings (rec), and
  the homophone stress set (data/derivations/homophone.json): v1, v2.
Synthetic clips with qc_wer > 0 (TTS QC: Whisper did not reproduce the verbalised text) are
excluded. Recordings have no label-quality signal (their manifest `wer` is raw Whisper WER of the
owner's speech, i.e. the ASR being evaluated, not a label check), so no recording is excluded.
The accented voice v3 is held out of calibration and is not re-run (its pools are pre-augmentation).
δ = k-th smallest margin, k = ⌊α(n+1)⌋ (+inf margins sort to the top). Margins depend on the previous
line's output, which depends on δ, so the calib pass is repeated at the new δ until δ stops changing
(calib data only). Test outcomes are never read before δ is fixed.
"""

import argparse
import json
import math
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from run_derivations import (cache_dir, clip_key, frozen_d, load_clips, read_cache,  # noqa: E402
                             run_records)

from src import config  # noqa: E402
from src.calibrate import (bound_pvalues, calibrate_delta, cluster_calibration, margin, metrics,  # noqa: E402
                           sweep)
from src.decide import is_valid  # noqa: E402

ALPHA = 0.05
VOICES = ("v1", "v2")               # + recordings (always included); v3 is not re-run
SOURCES = ("v1", "v2", "rec")
FIG = config.FIGURES_DIR
RUNS = {"cal": "records.jsonl", "aug_inf": "records_aug_inf.jsonl", "noaug_inf": "records_noaug_inf.jsonl"}
RUN_LABEL = {"cal": "augmented, calibrated δ (shipped)", "aug_inf": "augmented, δ = +∞ (no gate)",
             "noaug_inf": "not augmented, δ = +∞ (first-pass system)"}

# test-split groups: (sets, sources). v1 and v2 read the same derivations: never pooled together.
GROUPS = {
    "hom_v1": (("homophone",), ("v1",)), "hom_v2": (("homophone",), ("v2",)),
    "pooled_v1": (("first-pass", "homophone"), ("v1",)), "pooled_v2": (("first-pass", "homophone"), ("v2",)),
    "fp_v1": (("first-pass",), ("v1",)), "fp_v2": (("first-pass",), ("v2",)), "rec": (("first-pass",), ("rec",)),
    "fp_all": (("first-pass",), SOURCES),
}
DATASET = {"hom_v1": "deriv_test_hom_v1", "hom_v2": "deriv_test_hom_v2",
           "pooled_v1": "deriv_test_pooled_v1", "pooled_v2": "deriv_test_pooled_v2",
           "fp_v1": "deriv_test_v1", "fp_v2": "deriv_test_v2", "rec": "deriv_test_rec", "fp_all": "deriv_test_all"}
LABEL = {"hom_v1": "homophone test, v1", "hom_v2": "homophone test, v2",
         "pooled_v1": "pooled test (first-pass + homophone), v1", "pooled_v2": "pooled test (first-pass + homophone), v2",
         "fp_v1": "first-pass test, v1", "fp_v2": "first-pass test, v2", "rec": "owner's recordings (test)",
         "fp_all": "first-pass test, v1+v2+rec (v3 excluded; v1/v2 not independent)"}


def work_dir(llm):
    d = cache_dir(llm) / "phase4"
    d.mkdir(parents=True, exist_ok=True)
    return d


def dump(obj):
    """JSON with ±inf as the string "inf" (strict JSON)."""
    def fix(x):
        if isinstance(x, float) and math.isinf(x):
            return "inf" if x > 0 else "-inf"
        if isinstance(x, dict):
            return {k: fix(v) for k, v in x.items()}
        if isinstance(x, list):
            return [fix(v) for v in x]
        return x
    return json.dumps(fix(obj), indent=1, ensure_ascii=False)


def as_float(x):
    return math.inf if x == "inf" else x


def write_jsonl(path, rows):
    with open(path, "w") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def read_jsonl(path):
    return [json.loads(line) for line in open(path)]


def key(r):
    return (r["source"], r["id"], r["line"])


def set_of(r):
    return r.get("set") or ("homophone" if r["id"].startswith("hom-") else "first-pass")


# ---------- 4.2 calibration ----------

def excluded_qc(r):
    """Synthetic clip whose TTS QC failed (qc_wer > 0). Recordings: no label-quality signal, never excluded."""
    return r["source"] != "rec" and (r.get("qc_wer") or 0) > 0


def calib_items(records):
    """(kept, excluded) spoken-error items of the calibration set."""
    errs = [r for r in records if r["label"] == "error" and not r.get("missing") and r["split"] == "calib"]
    bad = [r for r in errs if excluded_qc(r)]
    return [r for r in errs if not excluded_qc(r)], bad


def _nearest_alt_augmented(r):
    alts = [c for c in r["cands"] if c["valid"] and not c["eq_said"]]
    return min(alts, key=lambda c: c["delta"]).get("augmented") if alts else None


def calibrate(llm, max_iter=6):
    """Fixed-point δ on the pooled calib split; writes calibration.json and the calib records."""
    delta, log, seen = 0.0, [], []
    for it in range(max_iter):
        records, _ = run_records(llm, delta, VOICES, splits={"calib"}, homophone=True)
        kept, bad = calib_items(records)
        ms = [margin(r) for r in kept]
        new, k, n = calibrate_delta(ms, ALPHA)
        finite = sorted(((m, r) for m, r in zip(ms, kept) if m < math.inf), key=lambda x: x[0])
        by = {}
        for r, m in zip(kept, ms):
            b = by.setdefault(f"{set_of(r)}/{r['source']}", {"n": 0, "finite": 0, "zero": 0})
            b["n"] += 1
            b["finite"] += m < math.inf
            b["zero"] += m == 0
        log.append({"iteration": it, "records_at_delta": delta, "n_items": len(kept) + len(bad),
                    "excluded_qc": [clip_key(r) for r in bad], "n": n, "k": k, "delta": new,
                    "n_finite_margins": len(finite), "n_inf_margins": n - len(finite),
                    "n_zero_margins": sum(m == 0 for m in ms), "by_set_source": by,
                    "smallest_margins": [{"m": round(m, 4), "item": clip_key(r), "set": set_of(r),
                                          "nearest_alt_augmented": _nearest_alt_augmented(r)}
                                         for m, r in finite[:max(k + 4, 12)]]})
        print(dump({x: y for x, y in log[-1].items() if x != "smallest_margins"}), flush=True)
        if new == delta or new in seen:     # converged (or a cycle: stop at the repeated value)
            delta = new
            break
        seen.append(delta)
        delta = new
    else:
        delta = new
    if log[-1]["records_at_delta"] != delta:   # records must be at the δ written out
        records, _ = run_records(llm, delta, VOICES, splits={"calib"}, homophone=True)
    write_jsonl(work_dir(llm) / "calib_records.jsonl", records)
    out = {"alpha": ALPHA, "delta": delta, "k": log[-1]["k"], "n": log[-1]["n"],
           "excluded_qc": log[-1]["excluded_qc"], "converged": log[-1]["delta"] == log[-1]["records_at_delta"],
           "augmented": True, "log": log}
    (work_dir(llm) / "calibration.json").write_text(dump(out))
    return out


def load_calibration(llm, name="calibration.json"):
    c = json.loads((work_dir(llm) / name).read_text())
    c["delta"] = as_float(c["delta"])
    return c


# ---------- 4.3 decision passes ----------

def run(llm, which=("cal", "aug_inf", "noaug_inf")):
    """Sequential decision passes (every split, v1/v2/rec, first-pass + homophone)."""
    delta = load_calibration(llm)["delta"]
    cfg = {"cal": (delta, True), "aug_inf": (math.inf, True), "noaug_inf": (math.inf, False)}
    for name in which:
        d, aug = cfg[name]
        start = time.perf_counter()
        records, n_jobs = run_records(llm, d, VOICES, homophone=True, augmented=aug)
        write_jsonl(work_dir(llm) / RUNS[name], records)
        print(f"run {name}: δ={d}, augmented={aug}, {n_jobs} derivations, {len(records)} line records, "
              f"{time.perf_counter() - start:.0f}s", flush=True)


def load_run(llm, name):
    recs = read_jsonl(work_dir(llm) / RUNS[name])
    for r in recs:
        r["set"] = set_of(r)
    return recs


def decision_row(r):
    return {"source": r["source"], "label": r["label"], "status": r["status"],
            "hat_status": r["c_hat"]["status"], "hat_eq_said": r["c_hat"]["eq_said"],
            "out_eq_said": r["out_eq_said"]}


def median(xs):
    xs = sorted(xs)
    n = len(xs)
    return round(xs[n // 2] if n % 2 else (xs[n // 2 - 1] + xs[n // 2]) / 2, 3) if n else None


LATENCY_DEF = ("median per-line wall time over test lines >= 1 (line 0 is never checked): Whisper 5-best + "
               "to_latex LLM + to_spoken LLM (slowest call each, original uncached times) + acoustic scoring "
               "on GPU + checker + decide")


def cascade(recs):
    """math_error flags on valid lines by cause: previous output misheard / this line misheard / both / neither."""
    by = {key(r): r for r in recs}
    out = {"prev_output_misheard": 0, "line_misheard": 0, "both": 0, "neither": 0}
    for r in recs:
        if r.get("missing") or r["label"] != "valid" or r.get("status") != "math_error":
            continue
        prev = by.get((r["source"], r["id"], r["line"] - 1))
        p_bad = prev is not None and not prev.get("missing") and not prev["out_eq_said"]
        l_bad = not r["c_hat"]["eq_said"]
        out["both" if p_bad and l_bad else "prev_output_misheard" if p_bad else
            "line_misheard" if l_bad else "neither"] += 1
    return out


def group_metrics(recs):
    """Line-level CER/SymEq vs said, decision metrics (src.calibrate.metrics), caution rate, latency."""
    from src.eval import summarise
    recs = [r for r in recs if not r.get("missing")]
    m, _ = summarise([r["said"] for r in recs], [r.get("output", "") for r in recs],
                     [r.get("latency_s", 0.0) for r in recs])
    m.update(metrics([decision_row(r) for r in recs]))
    checked = [r for r in recs if r["line"] > 0]
    m["n_derivations"] = len({(r["source"], r["id"]) for r in recs})
    m["n_checked_lines"] = len(checked)
    m["caution_rate"] = round(sum(bool(r.get("note")) for r in checked) / len(checked), 4) if checked else None
    m["latency_median_s"] = median([r["latency_s"] for r in checked if "latency_s" in r])
    m["latency_definition"] = LATENCY_DEF
    m["n_qc_imperfect_clips"] = sum(excluded_qc(r) for r in recs)
    m["n_qc_imperfect_error_items"] = sum(excluded_qc(r) for r in recs if r["label"] == "error")
    m["fp_valid_by_cause"] = cascade(recs)
    errs = [r for r in recs if r["label"] == "error"]
    ms = [margin(r) for r in errs]
    m["n_finite_margins"] = sum(x < math.inf for x in ms)
    m["silent_corrections_by_augmented_candidate"] = sum(
        1 for r in errs if r["status"] == "corrected" and not r["out_eq_said"]
        and any(c.get("augmented") and c["latex"] == r["output"] for c in r["cands"]))
    m["fixed_by_augmented_candidate"] = sum(
        1 for r in recs if r["label"] in ("valid", "follows_error") and not r["c_hat"]["eq_said"]
        and r["status"] == "corrected" and r["out_eq_said"]
        and any(c.get("augmented") and c["latex"] == r["output"] for c in r["cands"]))
    return m


def select_group(records, g, split="test"):
    sets, srcs = GROUPS[g]
    return [r for r in records if r["split"] == split and r["set"] in sets and r["source"] in srcs]


# ---------- 4.3 ablation (no decision rule) ----------

def ablation_derivation(args):
    """Plain selection per system over one derivation; checker against the system's own previous line."""
    lines, entries, params = args
    from run_ablation import latency
    from src.checker import check_step
    from src.rerank import select
    out = {}
    for system, p in params.items():
        prev, rows = None, []
        for c, e in zip(lines, entries):
            pool = e["pool"] if e else []
            start = time.perf_counter()
            i = select(system, pool, **p)
            sel_s = time.perf_counter() - start
            pred = pool[i]["latex"] if i is not None else ""
            if c["line"] == 0:
                status = "given"
            elif prev is None or not pred:
                status = "unchecked"
            else:
                status = check_step(prev, pred)["status"]
            rows.append({"source": c["source"], "set": c["set"], "id": c["id"], "line": c["line"],
                         "label": c["label"], "said": c["said"], "pred": pred, "check": status,
                         "status": "math_error" if status == "error" else status,
                         "latency_s": latency(system, e["timing"], sel_s) if e else None})
            prev = pred or None
        out[system] = rows
    return out


def ablation(llm, workers=32):
    """A-D on every test derivation of v1/v2/rec (first-pass + homophone), augmented (shipped) pools."""
    tuned = json.loads((config.RESULTS_DIR / "tuned.json").read_text())
    params = {"A": {}, "B": {}, "C": {"lam": tuned["C"]["lam"]}, "D": frozen_d()}
    done = read_cache(cache_dir(llm) / "stage1.jsonl")
    groups = {}
    for c in load_clips(VOICES, homophone=True):
        if c["split"] == "test":
            groups.setdefault((c["source"], c["id"]), []).append(c)
    jobs = []
    for lines in groups.values():
        lines.sort(key=lambda c: c["line"])
        jobs.append((lines, [done.get(clip_key(c)) for c in lines], params))
    with ProcessPoolExecutor(workers) as ex:
        res = list(ex.map(ablation_derivation, jobs, chunksize=4))
    rows = {s: [r for d in res for r in d[s]] for s in params}
    (work_dir(llm) / "ablation.json").write_text(json.dumps(rows, ensure_ascii=False))
    print({s: len(v) for s, v in rows.items()})


def ablation_metrics(rows):
    from src.calibrate import detection
    from src.eval import summarise
    m, _ = summarise([r["said"] for r in rows], [r["pred"] for r in rows],
                     [r["latency_s"] or 0.0 for r in rows])
    m["selection_latency_median_s"] = m.pop("latency_median_s")   # run_ablation convention: no checker
    m["detection"] = detection(rows)
    m["unchecked_rate"] = round(sum(r["check"] == "unchecked" for r in rows if r["line"] > 0)
                                / sum(r["line"] > 0 for r in rows), 4)
    return m


def ablation_groups(rows):
    for r in rows:
        r.setdefault("set", "first-pass")
    out = {}
    for g in ("fp_all", "fp_v1", "fp_v2", "rec", "hom_v1", "hom_v2"):
        sets, srcs = GROUPS[g]
        sel = [r for r in rows if r["set"] in sets and r["source"] in srcs]
        if sel:
            out[g] = ablation_metrics(sel)
    return out


# ---------- reviewer diagnostics (read-only; nothing here feeds δ) ----------

def cluster_of(r):
    """Derivation-level cluster: v1 and v2 read the same (set, id); recordings are their own derivations."""
    return ("rec" if r["source"] == "rec" else "tts", set_of(r), r["id"], r["line"])


def calib_pairing(calib_recs, alpha=ALPHA):
    """v1/v2 dependence inside the calibration set: margin correlation and a cluster-level δ (sensitivity only)."""
    from scipy.stats import spearmanr
    kept, _ = calib_items(calib_recs)
    ms = [margin(r) for r in kept]
    pairs = {}
    for r, m in zip(kept, ms):
        if r["source"] in VOICES:
            pairs.setdefault(cluster_of(r), {})[r["source"]] = m
    both = [(p["v1"], p["v2"]) for p in pairs.values() if len(p) == 2]
    fin = [(a, b) for a, b in both if a < math.inf and b < math.inf]
    rho, pval = spearmanr([a for a, _ in fin], [b for _, b in fin]) if len(fin) > 2 else (None, None)
    cl = cluster_calibration(ms, [cluster_of(r) for r in kept], alpha)
    return {"n_items": len(kept), "n_clusters": cl["n_clusters"], "n_v1_v2_pairs": len(both),
            "n_pairs_both_finite": len(fin), "spearman_rho": None if rho is None else float(rho),
            "spearman_p": None if pval is None else float(pval),
            "cluster_min": dict(zip(("delta", "k", "n"), cl["min"])),
            "cluster_max": dict(zip(("delta", "k", "n"), cl["max"]))}


def bound_tests(groups, cal):
    """Test SCR against k/(n+1): binomial (anti-conservative) and beta-binomial (reported) p-values."""
    out = {}
    for g in ("hom_v1", "hom_v2", "pooled_v1", "pooled_v2"):
        m = groups[g]
        out[g] = bound_pvalues(m["n_silent"], m["n_error_items"], cal["k"], cal["n"])
    return out


def _best(entry, kind):
    return max(entry[kind]) if entry.get(kind) else -math.inf


def margin_structure(llm, recs):
    """Homophone spoken-error items: how much of each margin is pool-relative z-scaling.

    Two-candidate pools z-normalise to +1/-1 whatever the raw gap, so their margin is eps + gamma*2 = 1.0.
    raw_* gaps are the best-form raw phoneme / whisper scores of what was heard (c_hat) minus the nearest
    valid alternative (positive = the scorer prefers what was heard)."""
    s1 = read_cache(cache_dir(llm) / "stage1.jsonl")
    items = []
    for r in recs:
        if r.get("missing") or r["label"] != "error" or set_of(r) != "homophone":
            continue
        m = margin(r)
        row = {"split": r["split"], "source": r["source"], "pool_size": len(r["cands"]), "m": m,
               "raw_phoneme_gap": None, "raw_whisper_gap": None}
        alts = [c for c in r["cands"] if c["valid"] and not c["eq_said"]]
        pool = {e["latex"]: e for e in s1.get(clip_key(r), {}).get("pool", [])}
        if alts and m < math.inf:
            alt = min(alts, key=lambda c: c["delta"])
            heard, other = pool.get(r["c_hat"]["latex"]), pool.get(alt["latex"])
            if heard and other:
                row["raw_phoneme_gap"] = _best(heard, "phoneme") - _best(other, "phoneme")
                row["raw_whisper_gap"] = _best(heard, "whisper") - _best(other, "whisper")
        items.append(row)

    def summ(rows):
        fin = [x for x in rows if x["m"] < math.inf]
        near1 = [x for x in fin if abs(x["m"] - 1.0) < 0.01]
        gaps = sorted(x["raw_phoneme_gap"] for x in fin if x["raw_phoneme_gap"] is not None)
        q = (lambda p: gaps[min(len(gaps) - 1, int(p * len(gaps)))]) if gaps else (lambda p: None)
        return {"n": len(rows), "n_finite": len(fin), "n_two_candidate_pools": sum(x["pool_size"] == 2 for x in rows),
                "n_margin_1_0": len(near1), "n_margin_1_0_in_two_candidate_pools":
                    sum(x["pool_size"] == 2 for x in near1),
                "n_two_candidate_finite_not_1_0": sum(x["pool_size"] == 2 and abs(x["m"] - 1) >= 0.01 for x in fin),
                "n_raw_gap": len(gaps), "raw_phoneme_gap_q10_q50_q90": [q(0.1), q(0.5), q(0.9)],
                "share_raw_phoneme_prefers_heard": round(sum(g > 0 for g in gaps) / len(gaps), 4) if gaps else None}
    return {sp: summ([x for x in items if x["split"] == sp]) for sp in ("calib", "test")}


def annotate_results(out):
    """Add the bound-test p-values to the existing second-pass rows in place (metrics unchanged, idempotent)."""
    rows = json.loads(config.RESULTS_JSON.read_text())
    ds = {DATASET[g]: g for g in out["bound_tests"]}
    for r in rows:
        if r.get("second_pass") and r["system"] == "D+decide" and r["dataset"] in ds:
            r["bound_test"] = out["bound_tests"][ds[r["dataset"]]]
            r["bound_wording"] = BOUND_WORDING[ds[r["dataset"]].split("_")[0]]
    config.RESULTS_JSON.write_text(json.dumps(rows, indent=2, ensure_ascii=False) + "\n")


BOUND_WORDING = {"pooled": "calibrated bound under exchangeability; not rejected on the pooled test "
                           "(beta-binomial), both point estimates above k/(n+1)",
                 "hom": "calibrated bound under exchangeability does not cover this subset; exceeded on the "
                        "homophone test subset (beta-binomial p < 0.05)"}


# ---------- margins.json ----------

def compact(r):
    """The fields calibrate.simulate needs (+ which candidates are augmentation variants)."""
    return {"line": r["line"], "label": r["label"], "source": r["source"],
            "c_hat": {"status": r["c_hat"]["status"], "eq_said": r["c_hat"]["eq_said"]},
            "cands": [{k: c.get(k, False) for k in ("delta", "valid", "eq_said", "cls", "augmented")}
                      for c in r["cands"]]}


def margins_json(cal, calib_recs, records, noaug):
    kept, _ = calib_items(calib_recs)
    kept_keys = {key(r) for r in kept}
    by_key = {key(r): r for r in calib_recs if not r.get("missing")}
    noaug_by = {key(r): r for r in noaug if not r.get("missing")}
    errs, mis = [], []
    for r in records:
        if r.get("missing"):
            continue
        k = key(r)
        base = {"source": r["source"], "set": r["set"], "split": r["split"], "id": r["id"], "line": r["line"],
                "test": r["split"] == "test"}
        na = noaug_by.get(k)
        if r["label"] == "error":
            src = by_key.get(k, r)
            errs.append({**base, "m_i": margin(src), "m_i_no_augmentation": margin(na) if na else None,
                         "nearest_alt_augmented": _nearest_alt_augmented(src),
                         "c_hat_valid_and_different": is_valid(src["c_hat"]["status"]) and not src["c_hat"]["eq_said"],
                         "excluded_qc": excluded_qc(r), "in_calibration": k in kept_keys,
                         "status_at_calibrated_delta": r["status"], "record": compact(src)})
        elif r["label"] in ("valid", "follows_error") and not r["c_hat"]["eq_said"]:
            mis.append({**base, "label": r["label"], "hat_status": r["c_hat"]["status"],
                        "detectable": r["c_hat"]["status"] == "error",
                        "fixed_at_calibrated_delta": r["out_eq_said"], "record": compact(r)})
    schema = {
        "alpha, k, n, delta": "calibration: δ = k-th smallest m_i over the n pooled calibration items, "
                              "k = floor(alpha*(n+1)); \"inf\" = +infinity",
        "augmentation": "shipped system: every pool carries confusable-number variants (src/augment.py) for "
                        "v1, v2, rec and the homophone set; v3 (accented) is not in this file (pre-augmentation, "
                        "not re-run)",
        "calibration_log": "one entry per fixed-point iteration of the pooled calib-split pass",
        "spoken_error_items[]": "every line item labelled 'error' (calib and test; sources v1|v2|rec; set "
                                "first-pass|homophone): split, id, line, test (bool), m_i (augmented pools: min Delta "
                                "over valid candidates not SymEq-equal to what was said; 0 if c_hat is valid and "
                                "different; \"inf\" if none), m_i_no_augmentation (same with the variants stripped, "
                                "δ = +inf run), nearest_alt_augmented (the valid alternative attaining m_i is an "
                                "augmentation variant), c_hat_valid_and_different, excluded_qc (synthetic qc_wer>0), "
                                "in_calibration, status_at_calibrated_delta, record",
        "mishearing_items[]": "every valid/follows_error line item whose c_hat differs from what was said: source, "
                              "set, split, id, line, test, label, hat_status, detectable (c_hat checker status "
                              "error), fixed_at_calibrated_delta, record",
        "record": "{line, label, source, c_hat{status, eq_said}, cands[{delta, valid, eq_said, cls, augmented}]}: "
                  "feed to src.calibrate.simulate(record, delta) to recompute the decision at any δ (statuses were "
                  "checked against the previous output of the calibrated run, so away from the calibrated δ this "
                  "is an approximation). Calib error records come from the final calib-split pass; everything "
                  "else from the calibrated run.",
        "SCR at δ": "share of test spoken-error items with simulate(...).status in {verified, corrected} and "
                    "not out_eq_said; bound event: m_i < δ",
        "fix rate at δ": "share of test mishearing items with simulate(...).out_eq_said",
    }
    return {"schema": schema, "alpha": cal["alpha"], "k": cal["k"], "n": cal["n"], "delta": cal["delta"],
            "excluded_qc": cal["excluded_qc"], "augmented": True, "calibration_log": cal["log"],
            "spoken_error_items": errs, "mishearing_items": mis}


# ---------- figure ----------

C_V1, C_V2, C_NOAUG, INK, INK2, ALPHA_C = "#2a78d6", "#eb6834", "#8a8984", "#0b0b0b", "#52514e", "#e34948"


def sweep_deltas(records):
    ds = sorted({round(c["delta"], 4) for r in records for c in r.get("cands", []) if c["valid"]})
    return [0.0] + [d + 1e-6 for d in ds] + [math.inf]


def _sweep(recs):
    recs = [r for r in recs if not r.get("missing")]
    ds = sweep_deltas(recs)
    return [dict(s, delta=d) for s, d in zip(sweep(recs, ds), ds)]


def plot_tradeoff(runs, cal, path, calib_recs=None):
    """Headline. Left: fix rate vs SCR on the pooled test set (per voice) as δ sweeps, augmented vs not,
    with the three real operating points. Right: homophone-test SCR against δ, with the calibration
    margins' empirical CDF (pooled, and homophone-only) that δ was read from."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.size": 9, "axes.edgecolor": "#b5b4ae", "axes.labelcolor": INK2,
                         "xtick.color": INK2, "ytick.color": INK2})
    fig, (ax, bx) = plt.subplots(1, 2, figsize=(13.5, 7.4), gridspec_kw={"width_ratios": [1.15, 1]})
    out = {}
    delta = cal["delta"]
    act = lambda name, g: metrics([decision_row(r) for r in select_group(runs[name], g) if not r.get("missing")])  # noqa: E731
    for v, col in (("v1", C_V1), ("v2", C_V2)):
        g = f"pooled_{v}"
        sw = _sweep(select_group(runs["cal"], g))
        na = _sweep(select_group(runs["noaug_inf"], g))
        pts = {n: act(n, g) for n in RUNS}
        out[g], out[g + "_noaug"], out[g + "_actual"] = sw, na, pts
        out[f"hom_{v}"] = _sweep(select_group(runs["cal"], f"hom_{v}"))
        ax.step([s["scr"] for s in sw], [s["fix_rate"] for s in sw], where="post", color=col, lw=2,
                label=f"{v}, augmented (shipped pools)")
        ax.step([s["scr"] for s in na], [s["fix_rate"] for s in na], where="post", color=col, lw=1.4,
                ls=":", alpha=0.9, label=f"{v}, not augmented (first-pass pools)")
        ax.plot(pts["noaug_inf"]["scr"], pts["noaug_inf"]["fix_rate"], "o", ms=9, color=col, mfc="white",
                mew=1.8, zorder=6)
        ax.plot(pts["aug_inf"]["scr"], pts["aug_inf"]["fix_rate"], "X", ms=11, color=col, mec=INK, mew=0.7,
                zorder=6)
        ax.plot(pts["cal"]["scr"], pts["cal"]["fix_rate"], "*", ms=17, color=col, mec=INK, mew=0.7, zorder=7)
        for name, dx, dy in (("aug_inf", -0.01, 0.012), ("cal", 0.012, 0.004 if v == "v1" else -0.012)):
            p = pts[name]
            ax.annotate(f"{v}: SCR {p['n_silent']}/{p['n_error_items']}, fix {p['fix_rate']:.3f}",
                        (p["scr"], p["fix_rate"]), xytext=(p["scr"] + dx, p["fix_rate"] + dy), fontsize=7,
                        color=INK2, ha="right" if name == "aug_inf" else "left")
    ax.axvline(ALPHA, color=ALPHA_C, ls="--", lw=1.2, label="α = 5%")
    hs = {n: "/".join(str(act(n, f"hom_{v}")["n_silent"]) for v in ("v1", "v2")) for n in ("aug_inf", "cal")}
    ax.plot([], [], "o", ms=8, color="#999", mfc="white", mew=1.6,
            label="not augmented, δ = +∞: cannot correct sound-alikes (SCR 0; no measured fix-rate cost here)")
    ax.plot([], [], "X", ms=10, color="#999", mec=INK,
            label=f"augmented, δ = +∞ (no gate): corrects sound-alikes indiscriminately "
                  f"(homophone v1/v2 {hs['aug_inf']} of 100 silent)")
    ax.plot([], [], "*", ms=14, color="#999", mec=INK,
            label=f"augmented + calibrated δ = {fmt_delta(delta)} (shipped): cuts homophone silent "
                  f"corrections to v1/v2 {hs['cal']} of 100, no fix-rate gain")
    ax.set_xlabel("silent-correction rate, SCR, pooled test errors (project-specific; lower is better)")
    ax.set_ylabel("fix rate of mishearings, pooled test (higher is better)")
    ax.set_xlim(-0.02, 0.72)
    ax.set_ylim(0, 0.2)
    ax.grid(alpha=0.25)
    ax.legend(fontsize=7, loc="upper center", bbox_to_anchor=(0.5, -0.13), ncol=1, frameon=False)
    n_err = out["pooled_v1"][0]["n_error_items"]
    n_mis = ", ".join(f"{v} {out[f'pooled_{v}'][0]['n_mishearings']}" for v in ("v1", "v2"))
    ax.set_title(f"Pooled test, per voice: fix rate vs SCR as δ sweeps\n(errors n={n_err} per voice, "
                 f"mishearings n: {n_mis})", fontsize=10, color=INK)

    hom = [m for m in (margin(r) for r in calib_items(calib_recs)[0] if r["set"] == "homophone")] if calib_recs else []
    pooled = [margin(r) for r in calib_items(calib_recs)[0]] if calib_recs else []
    fin = [m for m in pooled if m < math.inf]
    inf_x = max(fin + [delta if delta < math.inf else 0, 1.6]) * 1.08
    for v, col in (("v1", C_V1), ("v2", C_V2)):
        sw = out[f"hom_{v}"]
        bx.step([min(s["delta"], inf_x) for s in sw], [s["scr"] for s in sw], where="post", color=col, lw=2,
                label=f"SCR, homophone test {v} (n={sw[0]['n_error_items']})")
    for ms_, lab, ls in ((pooled, f"calibration margins, pooled (n={len(pooled)})", "-"),
                         (hom, f"calibration margins, homophone only (n={len(hom)})", "--")):
        if not ms_:
            continue
        xs = sorted(min(m, inf_x) for m in ms_)
        bx.step([0] + xs, [0] + [(i + 1) / len(xs) for i in range(len(xs))], where="post", color=C_NOAUG,
                lw=1.4, ls=ls, label=f"share below δ: {lab}")
    bx.axhline(ALPHA, color=ALPHA_C, ls="--", lw=1.2, label="α = 5%")
    if delta < math.inf:
        bx.axvline(delta, color=INK, lw=1.0, alpha=0.6)
        h1 = out["groups_cal_hom"] = {v: act("cal", f"hom_{v}") for v in ("v1", "v2")}
        bx.text(1.06, 0.12, f"calibrated δ = {delta:.3f}\n(n={cal['n']}, k={cal['k']}, "
                f"k/(n+1) = {cal['k'] / (cal['n'] + 1):.3f})\nhomophone test SCR:\n"
                f"v1 {h1['v1']['n_silent']}/100, v2 {h1['v2']['n_silent']}/100", fontsize=7.5, color=INK, va="bottom")
    ticks = [t for t in (0, 0.25, 0.5, 0.75, 1.0, 1.25, 1.5) if t < inf_x * 0.93]
    bx.set_xticks(ticks + [inf_x], [f"{t:g}" for t in ticks] + ["∞"])
    bx.set_xlim(-0.02, inf_x * 1.03)
    bx.set_ylim(-0.02, 1.02)
    bx.set_xlabel("δ (score margin below the best-sounding candidate)")
    bx.set_ylabel("rate")
    bx.grid(alpha=0.25)
    bx.legend(fontsize=7, loc="upper center", bbox_to_anchor=(0.5, -0.13), ncol=1, frameon=False)
    bx.set_title("Homophone stress test: SCR against δ\n(said number's sound-alike is the right answer)",
                 fontsize=10, color=INK)
    fig.text(0.5, 0.008,
             "SCR is project-specific. Calibrated bound under exchangeability: P(m < δ) ≤ k/(n+1) for an error drawn "
             "like the pooled calibration mix; it is not a per-subset guarantee. Error mix (injected algebra slips + "
             "homophone swaps)\nis researcher-constructed. v1 and v2 read the same derivations (not independent). "
             "Curves re-score stored pools (approximate away from the calibrated δ); markers are real sequential runs.\n"
             "On this test data augmentation buys no measured fix-rate benefit: the not-augmented point weakly dominates "
             "the shipped point (same or higher fix rate, SCR 0). The benefit is hypothetical (real -teen/-ty mishearings; "
             "none in the test set).",
             ha="center", fontsize=7.4, color=INK2)
    fig.tight_layout(rect=(0, 0.075, 1, 1))
    fig.savefig(path, dpi=160)
    return out


def fmt_delta(d):
    return "+∞" if d == math.inf else f"{d:.3f}"


# ---------- report ----------

def report(llm, log=False):
    from src.eval import append_result
    cal = load_calibration(llm)
    calib_recs = read_jsonl(work_dir(llm) / "calib_records.jsonl")
    for r in calib_recs:
        r["set"] = set_of(r)
    runs = {n: load_run(llm, n) for n in RUNS}
    out = {"delta": cal["delta"], "k": cal["k"], "n": cal["n"], "latency_definition": LATENCY_DEF}
    out["groups"] = {n: {g: group_metrics(select_group(runs[n], g)) for g in GROUPS} for n in RUNS}
    # latency parts (shipped run, test lines >= 1, v1/v2/rec, both sets)
    parts = [r["latency"] for r in runs["cal"] if r["split"] == "test" and r["line"] > 0 and "latency" in r]
    out["latency_parts_median_s"] = {k: median([p[k] for p in parts]) for k in parts[0]} if parts else {}
    out["latency_parts_median_s"]["total"] = median([r["latency_s"] for r in runs["cal"]
                                                     if r["split"] == "test" and r["line"] > 0 and "latency_s" in r])
    abl = json.loads((work_dir(llm) / "ablation.json").read_text())
    out["ablation"] = {s: ablation_groups(rows) for s, rows in abl.items()}
    out["ablation"]["D+decide"] = {g: out["groups"]["cal"][g] for g in ("fp_all", "fp_v1", "fp_v2", "rec",
                                                                       "hom_v1", "hom_v2")}
    pre = json.loads((work_dir(llm) / "ablation_pre_aug.json").read_text())
    out["ablation_pre_aug"] = {s: ablation_groups([r for r in rows if r["source"] != "v3"]) for s, rows in pre.items()}
    out["bound_tests"] = bound_tests(out["groups"]["cal"], cal)
    out["calib_pairing"] = calib_pairing(calib_recs)
    out["margin_structure"] = margin_structure(llm, calib_recs + [r for r in runs["cal"] if r["split"] == "test"])
    out["sweep"] = plot_tradeoff(runs, cal, FIG / "scr_tradeoff_pooled.png", calib_recs)
    (config.RESULTS_DIR / "margins.json").write_text(
        dump(margins_json(cal, calib_recs, runs["cal"], runs["noaug_inf"])) + "\n")
    (work_dir(llm) / "report.json").write_text(dump(out))
    if log:
        log_results(llm, cal, out, append_result)
    annotate_results(out)
    print(dump({k: v for k, v in out.items() if k != "sweep"})[:20000])


def log_results(llm, cal, out, append_result):
    """Mark the first-pass phase-4 rows superseded (kept), then append the second-pass rows."""
    rows = json.loads(config.RESULTS_JSON.read_text())
    for r in rows:
        if r.get("phase") == 4 and not r.get("second_pass"):
            if r["dataset"] == "deriv_test_v3":
                r["pre_augmentation"] = True
                r["note"] = ("accented voice v3: first-pass numbers, pre-augmentation, δ = +inf; not re-run "
                             "(still the current v3 result)")
            else:
                r["superseded"] = True
                r["superseded_by"] = "phase 4 second pass (confusable-number augmentation + homophone calibration)"
    config.RESULTS_JSON.write_text(json.dumps(rows, indent=2, ensure_ascii=False) + "\n")
    tuned = json.loads((config.RESULTS_DIR / "tuned.json").read_text())
    params = {"A": {}, "B": {}, "C": {"lam": tuned["C"]["lam"]}, "D": frozen_d()}
    extra = {"llm": llm, "asr": config.WHISPER_MODEL, "seed": config.SEED, "alpha": ALPHA,
             "delta": "inf" if cal["delta"] == math.inf else cal["delta"], "k": cal["k"],
             "n_calibration": cal["n"], "phase": 4, "second_pass": True, "augmented": True, **frozen_d(),
             "error_mix": "researcher-constructed (injected algebra slips + homophone swaps)"}
    for g, m in out["groups"]["cal"].items():
        append_result("D+decide", DATASET[g], json.loads(dump(m)), group=LABEL[g], **extra)
    for g in ("hom_v1", "hom_v2", "pooled_v1", "pooled_v2", "fp_all"):
        append_result("D+decide (no gate, δ=+inf)", DATASET[g], json.loads(dump(out["groups"]["aug_inf"][g])),
                      group=LABEL[g], **dict(extra, delta="inf", reference_point=True))
    for s in ("A", "B", "C", "D"):
        append_result(s, "deriv_test", out["ablation"][s]["fp_all"], llm=llm, asr=config.WHISPER_MODEL,
                      seed=config.SEED, ablation=True, phase=4, second_pass=True, augmented=True,
                      sources="v1+v2+rec (first-pass test; v3 excluded)", checker="own previous output", **params[s])
    append_result("D+decide", "deriv_test", json.loads(dump(out["groups"]["cal"]["fp_all"])), ablation=True,
                  sources="v1+v2+rec (first-pass test; v3 excluded)", **extra)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("step", choices=["calibrate", "run", "ablation", "report"])
    ap.add_argument("--llm", default=config.LLM_MODEL)
    ap.add_argument("--only", nargs="+", choices=list(RUNS), help="run: only these passes")
    ap.add_argument("--log", action="store_true", help="report: append rows to results/results.json")
    args = ap.parse_args()
    if args.step == "calibrate":
        calibrate(args.llm)
    elif args.step == "run":
        run(args.llm, tuple(args.only or RUNS))
    elif args.step == "ablation":
        ablation(args.llm)
    else:
        report(args.llm, args.log)


if __name__ == "__main__":
    main()
