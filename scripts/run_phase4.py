"""Phase 4.2-4.4: δ calibration, the test-split decision pass, ablation, margins and figures.

Usage (each step caches its output, so they can be rerun independently):
  .venv/bin/python scripts/run_phase4.py calibrate   # calib split only -> data/cache/.../phase4/calibration.json
  .venv/bin/python scripts/run_phase4.py run         # decision pass at the calibrated δ, all sources
  .venv/bin/python scripts/run_phase4.py ablation    # A/B/C/D plain selection + checker on test lines
  .venv/bin/python scripts/run_phase4.py report [--log]   # metrics, margins.json, figures (--log: results.json)

Needs stage 1 of scripts/run_derivations.py for every source (v1, v2, v3, rec).

Calibration set: spoken-error line items (label "error") of the calib split, from v1, v2 and the
owner's recordings (the accented v3 voice is held out of calibration), minus clips with
qc_wer > 0. δ = k-th smallest margin, k = ⌊α(n+1)⌋. Margins depend on the previous line's output,
which depends on δ, so the calib pass is repeated at the new δ until δ stops changing (calib data
only). Test outcomes are never read before δ is fixed.
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

from run_derivations import (DERIVS, cache_dir, frozen_d, load_clips, read_cache,  # noqa: E402
                             run_records, clip_key)

from src import config  # noqa: E402
from src.calibrate import calibrate_delta, margin, metrics, sweep  # noqa: E402
from src.decide import is_valid  # noqa: E402

ALPHA = 0.05
CALIB_VOICES = ("v1", "v2")          # + recordings (always included)
ALL_VOICES = ("v1", "v2", "v3")
GROUPS = {"overall": ("v1", "v2", "v3", "rec"), "synth": ("v1", "v2"), "v3": ("v3",), "rec": ("rec",)}
DATASET = {"overall": "deriv_test_all", "synth": "deriv_test_synth", "v3": "deriv_test_v3",
           "rec": "deriv_test_rec"}
LABEL = {"overall": "overall test (v1+v2+v3+rec)", "synth": "synthetic voices v1+v2",
         "v3": "accented voice v3", "rec": "owner's recordings"}
FIG = config.FIGURES_DIR


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


# ---------- 4.2 calibration ----------

def calib_items(records):
    """(kept, excluded) spoken-error items of the calibration set."""
    errs = [r for r in records if r["label"] == "error" and not r.get("missing")]
    bad = [r for r in errs if (r.get("qc_wer") or 0) > 0]
    return [r for r in errs if r not in bad], bad


def calibrate(llm, max_iter=5):
    """Fixed-point δ on the calib split; writes calibration.json and the calib records."""
    delta, log = 0.0, []
    for it in range(max_iter):
        records, _ = run_records(llm, delta, CALIB_VOICES, splits={"calib"})
        kept, bad = calib_items(records)
        ms = [margin(r) for r in kept]
        new, k, n = calibrate_delta(ms, ALPHA)
        finite = sorted(m for m in ms if m < math.inf)
        log.append({"iteration": it, "records_at_delta": delta, "n_items": len(kept) + len(bad),
                    "excluded_qc": len(bad), "n": n, "k": k, "delta": new,
                    "n_finite_margins": len(finite), "n_zero_margins": sum(m == 0 for m in ms),
                    "smallest_margins": finite[:10],
                    "by_source": {s: sum(r["source"] == s for r in kept) for s in ("v1", "v2", "rec")}})
        print(dump(log[-1]), flush=True)
        if new == delta:
            break
        delta = new
    write_jsonl(work_dir(llm) / "calib_records.jsonl", records)
    out = {"alpha": ALPHA, "delta": delta, "k": log[-1]["k"], "n": log[-1]["n"],
           "excluded_qc": log[-1]["excluded_qc"], "converged": log[-1]["delta"] == log[-1]["records_at_delta"],
           "log": log}
    (work_dir(llm) / "calibration.json").write_text(dump(out))
    return out


def load_calibration(llm):
    c = json.loads((work_dir(llm) / "calibration.json").read_text())
    c["delta"] = as_float(c["delta"])
    return c


# ---------- 4.3 decision pass ----------

def run(llm):
    """Sequential decision pass at the calibrated δ for every source and split."""
    delta = load_calibration(llm)["delta"]
    records, n_jobs = run_records(llm, delta, ALL_VOICES)
    write_jsonl(work_dir(llm) / "records.jsonl", records)
    print(f"run: δ={delta}, {n_jobs} derivations, {len(records)} line records")


def decision_row(r):
    return {"source": r["source"], "label": r["label"], "status": r["status"],
            "hat_status": r["c_hat"]["status"], "hat_eq_said": r["c_hat"]["eq_said"],
            "out_eq_said": r["out_eq_said"]}


def median(xs):
    xs = sorted(xs)
    n = len(xs)
    return round(xs[n // 2] if n % 2 else (xs[n // 2 - 1] + xs[n // 2]) / 2, 3) if n else None


def group_metrics(recs):
    """Line-level CER/SymEq vs said, decision metrics, caution rate and latency for one group."""
    from src.eval import summarise
    recs = [r for r in recs if not r.get("missing")]
    lat = [r.get("latency_s", 0.0) for r in recs]
    m, _ = summarise([r["said"] for r in recs], [r.get("output", "") for r in recs], lat)
    m.update(metrics([decision_row(r) for r in recs]))
    checked = [r for r in recs if r["line"] > 0]
    m["n_derivations"] = len({(r["source"], r["id"]) for r in recs})
    m["n_checked_lines"] = len(checked)
    m["caution_rate"] = round(sum(bool(r.get("note")) for r in checked) / len(checked), 4)
    m["latency_median_s_lines_ge1"] = median([r["latency_s"] for r in checked if "latency_s" in r])
    m["n_qc_imperfect_clips"] = sum((r.get("qc_wer") or 0) > 0 for r in recs)
    m["n_qc_imperfect_error_items"] = sum((r.get("qc_wer") or 0) > 0 for r in recs if r["label"] == "error")
    return m


def test_groups(records):
    test = [r for r in records if r["split"] == "test"]
    return {g: [r for r in test if r["source"] in srcs] for g, srcs in GROUPS.items()}


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
            rows.append({"source": c["source"], "id": c["id"], "line": c["line"], "label": c["label"],
                         "said": c["said"], "pred": pred, "check": status,
                         "status": "math_error" if status == "error" else status,
                         "latency_s": latency(system, e["timing"], sel_s) if e else None})
            prev = pred or None
        out[system] = rows
    return out


def ablation(llm, workers=16):
    tuned = json.loads((config.RESULTS_DIR / "tuned.json").read_text())
    params = {"A": {}, "B": {}, "C": {"lam": tuned["C"]["lam"]}, "D": frozen_d()}
    done = read_cache(cache_dir(llm) / "stage1.jsonl")
    groups = {}
    for c in load_clips(ALL_VOICES):
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
    m["detection"] = detection(rows)
    m["unchecked_rate"] = round(sum(r["check"] == "unchecked" for r in rows if r["line"] > 0)
                                / sum(r["line"] > 0 for r in rows), 4)
    return m


# ---------- 4.2 margins.json ----------

def compact(r):
    """The fields calibrate.simulate needs."""
    return {"line": r["line"], "label": r["label"], "source": r["source"],
            "c_hat": {"status": r["c_hat"]["status"], "eq_said": r["c_hat"]["eq_said"]},
            "cands": [{k: c[k] for k in ("delta", "valid", "eq_said", "cls")} for c in r["cands"]]}


def margins_json(cal, calib_recs, records):
    kept, _ = calib_items(calib_recs)
    kept_keys = {(r["source"], r["id"], r["line"]) for r in kept}
    by_key = {(r["source"], r["id"], r["line"]): r for r in calib_recs if not r.get("missing")}
    errs, mis = [], []
    for r in records:
        if r.get("missing"):
            continue
        key = (r["source"], r["id"], r["line"])
        base = {"source": r["source"], "split": r["split"], "id": r["id"], "line": r["line"],
                "test": r["split"] == "test"}
        if r["label"] == "error":
            src = by_key.get(key, r)
            errs.append({**base, "m_i": margin(src),
                         "c_hat_valid_and_different": is_valid(src["c_hat"]["status"]) and not src["c_hat"]["eq_said"],
                         "excluded_qc": (r.get("qc_wer") or 0) > 0, "in_calibration": key in kept_keys,
                         "record": compact(src)})
        elif r["label"] in ("valid", "follows_error") and not r["c_hat"]["eq_said"]:
            mis.append({**base, "label": r["label"], "hat_status": r["c_hat"]["status"],
                        "detectable": r["c_hat"]["status"] == "error",
                        "fixed_at_calibrated_delta": r["out_eq_said"], "record": compact(r)})
    schema = {
        "alpha, k, n, delta": "calibration: δ = k-th smallest m_i over the n calibration items, k = floor(alpha*(n+1)); "
                              "\"inf\" = +infinity",
        "calibration_log": "one entry per fixed-point iteration of the calib-split pass (see scripts/run_phase4.py)",
        "spoken_error_items[]": "every line item labelled 'error' (calib and test, all sources): source v1|v2|v3|rec, "
                                "split, id, line, test (bool), m_i (min Delta over valid candidates not SymEq-equal "
                                "to what was said; 0 if c_hat is valid and different; \"inf\" if none), "
                                "c_hat_valid_and_different, excluded_qc (qc_wer>0), in_calibration, record",
        "mishearing_items[]": "every valid/follows_error line item whose c_hat differs from what was said: source, "
                              "split, id, line, test, label, hat_status, detectable (c_hat checker status error), "
                              "fixed_at_calibrated_delta, record",
        "record": "{line, label, source, c_hat{status, eq_said}, cands[{delta, valid, eq_said, cls}]}: feed to "
                  "src.calibrate.simulate(record, delta) to recompute the decision at any δ (statuses were checked "
                  "against the previous output of the calibrated run). Calib error records come from the final "
                  "calib-split pass; everything else from the calibrated all-source run.",
        "SCR at δ": "share of test spoken-error items with simulate(...).status in {verified, corrected} and "
                    "not out_eq_said; bound event: m_i < δ",
        "fix rate at δ": "share of test mishearing items with simulate(...).out_eq_said",
    }
    return {"schema": schema, "alpha": cal["alpha"], "k": cal["k"], "n": cal["n"], "delta": cal["delta"],
            "excluded_qc": cal["excluded_qc"], "calibration_log": cal["log"],
            "spoken_error_items": errs, "mishearing_items": mis}


# ---------- figures ----------

COLORS = {"synth": "#2a78d6", "rec": "#eb6834", "v3": "#1baf7a", "overall": "#52514e"}


def sweep_deltas(records):
    ds = sorted({round(c["delta"], 4) for r in records for c in r.get("cands", []) if c["valid"]})
    return [0.0] + [d + 1e-6 for d in ds] + [math.inf]


def plot_tradeoff(groups, cal, path):
    """Left: fix rate vs SCR as δ sweeps (calibrated point marked). Right: both rates against δ."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, (ax, bx) = plt.subplots(1, 2, figsize=(12, 5.2))
    out, top = {}, 0.0
    for g in ("synth", "v3", "rec"):
        recs = [r for r in groups[g] if not r.get("missing")]
        if not recs:
            continue
        ds = sweep_deltas(recs)
        sw = sweep(recs, ds)
        out[g] = [dict(s, delta=d) for s, d in zip(sw, ds)]
        top = max(top, max(d for d in ds if d < math.inf))
        xs, ys = [s["scr"] for s in sw], [s["fix_rate"] for s in sw]
        n_err, n_mis = sw[0]["n_error_items"], sw[0]["n_mishearings"]
        lab = f"{LABEL[g]} (error items n={n_err}, mishearings n={n_mis})"
        ax.step(xs, ys, where="post", color=COLORS[g], lw=2, label=lab)
        ax.plot(xs[0], ys[0], "o", ms=6, color=COLORS[g], mfc="white", mew=1.5)
        at = sweep(recs, [cal["delta"]])[0]
        ax.plot(at["scr"], at["fix_rate"], "*", ms=15, color=COLORS[g], mec="#0b0b0b", mew=0.8)
        out[g + "_calibrated"] = at
    inf_x = top * 1.15
    for g in ("synth", "v3", "rec"):
        if g not in out:
            continue
        xs = [min(s["delta"], inf_x) for s in out[g]]
        bx.step(xs, [s["fix_rate"] for s in out[g]], where="post", color=COLORS[g], lw=2,
                label=f"fix rate, {LABEL[g]}")
        bx.step(xs, [s["scr"] for s in out[g]], where="post", color=COLORS[g], lw=1.5, ls="--",
                label=f"SCR, {LABEL[g]}")
    ax.axvline(ALPHA, color="#d03b3b", ls=":", lw=1.4, label="α = 5% target")
    bx.axhline(ALPHA, color="#d03b3b", ls=":", lw=1.4, label="α = 5%")
    ax.plot([], [], "*", ms=12, color="#888", mec="#0b0b0b", label=f"calibrated δ = {fmt_delta(cal['delta'])}")
    ax.plot([], [], "o", ms=6, color="#888", mfc="white", label="δ = 0")
    ax.set_xlabel("silent-correction rate, SCR (project-specific; lower is better)")
    ax.set_ylabel("fix rate of mishearings (higher is better)")
    ax.set_xlim(-0.005, 0.1)
    ax.set_ylim(-0.02, 1.02)
    ax.grid(alpha=0.25)
    ax.legend(fontsize=7.5, loc="center right")
    ax.set_title("Fix rate vs SCR as δ sweeps (test split)", fontsize=10)
    ticks = [t for t in bx.get_xticks() if 0 <= t <= top]
    bx.set_xticks(ticks + [inf_x], [f"{t:g}" for t in ticks] + ["∞\n(calibrated)"])
    bx.set_xlim(-0.02, inf_x * 1.03)
    bx.set_ylim(-0.02, 1.02)
    bx.set_xlabel("δ (score margin below the best-sounding candidate)")
    bx.set_ylabel("rate")
    bx.grid(alpha=0.25)
    bx.legend(fontsize=7, loc="upper left")
    bx.set_title(f"Both rates against δ (δ calibrated on n={cal['n']} calib errors, k={cal['k']})", fontsize=10)
    fig.text(0.5, 0.01, "SCR is a project-specific metric (lower is better). The conformal bound assumes calibration "
             "and test errors are exchangeable; calibration is mostly synthetic speech (v1/v2: 120 of 127),\n"
             "so curves are shown per source. No error item, calib or test, had a valid alternative in its pool "
             "(every margin is +∞), so SCR is 0 at every δ and δ = +∞. The sweep re-scores stored pools.",
             ha="center", fontsize=7.5)
    fig.tight_layout(rect=(0, 0.07, 1, 1))
    fig.savefig(path, dpi=150)
    return out


def fmt_delta(d):
    return "+∞" if d == math.inf else f"{d:.3f}"


def plot_accent(per_voice, path):
    """CER, SymEq, SCR, fix rate, P, R by voice on the same 50 test derivations."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    keys = [("cer", "CER ↓"), ("symeq_on_parseable", "SymEq ↑"), ("scr", "SCR ↓"), ("fix_rate", "fix rate ↑"),
            ("precision", "error P ↑"), ("recall", "error R ↑")]
    voices = list(per_voice)
    colors = ["#2a78d6", "#eb6834", "#1baf7a"]
    fig, ax = plt.subplots(figsize=(9, 4.4))
    w = 0.8 / len(voices)
    for k, v in enumerate(voices):
        m = per_voice[v]
        vals = [m["detection"][key] if key in ("precision", "recall") else m[key] for key, _ in keys]
        xs = [i + (k - (len(voices) - 1) / 2) * w for i in range(len(keys))]
        ax.bar(xs, [x or 0 for x in vals], w * 0.92, color=colors[k],
               label=f"{v} (lines n={m['n']}, error items n={m['n_error_items']}, mishearings n={m['n_mishearings']})")
        for x, y in zip(xs, vals):
            ax.text(x, (y or 0) + 0.01, "–" if y is None else f"{y:.2f}", ha="center", va="bottom", fontsize=7)
    ax.set_xticks(range(len(keys)), [lab for _, lab in keys])
    ax.set_ylim(0, 1.12)
    ax.legend(fontsize=7.5, loc="upper center", ncol=1)
    ax.set_title("Native (v1, v2) vs accented (v3) synthetic voice, same 50 test derivations, D+decide", fontsize=10)
    fig.text(0.5, 0.01, "Error P/R: math_error flags, follows_error lines ignored. Small n for v3 error items; "
             "no significance claim.", ha="center", fontsize=7.5)
    fig.tight_layout(rect=(0, 0.05, 1, 1))
    fig.savefig(path, dpi=150)


# ---------- report ----------

def report(llm, log=False):
    from src.eval import append_result
    cal = load_calibration(llm)
    calib_recs = read_jsonl(work_dir(llm) / "calib_records.jsonl")
    records = read_jsonl(work_dir(llm) / "records.jsonl")
    groups = test_groups(records)
    out = {"delta": cal["delta"], "groups": {g: group_metrics(rs) for g, rs in groups.items() if rs}}
    abl = json.loads((work_dir(llm) / "ablation.json").read_text())
    out["ablation"] = {s: ablation_metrics(rows) for s, rows in abl.items()}
    out["ablation"]["D+decide"] = out["groups"]["overall"]
    ids = set(json.loads((config.DATA_DIR / "derivations" / "accent_subset.json").read_text())["ids"])
    test = [r for r in records if r["split"] == "test" and r["id"] in ids]
    per_voice = {v: group_metrics([r for r in test if r["source"] == v]) for v in ALL_VOICES}
    out["native_vs_accented"] = per_voice
    if any(r["source"] == "v3" for r in records):
        plot_accent(per_voice, FIG / "native_vs_accented.png")
    out["sweep"] = plot_tradeoff(groups, cal, FIG / "scr_tradeoff.png")
    (config.RESULTS_DIR / "margins.json").write_text(dump(margins_json(cal, calib_recs, records)) + "\n")
    (work_dir(llm) / "report.json").write_text(dump(out))
    if log:
        extra = {"llm": llm, "asr": config.WHISPER_MODEL, "seed": config.SEED, "alpha": ALPHA,
                 "delta": "inf" if cal["delta"] == math.inf else cal["delta"], "k": cal["k"],
                 "n_calibration": cal["n"], "phase": 4, **frozen_d()}
        for g, m in out["groups"].items():
            append_result("D+decide", DATASET[g], json.loads(dump(m)), **extra)
        tuned = json.loads((config.RESULTS_DIR / "tuned.json").read_text())
        params = {"A": {}, "B": {}, "C": {"lam": tuned["C"]["lam"]}, "D": frozen_d()}
        for s, m in out["ablation"].items():
            if s != "D+decide":
                append_result(s, "deriv_test", m, llm=llm, asr=config.WHISPER_MODEL, seed=config.SEED,
                              ablation=True, phase=4, sources="v1+v2+v3+rec", checker="own previous output",
                              **params[s])
        append_result("D+decide", "deriv_test", json.loads(dump(out["groups"]["overall"])), ablation=True,
                      sources="v1+v2+v3+rec", **extra)
    print(dump({k: v for k, v in out.items() if k != "sweep"}))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("step", choices=["calibrate", "run", "ablation", "report"])
    ap.add_argument("--llm", default=config.LLM_MODEL)
    ap.add_argument("--log", action="store_true", help="report: append rows to results/results.json")
    args = ap.parse_args()
    if args.step == "calibrate":
        calibrate(args.llm)
    elif args.step == "run":
        run(args.llm)
    elif args.step == "ablation":
        ablation(args.llm)
    else:
        report(args.llm, args.log)


if __name__ == "__main__":
    main()
