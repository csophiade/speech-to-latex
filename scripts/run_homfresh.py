"""Confirmatory run on the fresh homophone test set (results/reports/prereg_homophone.md, commit 8dc764e).

Usage (after gen_homophone.py --fresh, synthesis, QC/repair and `run_derivations.py stage1 --items-file`):
  .venv/bin/python -W ignore scripts/run_homfresh.py run            # three sequential passes, records only
  .venv/bin/python -W ignore scripts/run_homfresh.py report [--log] # endpoints, figure, results.json rows

Passes over data/derivations/homophone_fresh.json (v1, v2; augmented pools; System D frozen):
  class   δ_pooled for every candidate, δ_hom for homophone-class candidates (primary)
  pooled  δ_pooled alone (secondary)
  inf     δ = +inf, no gate (secondary)
Thresholds are the prereg's, fixed here before any fresh output exists. No item is excluded.
Primary endpoint: SCR per voice at the class-conditional δ, beta-binomial p against Beta(10, 191).
"""

import argparse
import json
import math
import sys
import time
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from run_derivations import run_records  # noqa: E402
from run_phase4 import (ALPHA, ALPHA_C, C_V1, C_V2, INK, INK2, decision_row, dump, group_metrics,  # noqa: E402
                        read_jsonl, work_dir, write_jsonl)

from src import config  # noqa: E402
from src.calibrate import bound_pvalues, metrics  # noqa: E402
from src.decide import FLAGS  # noqa: E402

FRESH = config.DATA_DIR / "derivations" / "homophone_fresh.json"
DELTA_POOLED = 0.5518526991984581     # pooled calibration, n = 327, k = 16
DELTA_HOM = 0.5395088972175274        # homophone-only calibration, n = 200, k = 10
K_POOLED, N_POOLED, K_HOM, N_HOM = 16, 327, 10, 200
PREREG_COMMIT = "8dc764e"
VOICES = ("v1", "v2")
PASSES = {"class": (DELTA_POOLED, DELTA_HOM), "pooled": (DELTA_POOLED, None), "inf": (math.inf, None)}
FIG = config.FIGURES_DIR / "scr_tradeoff.png"


def out_dir(llm):
    d = work_dir(llm) / "homfresh"
    d.mkdir(parents=True, exist_ok=True)
    return d


def run(llm):
    """The three sequential passes; writes records_<pass>.jsonl and prints only counts."""
    timing = {}
    for name, (delta, delta_hom) in PASSES.items():
        start = time.perf_counter()
        recs, n_jobs = run_records(llm, delta, VOICES, homophone=FRESH, first_pass=False, delta_hom=delta_hom)
        write_jsonl(out_dir(llm) / f"records_{name}.jsonl", recs)
        timing[name] = round(time.perf_counter() - start, 1)
        print(f"pass {name}: {n_jobs} derivations, {len(recs)} line records, {timing[name]}s", flush=True)
    (out_dir(llm) / "run_timing.json").write_text(json.dumps(timing) + "\n")


def voice_recs(recs, v):
    return [r for r in recs if r["source"] == v and not r.get("missing")]


def endpoints(recs):
    """Decision metrics plus the prereg's descriptive extras for one voice."""
    m = group_metrics(recs)
    errs = [r for r in recs if r["label"] == "error"]
    valid = [r for r in recs if r["label"] == "valid"]
    m["error_line_statuses"] = dict(Counter(r["status"] for r in errs))
    m["n_valid_lines"] = len(valid)
    m["valid_line_flags"] = sum(r["status"] in FLAGS for r in valid)
    m["valid_line_flag_statuses"] = dict(Counter(r["status"] for r in valid if r["status"] in FLAGS))
    m["silent_by_hom_class"] = sum(1 for r in errs if r["status"] == "corrected" and not r["out_eq_said"]
                                   and any(c.get("hom_class") and c["latex"] == r["output"] for c in r["cands"]))
    return m


def qc_summary():
    """Manifest QC of the fresh clips and whether Whisper hears the said number on each error line."""
    from check_homophone_audio import MANIFEST, hears_said
    items = {it["id"]: it for it in json.loads(FRESH.read_text())}
    rows = [r for r in map(json.loads, MANIFEST.read_text().splitlines()) if r["id"] in items]
    out = {}
    for v in VOICES:
        mine = [r for r in rows if r["voice"] == v]
        err = [r for r in mine if r["line"] == items[r["id"]]["error"]["line"]]
        heard = [hears_said(r.get("qc_asr", ""), items[r["id"]]) for r in err]
        out[v] = {"n_clips": len(mine), "n_qc_wer_gt0": sum((r.get("qc_wer") or 0) > 0 for r in mine),
                  "n_error_clips": len(err), "n_error_clips_qc_wer_gt0": sum((r.get("qc_wer") or 0) > 0 for r in err),
                  "error_said_heard": sum(ok and not wrong for ok, wrong in heard),
                  "error_not_heard": sorted(r["id"] for r, (ok, wrong) in zip(err, heard) if not (ok and not wrong))}
    return out


def existing_hom(llm):
    """Existing homophone test set (Phase 4b), per voice: no gate and pooled δ."""
    rep = json.loads((work_dir(llm) / "report.json").read_text())["groups"]
    return {v: {n: rep[run][f"hom_{v}"] for n, run in (("inf", "aug_inf"), ("pooled", "cal"))} for v in VOICES}


def report(llm, log=False):
    recs = {n: read_jsonl(out_dir(llm) / f"records_{n}.jsonl") for n in PASSES}
    out = {"prereg_commit": PREREG_COMMIT, "delta_pooled": DELTA_POOLED, "delta_hom": DELTA_HOM,
           "k_hom": K_HOM, "n_hom": N_HOM, "qc": qc_summary(), "groups": {}, "bound_tests": {}}
    for v in VOICES:
        g = out["groups"][v] = {n: endpoints(voice_recs(recs[n], v)) for n in PASSES}
        out["bound_tests"][v] = {
            "class_vs_hom_bound": bound_pvalues(g["class"]["n_silent"], g["class"]["n_error_items"], K_HOM, N_HOM),
            "pooled_vs_pooled_bound": bound_pvalues(g["pooled"]["n_silent"], g["pooled"]["n_error_items"],
                                                    K_POOLED, N_POOLED)}
    out["existing"] = existing_hom(llm)
    plot(out, FIG)
    (out_dir(llm) / "report.json").write_text(dump(out))
    if log:
        log_results(llm, out)
    print(dump({k: out[k] for k in ("qc", "bound_tests")}))
    for v in VOICES:
        for n in PASSES:
            m = out["groups"][v][n]
            print(v, n, f"SCR {m['n_silent']}/{m['n_error_items']}", "err statuses", m["error_line_statuses"],
                  f"valid flags {m['valid_line_flags']}/{m['n_valid_lines']}")


def log_results(llm, out):
    from run_derivations import frozen_d
    from src.eval import append_result
    extra = {"llm": llm, "asr": config.WHISPER_MODEL, "alpha": ALPHA, "phase": 4, "confirmatory": True,
             "prereg": f"results/reports/prereg_homophone.md @ {PREREG_COMMIT}", "augmented": True,
             "generator_seed": 4242, **frozen_d(),
             "error_mix": "researcher-constructed homophone swaps (fresh, prespecified set)"}
    rows = (("D+decide(class)", "class", {"delta": DELTA_POOLED, "delta_hom": DELTA_HOM, "k_hom": K_HOM,
                                          "n_calibration_hom": N_HOM}),
            ("D+decide (pooled δ only)", "pooled", {"delta": DELTA_POOLED, "k": K_POOLED, "n_calibration": N_POOLED}),
            ("D+decide (no gate, δ=+inf)", "inf", {"delta": "inf", "reference_point": True}))
    for v in VOICES:
        for system, name, kw in rows:
            m = json.loads(dump(out["groups"][v][name]))
            if name == "class":
                m["bound_test"] = out["bound_tests"][v]["class_vs_hom_bound"]
            append_result(system, f"deriv_test_homfresh_{v}", m, group=f"fresh homophone test, {v}", **extra, **kw)


# ---------- figure ----------

def _cp(x, n):
    """Clopper-Pearson 95% interval for x/n (descriptive)."""
    from scipy.stats import beta
    lo = beta.ppf(0.025, x, n - x + 1) if x > 0 else 0.0
    hi = beta.ppf(0.975, x + 1, n - x) if x < n else 1.0
    return lo, hi


def plot(out, path):
    """Per voice: no-gate SCR -> gated SCR on the homophone test sets (arrows), α line, the fresh
    class-conditional point as the headline."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.size": 9, "axes.edgecolor": "#b5b4ae", "axes.labelcolor": INK2,
                         "xtick.color": INK2, "ytick.color": INK2})
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.6), sharey=True)
    rows = [("Existing homophone test\npooled δ = 0.552 (calibrated on the mix)", "existing", "pooled"),
            ("Fresh homophone test\npooled δ = 0.552", "fresh", "pooled"),
            ("Fresh homophone test, HEADLINE\nclass-conditional δ_hom = 0.540 (prespecified)", "fresh", "class")]
    for ax, v, col in zip(axes, VOICES, (C_V1, C_V2)):
        ex, fr = out["existing"][v], out["groups"][v]
        for y, (_, src, gated) in enumerate(rows[::-1]):
            g = ex if src == "existing" else fr
            ng, gt = g["inf"], g[gated]
            n = gt["n_error_items"]
            x0, x1 = ng["n_silent"] / ng["n_error_items"], gt["n_silent"] / n
            head = src == "fresh" and gated == "class"
            ax.annotate("", xy=(x1, y), xytext=(x0, y),
                        arrowprops={"arrowstyle": "-|>", "color": col, "lw": 2.2 if head else 1.4,
                                    "alpha": 1 if head else 0.55, "shrinkA": 7, "shrinkB": 9})
            ax.plot(x0, y, "o", ms=9, mfc="white", mec=col, mew=1.8, zorder=5)
            lo, hi = _cp(gt["n_silent"], n)
            ax.plot([lo, hi], [y, y], color=INK, lw=1.2, zorder=4)
            ax.plot(x1, y, "*" if head else "D", ms=16 if head else 8, color=col, mec=INK, mew=0.7, zorder=6)
            ax.text(x0, y + 0.2, f"no gate {ng['n_silent']}/{ng['n_error_items']}", ha="center", fontsize=7.5,
                    color=INK2)
            ax.text(x1, y - 0.3, f"{gt['n_silent']}/{n}", ha="center", va="top", fontsize=8.5 if head else 7.5,
                    color=INK, weight="bold" if head else "normal")
            ax.text((x0 + x1) / 2, y + 0.07, f"−{ng['n_silent'] - gt['n_silent']} silent corrections",
                    ha="center", va="bottom", fontsize=7, color=INK2)
        ax.axvline(ALPHA, color=ALPHA_C, ls="--", lw=1.2)
        ax.text(ALPHA + 0.008, 2.55, "α = 5%", color=ALPHA_C, fontsize=8)
        ax.set_xlim(-0.03, 1.1)
        ax.set_ylim(-0.7, 2.75)
        ax.set_xlabel("silent-correction rate on homophone error lines (lower is better)")
        ax.set_title(f"Voice {v}", fontsize=10, color=INK)
        ax.grid(axis="x", alpha=0.25)
    axes[0].set_yticks(range(3), [r[0] for r in rows[::-1]], fontsize=8)
    axes[0].plot([], [], "o", ms=8, mfc="white", mec="#777", mew=1.6, label="augmented, no gate (δ = +∞)")
    axes[0].plot([], [], "D", ms=7, color="#999", mec=INK, label="pooled δ (n_cal = 327, k = 16)")
    axes[0].plot([], [], "*", ms=13, color="#999", mec=INK,
                 label="class-conditional δ_hom (n_cal = 200 homophone errors, k = 10)")
    axes[0].plot([], [], color=INK, lw=1.2, label="95% Clopper-Pearson interval (descriptive)")
    axes[0].plot([], [], color=ALPHA_C, ls="--", label="α = 5%")
    fig.legend(*axes[0].get_legend_handles_labels(), loc="lower center", ncol=5, fontsize=7.5, frameon=False,
               bbox_to_anchor=(0.5, 0.075))
    p = {v: out["bound_tests"][v]["class_vs_hom_bound"]["beta_binomial"] for v in VOICES}
    fig.suptitle("Homophone stress test: what the δ gate removes (each test set n = 100 error lines per voice; "
                 "v1 and v2 = same derivations re-read: sensitivity, not replicates)", fontsize=10.5, color=INK)
    fig.text(0.5, 0.008,
             "SCR is project-specific: an error line output as a valid step that is not what was said. Bound "
             "P(m < δ) ≤ k/(n+1) holds under exchangeability of calibration and test errors; the errors are "
             "researcher-constructed\n(TTS readings of generated -teen/-ty swaps), not natural dictation. Fresh set "
             f"and thresholds prespecified in results/reports/prereg_homophone.md (commit {PREREG_COMMIT}); "
             f"beta-binomial p vs Beta(10, 191): v1 {p['v1']:.2f}, v2 {p['v2']:.2f}.",
             ha="center", fontsize=7.4, color=INK2)
    fig.tight_layout(rect=(0, 0.13, 1, 0.95))
    fig.savefig(path, dpi=160)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("step", choices=["run", "report"])
    ap.add_argument("--llm", default=config.LLM_MODEL)
    ap.add_argument("--log", action="store_true", help="report: append rows to results/results.json")
    args = ap.parse_args()
    run(args.llm) if args.step == "run" else report(args.llm, args.log)


if __name__ == "__main__":
    main()
