"""Task 3.3: checker-only eval on the 200 test derivations (text, no audio).

Usage: .venv/bin/python -W ignore scripts/eval_checker.py [--split test|calib] [--plain-products] [--no-log]
Step-level: a step is flagged if its status is "error" (or "error"/"caution" for the loose row).
TP = the injected wrong line flagged; FP = any other step flagged. Lines after the wrong line
continue correctly from it, so they count as valid steps. Line 0 (given) is not a step.
fp_downstream counts the FPs that sit after the injected line of an error item: e.g. a final answer
that keeps the extraneous root of an injected squaring step, which the checker rightly calls wrong
for the original problem; precision_excl_downstream leaves them out.
--plain-products rewrites derivgen's "x \\cdot (x - 3)" to "x(x - 3)", the way dictation writes it.
"""

import argparse
import json
import sys
import time
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.checker import check_derivation  # noqa: E402
from src.config import DATA_DIR, RESULTS_DIR  # noqa: E402
from src.eval import append_result  # noqa: E402

DERIV = DATA_DIR / "derivations" / "derivations.json"
SAMPLE_KINDS = ["divide_by_variable", "square_both_sides", "missed_flip"]


def prf(tp, fp, fn):
    p = tp / (tp + fp) if tp + fp else 0.0
    r = tp / (tp + fn) if tp + fn else 0.0
    return p, r, (2 * p * r / (p + r) if p + r else 0.0)


def score(items, runs, flag):
    """P/R/F1 of flagging the erroneous step, per-type recall and FP rate on valid steps."""
    tp = fp = fn = n_valid = down = 0
    per_type = Counter()
    for it, res in zip(items, runs):
        bad = it["error"]["line"] if it["error"] else None
        for r in res[1:]:
            hit = r["status"] in flag
            if r["line"] == bad:
                tp += hit
                fn += not hit
                per_type[it["error"]["type"]] += hit
            else:
                n_valid += 1
                fp += hit
                down += hit and bad is not None and r["line"] > bad
    p, r, f = prf(tp, fp, fn)
    return {"precision": p, "recall": r, "f1": f, "tp": tp, "fp": fp, "fn": fn,
            "fp_rate_valid": fp / n_valid if n_valid else 0.0, "fp_downstream": down,
            "precision_excl_downstream": prf(tp, fp - down, fn)[0]}, per_type


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="test")
    ap.add_argument("--plain-products", action="store_true")
    ap.add_argument("--no-log", action="store_true")
    args = ap.parse_args()
    items = [it for it in json.loads(DERIV.read_text()) if it["split"] == args.split]
    if args.plain_products:
        items = [{**it, "lines": [s.replace(r"x \cdot (", "x(") for s in it["lines"]]} for it in items]

    runs, secs = [], []
    for it in items:
        t0 = time.perf_counter()
        runs.append(check_derivation(it["lines"]))
        secs.append(time.perf_counter() - t0)

    strict, hits = score(items, runs, {"error"})
    loose, hits_loose = score(items, runs, {"error", "caution"})
    n_err = Counter(it["error"]["type"] for it in items if it["error"])
    steps = Counter(r["status"] for res in runs for r in res[1:])
    n_steps = sum(steps.values())
    metrics = {**strict,
               "loose": {k: loose[k] for k in ("precision", "recall", "f1", "fp_rate_valid", "fp_downstream",
                                               "precision_excl_downstream")},
               "recall_by_type": {t: hits[t] / n for t, n in sorted(n_err.items())},
               "recall_by_type_loose": {t: hits_loose[t] / n for t, n in sorted(n_err.items())},
               "caution_rate": steps["caution"] / n_steps, "unchecked_rate": steps["unchecked"] / n_steps,
               "sec_per_derivation": sum(secs) / len(secs), "n": len(items), "n_steps": n_steps,
               "n_error_items": sum(n_err.values())}

    print(f"{args.split}: {len(items)} derivations, {n_steps} steps, {metrics['n_error_items']} injected errors")
    print(f"{'flag':<16}{'P':>7}{'R':>7}{'F1':>7}{'FPR':>8}{'FPdown':>8}{'P-down':>8}")
    for name, m in (("error", strict), ("error+caution", loose)):
        print(f"{name:<16}{m['precision']:7.3f}{m['recall']:7.3f}{m['f1']:7.3f}{m['fp_rate_valid']:8.3f}"
              f"{m['fp_downstream']:8d}{m['precision_excl_downstream']:8.3f}")
    for t in sorted(n_err):
        print(f"  recall {t:<20}{metrics['recall_by_type'][t]:.3f}  (n={n_err[t]})")
    print(f"caution {metrics['caution_rate']:.3f}  unchecked {metrics['unchecked_rate']:.3f}  "
          f"{metrics['sec_per_derivation']:.3f} s/derivation")

    # every miss and false alarm, for inspection
    for it, res in zip(items, runs):
        bad = it["error"]["line"] if it["error"] else None
        for r in res[1:]:
            miss = r["line"] == bad and r["status"] != "error"
            alarm = r["line"] != bad and r["status"] in ("error", "unchecked")
            if miss or alarm:
                kind = f"FN {it['error']['type']}" if miss else r["status"].upper()
                print(f"{kind} {it['id']} line {r['line']}: {it['lines'][r['line'] - 1]!r} -> "
                      f"{it['lines'][r['line']]!r} [{r['status']}] {r['explanation']}")

    if args.no_log:
        return
    dataset = f"derivations_{args.split}" + ("_plain_products" if args.plain_products else "")
    append_result("checker", dataset, metrics, n=len(items))
    if not args.plain_products:
        write_samples(items, runs)


def write_samples(items, runs):
    """One caught example per kind -> results/reports/checker_samples.md."""
    out = ["# Checker sample explanations", "",
           "One caught injected error per kind from the derivation test split (text input).", ""]
    for kind in SAMPLE_KINDS:
        for it, res in zip(items, runs):
            if not it["error"] or it["error"]["type"] != kind:
                continue
            k = it["error"]["line"]
            if res[k]["status"] != "error":
                continue
            out += [f"## {kind} ({it['id']})", "", "```latex", *it["lines"], "```", "",
                    f"- line {k}: `{it['lines'][k - 1]}` -> `{it['lines'][k]}`",
                    f"- status: **{res[k]['status']}**", f"- explanation: \"{res[k]['explanation']}\"", ""]
            break
    path = RESULTS_DIR / "reports" / "checker_samples.md"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(out))
    print(f"samples -> {path}")


if __name__ == "__main__":
    main()
