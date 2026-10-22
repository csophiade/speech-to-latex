"""Task 3.3: checker-only eval on the 200 test derivations (text, no audio).

Usage: .venv/bin/python -W ignore scripts/eval_checker.py [--split test|calib] [--plain-products] [--no-log]
Step-level, scored against derivgen's step_label (line 0 "given" is not a step). The flag is status
"error" only; "caution" is a note shown to the user, never a flag (its row is printed for reference).
TP = flag on the "error" line, FN = no flag there, FP = flag on a "valid" line. Lines labelled
"follows_error" continue correctly from the injected mistake (e.g. a final answer that keeps the
extraneous root of a squaring step, which is wrong for the original problem): ADJUSTED scoring counts
flags there as neither TP nor FP; STRICT scoring counts them as FP (the pre-step_label convention).
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
    """Strict and adjusted P/R/F1 for flagging the injected step, plus per-type recall."""
    tp = fn = fp_valid = fp_follow = n_valid = n_follow = 0
    per_type = Counter()
    for it, res in zip(items, runs):
        for r in res[1:]:
            hit = r["status"] in flag
            label = it["step_label"][r["line"]]
            if label == "error":
                tp += hit
                fn += not hit
                per_type[it["error"]["type"]] += hit
            elif label == "follows_error":
                n_follow += 1
                fp_follow += hit
            else:
                n_valid += 1
                fp_valid += hit
    out = {"tp": tp, "fn": fn, "fp_valid": fp_valid, "flags_follows_error": fp_follow,
           "n_valid_steps": n_valid, "n_follows_error_steps": n_follow,
           "fp_rate_valid": fp_valid / n_valid if n_valid else 0.0}
    for name, fp in (("adjusted", fp_valid), ("strict", fp_valid + fp_follow)):
        p, r, f = prf(tp, fp, fn)
        out[name] = {"precision": p, "recall": r, "f1": f, "fp": fp}
    return out, per_type


def caution_report(items, runs):
    """Every caution step; True if all are multiplications by a nonzero denominator."""
    rows = [(it, r) for it, res in zip(items, runs) for r in res[1:] if r["status"] == "caution"]
    def is_denominator_mult(it, r):
        prev = it["lines"][r["line"] - 1]
        return r["explanation"].startswith("multiplying by ") and "≠ 0" in r["explanation"] \
            and r"\frac{" in prev
    return rows, bool(rows) and all(is_denominator_mult(it, r) for it, r in rows)


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

    err, hits = score(items, runs, {"error"})
    loose, hits_loose = score(items, runs, {"error", "caution"})
    cautions, all_denom = caution_report(items, runs)
    n_err = Counter(it["error"]["type"] for it in items if it["error"])
    steps = Counter(r["status"] for res in runs for r in res[1:])
    labels = Counter(it["step_label"][r["line"]] for it, res in zip(items, runs) for r in res[1:])
    n_steps = sum(steps.values())
    metrics = {"flag": "error", "headline": "adjusted",
               **{k: err["adjusted"][k] for k in ("precision", "recall", "f1", "fp")},
               **err, "caution_row": {k: loose[k] for k in ("adjusted", "strict", "fp_rate_valid",
                                                             "flags_follows_error")},
               "recall_by_type": {t: hits[t] / n for t, n in sorted(n_err.items())},
               "caution_rate": steps["caution"] / n_steps, "unchecked_rate": steps["unchecked"] / n_steps,
               "n_caution": steps["caution"], "cautions_all_denominator_mult": all_denom,
               "step_labels": dict(labels),
               "sec_per_derivation": sum(secs) / len(secs), "n": len(items), "n_steps": n_steps,
               "n_error_items": sum(n_err.values())}

    print(f"{args.split}: {len(items)} derivations, {n_steps} steps {dict(labels)}, "
          f"{metrics['n_error_items']} injected errors")
    print(f"{'flag':<26}{'P':>7}{'R':>7}{'F1':>7}{'TP':>5}{'FP':>5}{'FN':>5}{'FPR':>8}")
    for name, m, kind in (("error (adjusted)", err, "adjusted"), ("error (strict)", err, "strict"),
                          ("[caution row] err+caut adj", loose, "adjusted"),
                          ("[caution row] err+caut str", loose, "strict")):
        x = m[kind]
        print(f"{name:<26}{x['precision']:7.3f}{x['recall']:7.3f}{x['f1']:7.3f}{m['tp']:5d}{x['fp']:5d}"
              f"{m['fn']:5d}{m['fp_rate_valid']:8.3f}")
    print(f"error flags on follows_error lines: {err['flags_follows_error']} / {err['n_follows_error_steps']}")
    for t in sorted(n_err):
        print(f"  recall {t:<20}{metrics['recall_by_type'][t]:.3f}  (n={n_err[t]})")
    print(f"caution {metrics['caution_rate']:.3f} ({steps['caution']} steps)  unchecked "
          f"{metrics['unchecked_rate']:.3f}  {metrics['sec_per_derivation']:.3f} s/derivation")
    for it, r in cautions:
        print(f"CAUTION {it['id']} line {r['line']} [{it['step_label'][r['line']]}]: "
              f"{it['lines'][r['line'] - 1]!r} -> {it['lines'][r['line']]!r} {r['explanation']}")
    if all_denom:
        print("all cautions were multiplications by a nonzero denominator")

    # every miss and false alarm, for inspection
    for it, res in zip(items, runs):
        for r in res[1:]:
            label = it["step_label"][r["line"]]
            miss = label == "error" and r["status"] != "error"
            alarm = label != "error" and r["status"] in ("error", "unchecked")
            if miss or alarm:
                kind = f"FN {it['error']['type']}" if miss else f"{r['status'].upper()} on {label}"
                print(f"{kind} {it['id']} line {r['line']}: {it['lines'][r['line'] - 1]!r} -> "
                      f"{it['lines'][r['line']]!r} [{r['status']}] {r['explanation']}")

    if args.no_log:
        return
    dataset = f"derivations_{args.split}" + ("_plain_products" if args.plain_products else "")
    supersede("checker", dataset)
    append_result("checker", dataset, metrics, n=len(items), scoring="step_label")
    if not args.plain_products:
        write_samples(items, runs, metrics)


def supersede(system, dataset):
    """Mark earlier results.json rows of (system, dataset) superseded: true."""
    path = RESULTS_DIR / "results.json"
    if not path.exists():
        return
    rows = json.loads(path.read_text())
    for r in rows:
        if r["system"] == system and r["dataset"] == dataset:
            r["superseded"] = True
    path.write_text(json.dumps(rows, indent=2, ensure_ascii=False) + "\n")


def summary_table(m):
    """Markdown header table: strict vs adjusted error-only P/R/F1, plus the caution row."""
    c = m["caution_row"]
    rows = [("error, adjusted (headline)", m["adjusted"], m["tp"], m["fn"]),
            ("error, strict", m["strict"], m["tp"], m["fn"]),
            ("error + caution, adjusted (reference only)", c["adjusted"], m["tp"], m["fn"]),
            ("error + caution, strict (reference only)", c["strict"], m["tp"], m["fn"])]
    out = [f"Derivation test split: {m['n']} derivations, {m['n_steps']} steps "
           f"({m['step_labels'].get('valid', 0)} valid, {m['step_labels'].get('error', 0)} error, "
           f"{m['step_labels'].get('follows_error', 0)} follows_error).", "",
           "| flag | precision | recall | F1 | TP | FP | FN |", "|---|---|---|---|---|---|---|"]
    out += [f"| {name} | {x['precision']:.3f} | {x['recall']:.3f} | {x['f1']:.3f} | {tp} | {x['fp']} | {fn} |"
            for name, x, tp, fn in rows]
    note = (f"Only status \"error\" is a flag; \"caution\" is a note to the user, never a flag. Strict counts "
            f"flags on lines that correctly follow from an earlier injected error as false positives "
            f"({m['flags_follows_error']} such flags, e.g. a final answer that keeps the extraneous root of an injected squaring step); "
            f"adjusted counts them as neither TP nor FP.")
    if m["cautions_all_denominator_mult"]:
        note += f" All {m['n_caution']} cautions were multiplications by a nonzero denominator."
    return out + ["", note, ""]


def write_samples(items, runs, metrics):
    """Header table plus one caught example per kind -> results/reports/checker_samples.md."""
    out = ["# Checker sample explanations", "", *summary_table(metrics),
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
