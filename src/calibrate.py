"""δ calibration (Phase 4.2) and model-free re-evaluation of the decision rule at any δ.

Margin of a spoken-error item: m_i = min delta over valid candidates not SymEq-equal to what was
said (0 if ĉ itself is valid and differs, +inf if there is none). Rule 2 corrects silently iff
m_i < δ (ask_back is ignored, so the bound is conservative). With δ = m_(k), k = ⌊α(n+1)⌋, split
conformal gives P(m_test < δ) ≤ k/(n+1) ≤ α for an exchangeable test item. The bound covers the
event {m < δ} only; items with ĉ valid-and-different (m = 0) are silent at every δ > 0 and are
reported separately as "misrecognised into valid maths".

Line record (one clip of one line):
  {source, split, id, line, label, said,
   c_hat: {latex, status, eq_said}, cands: [{delta, valid, eq_said, cls}]}
label ∈ given|valid|error|follows_error; eq_said = symeq(said, latex) == 'equal'; cls = SymEq class
id within the line. Candidate statuses were computed against prev_out from the calibrated run, so
the sweep is an approximation away from that δ (prev_out itself depends on δ); headline numbers
at the calibrated δ come from the real sequential run, fed to metrics() as decision rows.
"""

import math

from src.decide import FLAGS, is_valid

CHECKED = ("valid", "error", "follows_error")


def calibrate_delta(margins, alpha=0.05):
    """(δ, k, n): δ = k-th smallest margin, k = ⌊α(n+1)⌋. k = 0 (n < 1/α − 1) gives δ = 0: never correct."""
    n = len(margins)
    k = math.floor(alpha * (n + 1) + 1e-9)
    if k == 0:
        return 0.0, 0, n
    return sorted(margins)[k - 1], k, n        # may be inf if fewer than k finite margins


def margin(rec):
    """m_i for a spoken-error line record."""
    hat = rec["c_hat"]
    if is_valid(hat["status"]) and not hat["eq_said"]:
        return 0.0
    ds = [c["delta"] for c in rec["cands"] if c["valid"] and not c["eq_said"]]
    return min(ds, default=math.inf)


def simulate(rec, delta):
    """Decision row for a record at δ (mirrors decide.decide)."""
    hat = rec["c_hat"]
    row = {"source": rec["source"], "label": rec["label"], "hat_status": hat["status"],
           "hat_eq_said": hat["eq_said"]}
    if rec["line"] == 0:
        return dict(row, status="given", out_eq_said=hat["eq_said"])
    if is_valid(hat["status"]):
        return dict(row, status="verified", out_eq_said=hat["eq_said"])
    close = sorted((c for c in rec["cands"] if c["valid"] and c["delta"] < delta),
                   key=lambda c: c["delta"])
    n_cls = len({c["cls"] for c in close})
    if n_cls == 1:
        return dict(row, status="corrected", out_eq_said=close[0]["eq_said"])
    if n_cls >= 2:
        return dict(row, status="ask_back", out_eq_said=hat["eq_said"])
    return dict(row, status="math_error" if hat["status"] == "error" else "unchecked",
                out_eq_said=hat["eq_said"])


def _rate(num, den):
    return round(num / den, 4) if den else None


def detection(rows, flags=("math_error",)):
    """Error-detection P/R/F1: lenient (follows_error ignored) and strict (follows_error flags = FP)."""
    flagged = [r["status"] in flags for r in rows]
    tp = sum(f for f, r in zip(flagged, rows) if r["label"] == "error")
    fn = sum(not f for f, r in zip(flagged, rows) if r["label"] == "error")
    fp = sum(f for f, r in zip(flagged, rows) if r["label"] == "valid")
    fp_follow = sum(f for f, r in zip(flagged, rows) if r["label"] == "follows_error")
    out = {"tp": tp, "fn": fn, "fp": fp, "fp_follows_error": fp_follow}
    for name, false_pos in (("", fp), ("strict_", fp + fp_follow)):
        p, r = _rate(tp, tp + false_pos), _rate(tp, tp + fn)
        out[name + "precision"], out[name + "recall"] = p, r
        out[name + "f1"] = round(2 * p * r / (p + r), 4) if p and r else 0.0
    return out


def metrics(rows):
    """SCR, fix rate, ask-back rate, status counts and detection from decision rows."""
    errs = [r for r in rows if r["label"] == "error"]
    silent = [r for r in errs if r["status"] in ("verified", "corrected") and not r["out_eq_said"]]
    uncontrollable = [r for r in errs if is_valid(r["hat_status"]) and not r["hat_eq_said"]]
    mis = [r for r in rows if r["label"] in ("valid", "follows_error") and not r["hat_eq_said"]]
    det = [r for r in mis if r["hat_status"] == "error"]
    checked = [r for r in rows if r["label"] in CHECKED]
    statuses = {}
    for r in rows:
        statuses[r["status"]] = statuses.get(r["status"], 0) + 1
    return {
        "n_lines": len(rows), "n_error_items": len(errs), "n_silent": len(silent),
        "scr": _rate(len(silent), len(errs)),
        "valid_misrecognised_rate": _rate(len(uncontrollable), len(errs)),
        "n_mishearings": len(mis), "fix_rate": _rate(sum(r["out_eq_said"] for r in mis), len(mis)),
        "n_detectable": len(det), "fix_rate_detectable": _rate(sum(r["out_eq_said"] for r in det), len(det)),
        "ask_back_rate": _rate(sum(r["status"] == "ask_back" for r in checked), len(checked)),
        "flag_rate": _rate(sum(r["status"] in FLAGS for r in checked), len(checked)),
        "statuses": statuses,
        "detection": detection(rows),
        "detection_with_ask_back": detection(rows, flags=tuple(FLAGS)),
    }


def evaluate(records, delta):
    return dict(metrics([simulate(r, delta) for r in records]), delta=delta)


def sweep(records, deltas):
    """evaluate() at each δ; also the bound event rate P(m < δ) over spoken-error items."""
    ms = [margin(r) for r in records if r["label"] == "error"]
    return [dict(evaluate(records, d), bound_event_rate=_rate(sum(m < d for m in ms), len(ms)))
            for d in deltas]
