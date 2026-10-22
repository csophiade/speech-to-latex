import math
import random

from src.calibrate import calibrate_delta, evaluate, margin, metrics, simulate, sweep


def test_quantile_recovered():
    ms = list(range(1, 128))
    random.Random(0).shuffle(ms)
    assert calibrate_delta(ms, 0.05) == (6, 6, 127)


def test_exact_multiple_no_float_slip():
    assert calibrate_delta(list(range(1, 20)), 0.05) == (1, 1, 19)     # 0.05 * 20 = 1
    assert calibrate_delta(list(range(1, 40)), 0.05) == (2, 2, 39)


def test_ties_and_inf():
    ms = [0.0] * 3 + [0.5] * 5 + [math.inf] * 119
    assert calibrate_delta(ms, 0.05) == (0.5, 6, 127)
    assert calibrate_delta([math.inf] * 127 + [1.0], 0.05)[0] == math.inf


def test_k_zero_gives_zero():
    assert calibrate_delta([0.3] * 10, 0.05) == (0.0, 0, 10)


def test_empirical_coverage_on_exchangeable_margins():
    rng = random.Random(1)
    hits = 0
    for _ in range(400):
        ms = [rng.random() for _ in range(128)]
        delta, _, _ = calibrate_delta(ms[:127], 0.05)
        hits += ms[127] < delta
    assert hits / 400 <= 0.08


def rec(label, line, hat_status, hat_eq, cands, source="v1"):
    return {"source": source, "split": "test", "id": "d", "line": line, "label": label, "said": "",
            "c_hat": {"latex": "", "status": hat_status, "eq_said": hat_eq}, "cands": cands}


def cd(delta, valid, eq, cls):
    return {"delta": delta, "valid": valid, "eq_said": eq, "cls": cls}


ERR_HEARD_RIGHT = rec("error", 2, "error", True, [cd(0.0, False, True, 0), cd(0.3, True, False, 1)])
ERR_TWO_ALTS = rec("error", 2, "error", True, [cd(0.0, False, True, 0), cd(0.2, True, False, 1),
                                               cd(0.6, True, False, 2)])
ERR_VALID_WRONG = rec("error", 3, "valid", False, [cd(0.0, True, False, 0)])
MISHEARD = rec("valid", 1, "error", False, [cd(0.0, False, False, 0), cd(0.4, True, True, 1)])
FOLLOW = rec("follows_error", 3, "error", True, [cd(0.0, False, True, 0)])
CLEAN = rec("valid", 1, "valid", True, [cd(0.0, True, True, 0)])
GIVEN = rec("given", 0, "unchecked", True, [cd(0.0, False, True, 0)])
RECORDS = [ERR_HEARD_RIGHT, ERR_TWO_ALTS, ERR_VALID_WRONG, MISHEARD, FOLLOW, CLEAN, GIVEN]


def test_margins():
    assert margin(ERR_HEARD_RIGHT) == 0.3
    assert margin(ERR_VALID_WRONG) == 0.0
    assert margin(rec("error", 1, "error", True, [cd(0.0, False, True, 0)])) == math.inf


def test_simulate_branches():
    assert simulate(ERR_HEARD_RIGHT, 0.1)["status"] == "math_error"
    assert simulate(ERR_HEARD_RIGHT, 0.5)["status"] == "corrected"
    assert simulate(ERR_TWO_ALTS, 1.0)["status"] == "ask_back"
    assert simulate(GIVEN, 1.0)["status"] == "given"
    assert simulate(rec("valid", 1, "unchecked", True, [cd(0.0, False, True, 0)]), 1.0)["status"] == "unchecked"


def test_metrics_at_delta():
    m = evaluate(RECORDS, 0.5)
    assert m["n_error_items"] == 3 and m["n_silent"] == 3          # heard-right corrected, two-alts corrected, valid-wrong
    assert m["valid_misrecognised_rate"] == round(1 / 3, 4)
    assert m["fix_rate"] == 1.0 and m["n_mishearings"] == 1 and m["fix_rate_detectable"] == 1.0
    m0 = evaluate(RECORDS, 0.0)
    assert m0["n_silent"] == 1 and m0["fix_rate"] == 0.0
    d = m0["detection"]
    assert (d["tp"], d["fn"], d["fp"], d["fp_follows_error"]) == (2, 1, 1, 1)
    assert d["precision"] == round(2 / 3, 4) and d["strict_precision"] == 0.5


def test_ask_back_not_silent_and_counted():
    m = evaluate(RECORDS, 1.0)
    assert m["statuses"]["ask_back"] == 1 and m["n_silent"] == 2
    assert m["detection_with_ask_back"]["tp"] == 1 and m["detection"]["tp"] == 0


def test_sweep_monotone():
    rng = random.Random(3)
    recs = []
    for i in range(200):
        label = rng.choice(["error", "valid", "follows_error"])
        hat_status = rng.choice(["error", "error", "valid", "unchecked"])
        cands = [cd(0.0, hat_status == "valid", rng.random() < 0.5, 0)]
        cands += [cd(rng.random() * 2, rng.random() < 0.6, rng.random() < 0.3, j) for j in range(1, 5)]
        recs.append(rec(label, 1 + i % 3, hat_status, cands[0]["eq_said"], cands))
    deltas = [0.0, 0.1, 0.25, 0.5, 1.0, 2.0, 3.0]
    rows = sweep(recs, deltas)
    bound = [r["bound_event_rate"] for r in rows]
    resolved = [r["statuses"].get("corrected", 0) + r["statuses"].get("ask_back", 0) for r in rows]
    errors = [r["statuses"].get("math_error", 0) for r in rows]
    assert bound == sorted(bound) and resolved == sorted(resolved) and errors == sorted(errors, reverse=True)
    assert rows[0]["statuses"].get("corrected", 0) == 0


def test_metrics_accepts_real_run_rows():
    rows = [{"source": "rec", "label": "error", "status": "verified", "hat_status": "valid",
             "hat_eq_said": False, "out_eq_said": False}]
    assert metrics(rows)["scr"] == 1.0


def test_bound_pvalues_beta_binomial_less_extreme_than_binomial():
    from src.calibrate import bound_pvalues
    # phase 4 numbers: k=16, n=327; homophone v1 11/100
    p = bound_pvalues(11, 100, 16, 327)
    assert abs(p["bound"] - 16 / 328) < 1e-3
    assert abs(p["binomial"] - 0.00966) < 2e-4
    assert abs(p["beta_binomial"] - 0.0218) < 5e-4
    assert p["beta_binomial"] > p["binomial"]          # calibration randomness widens the null
    assert bound_pvalues(0, 100, 16, 327)["beta_binomial"] == 1.0


def test_cluster_calibration_collapses_pairs():
    from src.calibrate import cluster_calibration
    ms = [0.1, 0.2, 0.3, 0.4, math.inf, math.inf]
    cl = ["a", "a", "b", "b", "c", "c"]
    out = cluster_calibration(ms, cl, alpha=0.5)
    assert out["n_items"] == 6 and out["n_clusters"] == 3
    assert out["min"] == (0.3, 2, 3) and out["max"] == (0.4, 2, 3)
