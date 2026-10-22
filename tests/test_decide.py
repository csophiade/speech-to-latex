from src.decide import classes, decide


def c(latex, delta, status, cls=None, explanation=None):
    d = {"latex": latex, "delta": delta, "status": status, "explanation": explanation}
    if cls is not None:
        d["cls"] = cls
    return d


def test_line_zero_is_given():
    out = decide([c("x^2 = 4", 0.0, "unchecked", 0)], delta=1.0, line_index=0)
    assert out["status"] == "given" and out["output"] == "x^2 = 4"


def test_verified_when_hat_valid():
    out = decide([c("x = 2", 0.0, "valid", 0), c("x = 3", 0.1, "error", 1)], 1.0, 1)
    assert out["status"] == "verified" and out["output"] == "x = 2" and out["note"] is None


def test_caution_is_verified_with_note():
    out = decide([c("x = 2", 0.0, "caution", 0, "squaring may add roots")], 1.0, 1)
    assert out["status"] == "verified" and out["note"] == "squaring may add roots"


def test_hat_is_smallest_delta_not_list_order():
    out = decide([c("x = 3", 0.4, "error", 1), c("x = 2", 0.0, "valid", 0)], 1.0, 1)
    assert out["status"] == "verified" and out["output"] == "x = 2"


def test_corrected_single_class():
    cands = [c("x = 15", 0.0, "error", 0, "solutions changed"),
             c("x = 50", 0.3, "valid", 1), c("x = 50.0", 0.5, "valid", 1)]
    out = decide(cands, 1.0, 2)
    assert out["status"] == "corrected" and out["output"] == "x = 50" and out["heard"] == "x = 15"


def test_corrected_needs_delta_strictly_below():
    cands = [c("x = 15", 0.0, "error", 0, "why"), c("x = 50", 0.3, "valid", 1)]
    assert decide(cands, 0.3, 2)["status"] == "math_error"
    assert decide(cands, 0.0, 2)["status"] == "math_error"


def test_ask_back_two_classes():
    cands = [c("y = 1", 0.0, "error", 0, "why"), c("y = 2", 0.2, "valid", 1),
             c("y = 2", 0.25, "valid", 1), c("y = 3", 0.4, "caution", 2), c("y = 4", 0.45, "valid", 3)]
    out = decide(cands, 1.0, 1)
    assert out["status"] == "ask_back" and out["options"] == ["y = 2", "y = 3"]
    assert out["output"] == "y = 1"


def test_ask_back_shrinks_to_corrected_at_smaller_delta():
    cands = [c("y = 1", 0.0, "error", 0, "why"), c("y = 2", 0.2, "valid", 1), c("y = 3", 0.4, "valid", 2)]
    assert decide(cands, 0.3, 1)["status"] == "corrected"


def test_math_error_keeps_hat_with_explanation():
    cands = [c("x = 0", 0.0, "error", 0, "dividing by x may lose x = 0"), c("x = 1", 2.0, "valid", 1)]
    out = decide(cands, 1.0, 3)
    assert out["status"] == "math_error" and out["output"] == "x = 0"
    assert out["explanation"] == "dividing by x may lose x = 0"


def test_unchecked_hat_never_valid():
    cands = [c(r"\text{oops}", 0.0, "unchecked", 0), c("x = 1", 2.0, "valid", 1),
             c("x = 9", 0.5, "unchecked", 2)]
    assert decide(cands, 1.0, 1)["status"] == "unchecked"
    assert decide(cands, 3.0, 1)["status"] == "corrected"


def test_classes_from_symeq_when_cls_missing():
    assert classes(["(x+1)^2", "x^2 + 2x + 1", "x^2 + 1"]) == [0, 0, 2]
    cands = [c("y = 0", 0.0, "error"), c("y = (x+1)^2", 0.1, "valid"), c("y = x^2 + 2x + 1", 0.2, "valid")]
    assert decide(cands, 1.0, 1)["status"] == "corrected"
