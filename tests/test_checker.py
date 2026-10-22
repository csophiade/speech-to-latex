"""Residual-ratio checker (Phase 3.2): hand-written LaTeX fixtures, independent of derivgen."""

import pytest

from src.checker import check_derivation, check_step


def status(prev, cur):
    return check_step(prev, cur)["status"]


def why(prev, cur):
    r = check_step(prev, cur)
    assert r["status"] == "error", r
    return r["explanation"]


# ---------- valid steps ----------

@pytest.mark.parametrize("prev, cur", [
    (r"2x+3=7", r"2x=4"),                              # subtract a constant
    (r"2x=6", r"x=3"),                                 # divide by a constant
    (r"x^2-5x+6=0", r"(x-2)(x-3)=0"),                  # factorise
    (r"-2x \leq 6", r"x \geq -3"),                     # multiply by a negative, flipped
    (r"4x > 8", r"x > 2"),
])
def test_valid_steps(prev, cur):
    assert status(prev, cur) == "valid"


def test_constant_ratio_is_reported():
    assert check_step(r"2x=6", r"x=3")["g"] == "1/2"


def test_final_answers_match_solveset():
    assert status(r"(x-2)(x-3)=0", r"x=2 \text{ or } x=3") == "valid"
    assert status(r"x^2-5x+6=0", r"x=\frac{5 \pm 1}{2}") == "valid"
    assert status(r"x^2=4", r"x \in \{2, -2\}") == "valid"



def test_unevaluated_stated_values():
    # fractions and \sqrt{16} in a stated answer are evaluated before comparing sets
    assert status(r"(2x-3)(x-4)=0", r"x = \frac{3}{2} \text{ or } x = 4") == "valid"
    formula = r"x = \frac{12 + \sqrt{16}}{8} \text{ or } x = \frac{12 - \sqrt{16}}{8}"
    assert status(r"4x^2 - 12x + 8 = 0", formula) == "valid"
    assert status(formula, r"x = 2 \text{ or } x = 1") == "valid"   # answer to answer
    assert status(formula, r"x = 2 \text{ or } x = 3") == "error"


def test_equality_chain():
    assert status(r"2(x+1)=4", r"2(x+1)=2x+2=4") == "valid"
    assert "not equal" in why(r"2(x+1)=4", r"2(x+1)=2x+1=4")


# ---------- the six injected error types ----------

def test_sign_flip():
    assert "sign of 3" in why(r"2x+3=7", r"2x=7+3")


def test_dropped_term():
    assert "dropped" in why(r"2x+3=7", r"2x=7")


def test_wrong_distribution():
    assert "expansion of 2(x + 3)" in why(r"2(x+3)=10", r"2x+3=10")


def test_divide_by_variable():
    assert why(r"x^2=3x", r"x=3") == "dividing by x may lose the solution x = 0"


@pytest.mark.parametrize("prev, cur", [(r"\sqrt{x+2}=x", r"x+2=x^2"), (r"x-1=2", r"(x-1)^2=4")])
def test_square_both_sides(prev, cur):
    assert "squaring both sides can add solutions where L = -R: x = -1" in why(prev, cur)


def test_missed_flip():
    assert "must flip the inequality" in why(r"-2x \leq 6", r"x \leq -3")


# ---------- caution, other errors, unchecked ----------

def test_clearing_nonzero_denominator_is_caution():
    r = check_step(r"\frac{3}{x-1}=2", r"3=2(x-1)")
    assert r["status"] == "caution" and "x ≠ 1" in r["explanation"]


def test_answer_outside_domain():
    assert "outside the domain" in why(r"\frac{x}{x-1}=\frac{1}{x-1}", r"x=1")


def test_strictness_change():
    assert status(r"x+1<3", r"x \le 2") == "error"


def test_missing_root_in_final_answer():
    assert "x = 3" in why(r"(x-2)(x-3)=0", r"x=2")


def test_unparseable_is_unchecked():
    assert status(r"2x+3=7", r"2x+3=") == "unchecked"
    assert status(r"2x+3=", r"x=2") == "unchecked"


def test_derivation_with_gap():
    ok = check_derivation([r"2x+3=7", r"2x=", r"x=2"])
    assert [r["status"] for r in ok] == ["given", "unchecked", "valid"]   # equivalence across the gap
    bad = check_derivation([r"2x+3=7", r"2x=", r"x=3"])
    assert [r["status"] for r in bad] == ["given", "unchecked", "unchecked"]   # cannot localise


def test_derivation_flags_first_wrong_line():
    lines = [r"3(x-1)=2x+5", r"3x-3=2x+5", r"3x=2x+2", r"x=2"]
    out = check_derivation(lines)
    assert [r["status"] for r in out] == ["given", "valid", "error", "valid"]
    assert out[2]["explanation"].startswith("check the sign of -3")


# ---------- review regressions ----------

def test_implicit_product_factoring():
    # dictated factoring is written x(x - 3), which the parser reads as a call x(...)
    assert status(r"x^2 - 3x = 0", r"x(x - 3) = 0") == "valid"
    assert status(r"x(x - 3) = 0", r"x = 0 \text{ or } x = 3") == "valid"
    assert status(r"2x^2 - 8x = 0", r"2x(x - 4) = 0") == "valid"
    assert "x = 3" in why(r"x(x - 3) = 0", r"x = 0")


def test_extraneous_root_after_squaring():
    head = [r"\sqrt{2x+3} = x", r"2x + 3 = x^2", r"x^2 - 2x - 3 = 0"]
    right = check_derivation(head + [r"x = 3"])
    assert right[-1]["status"] == "valid" and right[-1]["explanation"] is None
    wrong = check_derivation(head + [r"x = 3 \text{ or } x = -1"])
    assert wrong[-1]["status"] == "error"
    assert wrong[-1]["explanation"].startswith("x = -1 does not satisfy the original equation")
    rejected = check_derivation([r"\sqrt{x+2} = x", r"x + 2 = x^2", r"(x-2)(x+1)=0",
                                 r"x = 2 \text{ or } x = -1", r"x = 2"])
    assert rejected[-1]["status"] == "valid"   # rejecting the extraneous root is right


@pytest.mark.parametrize("prev, cur, bracket", [
    (r"2(x + 6) + 6 = -3x + 24", r"2x + 6 + 6 = -3x + 24", "2(x + 6)"),   # lin-0058
    (r"-9 = -5(x + 3)", r"-9 = -5x + 3", "-5(x + 3)"),                     # rat-0073
    (r"3(x + 3) - 6 = x", r"3x + 3 - 6 = x", "3(x + 3)"),                  # lin-0026
    (r"-2(x - 4) = 10", r"-2x - 8 = 10", "-2(x - 4)"),
])
def test_expansion_slip_not_blamed_on_a_coincident_term(prev, cur, bracket):
    assert why(prev, cur).startswith(f"check the expansion of {bracket}")


def test_dropped_term_next_to_a_bracket():
    assert "the term 5 seems to have been dropped" in why(r"2(x + 3) + 5 = 11", r"2x + 6 = 11")


def test_swapped_sides_explained():
    assert why(r"2x + 3 < 7", r"7 < 2x + 3").startswith("swapping the two sides reverses the inequality sign")


def test_caution_lists_every_excluded_value():
    r = check_step(r"\frac{2}{x} = \frac{3}{x+1}", r"2(x+1) = 3x")
    assert r["status"] == "caution" and "x(x + 1)" in r["explanation"]
    assert "x ≠ -1 and x ≠ 0" in r["explanation"]
    bare = check_derivation([r"\frac{3}{x} + 1 = \frac{5}{x}", r"3 + x = 5"])[1]["explanation"]
    assert bare == "multiplying by x is only allowed because x ≠ 0 here"


def test_identity_step_is_valid():
    r = check_step(r"2(x+1) = 2x + 2", r"0 = 0")
    assert r["status"] == "valid" and "every real number" in r["explanation"]
    assert check_step(r"x = x", r"0 = 0")["g"] is None
    assert status(r"x = 3", r"0 = 0") == "error"


def test_no_solution_and_compound_answers():
    out = check_derivation([r"\frac{x}{x-3} = \frac{3}{x-3} + 2", r"x = 3 + 2(x - 3)", r"x = 3",
                            r"\text{no solution}"])
    assert out[-1]["status"] == "valid"
    assert status(r"x(1 - 2x) > 0", r"0 < x < \frac{1}{2}") == "valid"
    assert status(r"x(1 - 2x) > 0", r"x > 0 \text{ and } 1 > 2x") == "valid"
    assert status(r"x(1 - 2x) > 0", r"0 < x < 1") == "error"


@pytest.mark.parametrize("prev, cur", [
    (r"x = 10^{10^{10}}", r"x = 1"), (r"2x = 4", r"x = 2^{2^{30}}"), (r"2x = 4", r"x = 9^{9^{9}}")])
def test_huge_powers_are_unchecked_quickly(prev, cur):
    import time
    t = time.monotonic()
    assert status(prev, cur) == "unchecked"
    assert time.monotonic() - t < 5


@pytest.mark.parametrize("lines", [
    [r"2x=4", r"x \in (\frac{1}{0}, 1)"], [r"x \in (\frac{1}{0}, 1)", r"x = 2"],
    [r"2x=4", r"x = \frac{0}{0}"], [r"2x=4", r"x = 2 = \frac{1}{0}"]])
def test_undefined_values_are_unchecked(lines):
    assert check_derivation(lines)[1]["status"] == "unchecked"


def test_answer_lines_report_no_nonconstant_g():
    assert check_step(r"\frac{1}{x}=1", r"x=1") == {"status": "valid", "g": None, "explanation": None}
    r = check_step(r"\frac{1}{x - 0.5} = 2", r"1 = 2(x - 0.5)")
    assert "x ≠ 1/2" in r["explanation"]
    assert "ImageSet" not in (check_step(r"\sin(x) = 0", r"x = 0")["explanation"] or "")


def test_bounded_check_from_a_thread():
    import threading
    from src.checker import check_derivation_bounded
    res = {}

    def run():
        res["ok"] = check_derivation_bounded([r"2x=4", r"x=3"])
        res["late"] = check_derivation_bounded([r"2x=4", r"x=2"], timeout=0.001)
    t = threading.Thread(target=run)
    t.start()
    t.join(120)
    assert [r["status"] for r in res["ok"]] == ["given", "error"]
    assert [r["status"] for r in res["late"]] == ["given", "unchecked"]


def test_dropped_term_from_an_expansion_or_its_side():
    assert "the term 24 seems to have been dropped when expanding 6(x + 4)" in why(r"19 = 6(x + 4)", r"19 = 6x")
    assert "the term -7 seems to have been dropped" in why(r"6(x + 2) - 7 = -4x - 7", r"6x + 12 = -4x - 7")
