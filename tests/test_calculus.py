"""Regression tests: calculus steps (derivatives, integrals, evaluation bars), "and" vs "or",
operation-based explanations. Each verdict is one of valid / caution / error / unchecked;
a failure to verify must never come back as valid."""

import re

import pytest

from src.checker import check_step

CASES = [
    # indefinite integrals: antiderivative checked by differentiating back; + C expected
    (r"\int 5x\,dx=0", r"\frac{5}{2}x^2 + C=0", "valid", None),
    (r"\int 5x\,dx=0", r"\frac{5}{2}x^2=0", "caution", "constant of integration"),
    (r"\int 5x\,dx=0", r"5x^2 + C=0", "error", "not an antiderivative"),
    (r"y = \int 5x\,dx", r"y = \frac{5}{2}x^2 + C", "valid", None),
    (r"y = \int 5x\,dx", r"y = 5x^2 + C", "error", "not an antiderivative"),
    # derivatives: evaluated, never read as d * x products
    (r"\frac{d}{dx}(5x^2+5x)=10", r"10x+5=10", "valid", None),
    (r"\frac{d}{dx}(5x^2+5x)=10", r"10x=10", "error", "derivative of 5x^2 + 5x is 10x + 5"),
    # definite integrals: antiderivative error vs evaluation error are told apart
    (r"\int_0^1 \cos x\,dx=0", r"-\sin x\Big|_0^1=0", "error", "incorrect antiderivative"),
    (r"\int_0^1 \cos x\,dx=0", r"\left[\sin x\right]_0^1=0", "valid", None),
    (r"\int_0^1 \cos x\,dx=0", r"\sin x \right|_{0}^{1}=0", "valid", None),
    (r"\int_0^1 \cos x\,dx=0", r"\sin 2=0", "error", "incorrect evaluation"),
    (r"\left[\sin x\right]_0^1=0", r"\sin 1 - \sin 0=0", "valid", None),
    # "and" is a logical error where "or" is meant
    (r"(x-2)(x-3)=0", r"x=2 \text{ and } x=3", "error", "did you mean"),
    (r"(x-2)(x-3)=0", r"x=2 \text{ or } x=3", "valid", None),
    # operation-based wording
    (r"x+50=90", r"x=30", "error", "subtracting 50 from both sides gives x = 40, not x = 30"),
    (r"3x=12", r"x=3", "error", "dividing both sides by 3 gives x = 4, not x = 3"),
    (r"x+50=90", r"x=40", "valid", None),
]


@pytest.mark.parametrize("prev,cur,status,phrase", CASES)
def test_calculus_and_logic(prev, cur, status, phrase):
    out = check_step(prev, cur)
    assert out["status"] == status, out
    if phrase:
        assert phrase.lower() in (out["explanation"] or "").lower(), out


def test_reported_example_no_longer_crashes():
    """Owner's report: \\int 5x dx = 0 then "5 over 2 x squared ... = 0" gave PrintMethodNotImplementedError."""
    for cur in (r"\frac{5}{2}x^2=0", r"\frac{5}{2}x^2+c=0"):
        out = check_step(r"\int 5x\,dx=0", cur)
        assert "PrintMethodNotImplementedError" not in (out["explanation"] or "")
        assert out["status"] in {"valid", "caution"}


def test_unreadable_is_never_valid():
    out = check_step(r"\int_0^1 \cos x\,dx=0", r"-\sin x \Big|_{0}^{")
    assert out["status"] == "unchecked"


def test_display_keeps_structure():
    """'52x2' is only the flattened text of correct MathML, not lost structure."""
    from demo.ui_render import mathml
    m = mathml(r"\frac{5}{2}x^2=0")
    assert "<mfrac>" in m and "<msup>" in m
    m = mathml(r"\int_0^1 \cos x\,dx=0")
    assert "<msubsup>" in m and re.search(r"&#x0222B;|∫", m)
