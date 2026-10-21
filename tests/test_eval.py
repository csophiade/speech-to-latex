import time

import pytest
import sympy

from src import eval as ev
from src.mathparse import parses

# (reference, hypothesis, NEM, SymEq)
PAIRS = [
    (r"x^{2}", r"x^2", True, "equal"),
    (r"\frac12", r"\frac{1}{2}", True, "equal"),
    (r"\dfrac{a}{b}", r"\frac{a}{b}", True, "equal"),
    (r"\left( x + 1 \right)^2", r"(x+1)^2", True, "equal"),
    (r"\lim_{{x \to 5}} \frac{{2}}{{x - 5}}", r"\lim_{x \to 5} \frac{2}{x - 5}", True, "equal"),
    (r"(x+1)^2", r"x^2 + 2x + 1", False, "equal"),
    (r"y = x^2", r"x^2 = y", False, "equal"),
    (r"x^2", r"x^3", False, "not_equal"),
    (r"x < 1", r"x > 1", False, "not_equal"),
    (r"y^2 = x^3 - 8x", r"y^2 = x^3 - 8", False, "not_equal"),
    (r"\tau = \text{shear stress}", r"\tau = \text{sheer stress}", False, "unparseable"),
]


@pytest.mark.parametrize("ref,hyp,want_nem,want_sym", PAIRS)
def test_pairs(ref, hyp, want_nem, want_sym):
    assert ev.nem(ref, hyp) is want_nem
    assert ev.symeq(ref, hyp) == want_sym


def test_normalise_rules():
    assert ev.normalise(r"$ \mathcal { X } = \ { x_1 , \dots \ } $") == r"\mathcalX=\{x_1,\dots\}"
    assert ev.normalise(r"x^{12}") == r"x^{12}"           # multi-token groups keep their braces
    assert ev.normalise(r"a \leq b \, .") == ev.normalise(r"a\le b")


def test_cer():
    assert ev.cer([r"x^{2}"], [r"x^2"]) == 0
    assert ev.cer(["abcd"], ["abce"]) == pytest.approx(0.25)
    assert ev.cer(["abcd"], [""]) == 1


def test_unparseable_and_parse_guards():
    assert not parses(r"\text{hello world}")
    assert not parses(r"a = b = c")
    assert not parses("")
    assert parses(r"\frac{x^2 + 1}{2} = 50")


def test_timeout(monkeypatch):
    # f is an undefined function, so the random-point check is inconclusive and simplify runs
    def slow_simplify(expr):
        time.sleep(10)

    monkeypatch.setattr(sympy, "simplify", slow_simplify)
    start = time.perf_counter()
    assert ev.symeq(r"f(2x)", r"f(x) + f(x)", timeout=1) == "timeout"
    assert time.perf_counter() - start < 3


def test_latency():
    with ev.latency() as t:
        time.sleep(0.05)
    assert 0.04 < t.seconds < 0.5
