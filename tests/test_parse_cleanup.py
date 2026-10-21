"""Task 2.0: graded parsing (clean / cleaned / none), structured relations, meaning preservation."""

import time

import pytest
import sympy
from sympy import Eq, Function, Symbol as S, Tuple

from src import eval as ev
from src.mathparse import parse_graded, parses, time_limit

x, y, m, a, b, c, t, T, i = sympy.symbols("x y m a b c t T i")


def obj_level(s):
    return parse_graded(s)


# ---------- A. cosmetic references that must now parse ----------

def test_greek_and_pi():
    o, lv = obj_level(r"\theta = \frac{\pi}{4}")
    assert lv == "clean" and o == Eq(S("theta"), sympy.pi / 4)


def test_tanh():
    o, lv = obj_level(r"v=\tanh\psi")
    assert lv == "clean" and o == Eq(S("v"), sympy.tanh(S("psi")))


def test_plain_expression_clean():
    o, lv = obj_level(r"- nd_T \cos \theta + \frac { n^2 d_T^2 \sin^2 \theta } { 2r }")
    assert lv == "clean" and o is not None


def test_hat():
    o, lv = obj_level(r"\hat{p} = m \hat{v}")
    assert lv == "cleaned" and o == Eq(S(r"\hat{p}"), m * S(r"\hat{v}"))


def test_double_brace_limit():
    o, lv = obj_level(r"\lim_{{x \to 4}} \sqrt{x}")
    assert lv == "cleaned" and isinstance(o, sympy.Limit)
    assert o.args[:3] == (sympy.sqrt(x), x, 4)


def test_bracket_pair():
    o, lv = obj_level(r"[-7,7]")
    assert lv == "cleaned" and o == Tuple(S("__bracket__"), -7, 7)


def test_text_function_name():
    o, lv = obj_level(r"\text{Kernel}(T)")
    assert lv == "cleaned" and o == Function("Kernel")(T)


def test_in_half_open_interval():
    o, lv = obj_level(r"\epsilon\in [ 0 , K^ * )")
    assert lv == "cleaned"
    assert o == Tuple(S("__rel__:in"), S("epsilon"), Tuple(S("__interval[)__"), 0, S("K^{*}")))


def test_mathcal_subscript():
    o, lv = obj_level(r"\mathcal { B } _1")
    assert lv == "cleaned" and o == S(r"\mathcal{B}_{1}")


def test_implication_chain():
    o, lv = obj_level(r"3y - 5 = 7 \quad \Rightarrow \quad y = 4")
    assert lv == "cleaned" and o.args[0] == S("__rel__:=,Rightarrow,=") and len(o.args) == 5


def test_command_prime_kept():
    o, lv = obj_level(r"\alpha' = \frac{d\alpha}{dt}")
    assert lv == "cleaned" and o.lhs == S("\\alpha'") and o.lhs != S("alpha")


def test_reordered_comma_subscript():
    o, lv = obj_level(r"\rho^ { X } _ { 1/2 , L }")
    assert lv == "cleaned" and o == sympy.Pow(S(r"\rho_{1/2,L}"), S("X"))


def test_text_single_letter_function():
    o, lv = obj_level(r"\text{g}(x)=x-1")
    assert lv == "cleaned" and isinstance(o, sympy.Eq)
    assert o.lhs == Function("g")(x) and sympy.expand(o.rhs - (x - 1)) == 0   # rhs is unevaluated


def test_paren_tuple():
    o, lv = obj_level(r"( 0 , i )")
    assert lv == "cleaned" and o == Tuple(S("__tuple__"), 0, i)


def test_norm():
    o, lv = obj_level(r"\text{Condition number } \kappa(A) = \|A\| \|A^{-1}\|")
    assert lv == "cleaned"
    assert any(f.func == Function("__norm__") for f in o.atoms(sympy.core.function.AppliedUndef))


def test_text_prose_symbol():
    o, lv = obj_level(r"\tau = \text{shear stress}")
    assert lv == "cleaned" and o == Eq(S("tau"), S("shearstress"))


def test_map_arrow_no_arrow_symbol():
    o, lv = obj_level(r"Z_ { q } : \widehat { \mathfrak { H } ^ { 0 } } \rightarrow \mathbb { C }")
    assert lv == "cleaned" and o.args[0] == S("__rel__::,to")
    assert not any("arrow" in str(s) for s in o.atoms(sympy.Symbol))


# ---------- B. meaning preservation ----------

NOT_EQUAL = [
    (r"\bar{x}", r"x"),
    (r"\overline{x}", r"x"),
    (r"\mathcal{N}", r"N"),
    (r"x'", r"x"),
    (r"\alpha'", r"\alpha"),
    (r"(a,b)", r"(b,a)"),
    (r"(a,b)", r"[a,b]"),
    (r"\{a,b\}", r"(a,b)"),
    (r"a \in B", r"a \subset B"),
    (r"G^{++}", r"G"),
    (r"F^+", r"F^-"),
    (r"x_{ij}", r"x_{ji}"),
    (r"x \pm y", r"x + y"),
    (r"x \pm y", r"x \mp y"),
    (r"x^{(3)}", r"x^3"),
    (r"\rho^{X}_{1}", r"\rho^{X}"),
    (r"\|A\|", r"|A|"),
    (r"a=b=c", r"a=c"),
    (r"A \rightarrow B", r"A arrow B"),
]


@pytest.mark.parametrize("p,q", NOT_EQUAL)
def test_distinct_stay_distinct(p, q):
    assert parses(p) and parses(q)
    assert ev.symeq(p, q) == "not_equal"
    assert ev.symeq(q, p) == "not_equal"


# ---------- C. intended equalities ----------

EQUAL = [
    (r"\{a,b\}", r"\{b,a\}"),
    (r"\hat{p} = m \hat{v}", r"m \hat{v} = \hat{p}"),
    (r"\varepsilon x", r"\epsilon x"),
    (r"\text{H} \approx 0", r"H \approx 0"),
    (r"\bar{x}", r"\overline{x}"),
    (r"\mathrm{d}x", r"dx"),
]


@pytest.mark.parametrize("p,q", EQUAL)
def test_intended_equalities(p, q):
    assert ev.symeq(p, q) == "equal"


# ---------- D. regressions ----------

def test_chain_is_structured():
    o, lv = obj_level(r"a = b = c")
    assert lv == "cleaned" and o == Tuple(S("__rel__:=,="), a, b, c)
    assert not isinstance(o, sympy.Eq)


def test_lim_arrow_not_split():
    o, lv = obj_level(r"\lim_{x \to 5} f(x)")
    assert lv == "clean" and isinstance(o, sympy.Limit)


def test_prose_only_and_empty():
    assert obj_level(r"\text{hello world}") == (None, "none")
    assert not parses(r"\text{hello world}")
    assert obj_level("") == (None, "none")


def test_plain_equation_clean():
    o, lv = obj_level(r"\frac{x^2 + 1}{2} = 50")
    assert lv == "clean" and isinstance(o, sympy.Eq)


def test_singleton_set_not_number():
    o, lv = obj_level(r"\{1\}")
    assert lv == "cleaned" and o == Tuple(S("__set__"), 1)


def test_invented_integration_variable():
    assert obj_level(r"\int f") == (None, "none")


def test_second_derivative():
    o, lv = obj_level(r"\frac{d^2 x}{dt^2}")
    assert lv == "cleaned" and o == sympy.Derivative(x, (t, 2))


def test_level_values():
    for s in [r"x^2", r"\bar{x}", r"\text{hello world}"]:
        o, lv = parse_graded(s)
        assert lv in ("clean", "cleaned", "none") and (o is None) == (lv == "none")


def test_parse_timeout_bounded():
    long = " + ".join(r"\frac{x^{%d} + \sqrt{y_{%d}}}{(x - %d)^2}" % (k, k, k) for k in range(400))
    start = time.perf_counter()
    parse_graded(long, timeout=1)
    assert time.perf_counter() - start < 3


def test_symeq_report():
    refs = [r"x^2", r"\text{hello world}", r"\bar{x}", r"a=b"]
    hyps = [r"x^{2}", r"x", r"x", r"b=a"]
    verdicts = [ev.symeq(r, h) for r, h in zip(refs, hyps)]
    rep = ev.symeq_report(refs, verdicts)
    assert rep["symeq_coverage"] == 0.75
    assert rep["symeq_on_parseable"] == round(2 / 3, 4)
    metrics, _ = ev.summarise(refs, hyps, [0.1] * 4)
    assert {"symeq", "symeq_unparseable", "symeq_coverage", "symeq_on_parseable"} <= set(metrics)


# ---------- E. review fixes (task 2.0, round 2) ----------

REVIEW_NOT_EQUAL = [
    # derivatives of plain symbols must not evaluate to 0
    (r"\frac{d^2 y}{dx^2} = -y", r"\frac{d^2 y}{dx^2} = y"),
    (r"\frac{d^2 y}{dx^2} = -y", r"y = 0"),
    (r"\frac{d^2 y}{dx^2}", r"\frac{dy}{dx}"),
    (r"\frac{dy}{dx}", r"0"),
    (r"\frac{\partial u}{\partial t} = k \frac{\partial^2 u}{\partial x^2}", r"0 = 0"),
    (r"\alpha' = \frac{d\alpha}{dt}", r"\alpha' = 0"),
    # identities / numeric facts must not all compare equal
    (r"2 + 3 = 5", r"2 + 2 = 4"),
    (r"\frac{1}{2} - \frac{1}{4} = \frac{1}{4}", r"\frac{9}{10} \div 0.3 = 3"),
    (r"2+3=5", r"x=x"),
    (r"x^2 - 1 = (x-1)(x+1)", r"0 = 0"),
    (r"1 < 2", r"3 < 4"),
    # nabla: div / curl / grad / laplacian stay distinct
    (r"\nabla \cdot F", r"\nabla \times F"),
    (r"\nabla f", r"\nabla \cdot f"),
    (r"\nabla^2 f", r"\nabla f"),
    (r"\nabla^2 \mathbf{E} - \frac{1}{c^2} \frac{\partial^2 \mathbf{E}}{\partial t^2} = 0", r"\nabla^2 \mathbf{E} = 0"),
    # \text keeps non-alphanumerics
    (r"5 \text{ m/s}", r"5 \text{ ms}"),
    (r"y, \text{x > 0}", r"y, \text{x < 0}"),
    (r"a \text{-} b", r"a b"),
    (r"a \text{ - } b", r"a \text{ + } b"),
    # \quad between statements is a separator, not multiplication
    (r"a=1 \quad b=2", r"a = b = 2"),
]


@pytest.mark.parametrize("p,q", REVIEW_NOT_EQUAL)
def test_review_distinct_stay_distinct(p, q):
    assert ev.symeq(p, q) in ("not_equal", "unparseable")
    assert ev.symeq(q, p) in ("not_equal", "unparseable")


REVIEW_EQUAL = [
    (r"\frac{d}{dx} x^2 = 2x", r"2x = \frac{d}{dx} x^2"),
    (r"\frac{dy}{dx} = 3y", r"3y = \frac{dy}{dx}"),
    (r"x^2 - 1 = (x-1)(x+1)", r"(x-1)(x+1) = x^2 - 1"),
    (r"\sum_{\substack{i=1}}^{n} x_i", r"\sum_{i=1}^{n} x_i"),
    (r"{x = 1}", r"x = 1"),
    (r"\{1,1,2\}", r"\{1,2\}"),
    (r"x \equiv 1 \pmod 3", r"x \equiv 1 \pmod{3}"),
]


@pytest.mark.parametrize("p,q", REVIEW_EQUAL)
def test_review_intended_equalities(p, q):
    assert ev.symeq(p, q) == "equal"


def test_nabla_operators_structured():
    o, lv = obj_level(r"\nabla^2 f = \nabla \cdot (\nabla f)")
    assert lv == "cleaned"
    F = sympy.Function
    assert o == Eq(F("__lap__")(S("f")), F("__div__")(F("__grad__")(S("f"))))


def test_quad_separator():
    o, lv = obj_level(r"a=1 \quad b=2")
    assert lv == "cleaned" and o.args[0] == S("__rel__:quad")


def test_text_keeps_symbols():
    o, lv = obj_level(r"5 \text{ m/s}")
    assert lv == "cleaned" and S("m/s") in o.free_symbols


def test_substack():
    o, lv = obj_level(r"\sum_{\substack{i=1}}^{n} x_i")
    assert lv == "cleaned" and isinstance(o, sympy.Sum)
    assert obj_level(r"\sum_{\substack{i=1\\ j=2}} x_{ij}") == (None, "none")   # never drop a condition


def test_eval_bar_rejected():
    assert obj_level(r"\left. f \right|_{x=0}") == (None, "none")
    assert obj_level(r"\left| x \right|^2")[1] == "clean"


def test_thousands_separator():
    assert ev.symeq(r"1,000", r"1, 0") != "equal"
    assert obj_level(r"1,000")[0] == 1000


def test_empty_decoration_and_lone_dots():
    for s in (r"\mathcal{}", r"\bar{}", r"\dots"):
        assert obj_level(s) == (None, "none")


def test_vanished_text_word_rejected():
    assert ev.symeq(r"f(x)=0 \text{ unless } x<0", r"f(x)=0<0") != "equal"
    assert ev.symeq(r"y = 0 \text{ unless } x", r"y = 0") != "equal"


def test_clean_level_is_raw():
    # spacing that HEAD's raw attempt stripped stays 'clean'; \quad and sizing commands need cleanup
    assert obj_level(r"x \, y")[1] == "clean"
    assert obj_level(r"x \bigl( y \bigr)")[1] == "cleaned"


def test_numeric_equal_does_not_swallow_timeout():
    start = time.perf_counter()
    v = ev.symeq(r"(a, \prod_{n=1}^{20000}(1+\frac{x}{n}))", r"(a, \prod_{n=1}^{20000}(1+\frac{y}{n}))", timeout=2)
    assert v == "timeout" and time.perf_counter() - start < 15


def test_length_cap_and_recursion():
    long = "+".join(f"x_{{{k}}}" for k in range(600))
    assert parse_graded(long, timeout=60) == (None, "none")
    assert ev.symeq("{" * 3000 + "x" + "}" * 3000, "y") in ("unparseable", "not_equal")

