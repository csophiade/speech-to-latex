"""ProcessBench step extraction and policy (3.4): hand-written step strings only, no benchmark rows."""

import pytest

from src.stepmath import check_steps, clean_latex, extract, is_algebra_step, predict, spans


def squash(t):
    return " ".join(t.split())


def latex(step):
    return [squash(d["latex"]) for d in extract(step)]


def statuses(*steps):
    return [st["statuses"] for st in check_steps(list(steps))]


# ---------- spans and currency ----------

def test_spans_all_delimiters():
    s = r"A \(a = 1\), B \[b = 2\], C $c = 3$ and D $$d = 4$$."
    assert [x[2] for x in spans(s)] == ["a = 1", "b = 2", "c = 3", "d = 4"]


def test_currency_dollars_are_prose():
    assert spans("He had $12 and spent $5 on lunch.") == []
    assert latex("He had $12 and spent $5, so $12 - $5 = $7.") == ["12 - 5=7"]


def test_environment_split_into_lines():
    s = r"\begin{aligned} 3y + 1 &= 10 \\ 3y &= 9 \end{aligned}"
    assert latex(s) == ["3y + 1 = 10", "3y = 9"]


# ---------- clean_latex ----------

@pytest.mark.parametrize("raw, out", [
    (r"\boxed{12{,}500}", "12500"),
    (r"\dfrac{3}{4} = 0.75", r"\frac{3}{4} = 0.75"),
    (r"5 \text{ apples} + 2 \text{ apples} = 7 \text{ apples}", "5 + 2 = 7"),
    (r"25\% \times 8 = 2", r"(\frac{25}{100}) \times 8 = 2"),
    (r"45^\circ", r"(45 \cdot \frac{\pi}{180})"),
])
def test_clean_latex(raw, out):
    assert squash(clean_latex(raw)) == out


def test_clean_latex_literal_and_conversion():
    assert clean_latex(r"20\%", literal=True) == "20"
    assert clean_latex(r"3 \text{ feet} = 36 \text{ inches}") == ""


def test_named_members_and_approx_dropped():
    assert latex(r"\(\text{Cost} = 4 \times 2 = 8\)") == [r"4 \times 2 = 8"]
    assert latex(r"\(x = \sqrt{2} \approx 1.41\)") == [r"x = \sqrt{2}"]


# ---------- plain-text arithmetic ----------

@pytest.mark.parametrize("step, out", [
    ("Each box holds 36/4 = 9 pens.", ["36/4=9"]),
    ("She walks 2 hours * 3 miles/hour = 6 miles.", [r"2 \times 3=6"]),
    ("We need 7 (green) + 5 (yellow) = 12 balls.", ["7 + 5=12"]),
    ("A ticket costs 15 x 3 = 45 dollars.", [r"15 \times 3=45"]),
])
def test_bare_lines(step, out):
    assert latex(step) == out


@pytest.mark.parametrize("step", [
    "Half of 30 = 15 cookies.",                         # "of": the left operand is prose
    "There are 3 hours = 180 minutes in the trip.",     # unit conversion
    "So 50 / 7 = 7 with a remainder of 1.",             # remainder
    "To find the total, we add the two amounts.",       # no maths
])
def test_bare_lines_skipped(step):
    assert latex(step) == []


# ---------- line flags ----------

def test_hypothesis_negation_alt_flags():
    hyp = extract(r"Suppose \(t = 4\); then the sum is even.")[0]
    neg = extract(r"This gives \(5 = 2\), which is impossible.")[0]
    alt = extract(r"Hence \(2a - 1 = 0\) or \(a + 4 = 0\).")
    assert hyp["hyp"] and not hyp["neg"]
    assert neg["neg"]
    assert [d["alt"] for d in alt] == [True, True]


def test_solutions_joined():
    assert latex(r"The roots are \(m = 6\) and \(m = -2\).") == [r"m = 6 \text{ or } m = -2"]


def test_continuation_appended():
    assert latex(r"$$5k + 2k$$ $$= 7k$$ and $$k = 1$$") == ["k = 1"]
    assert latex(r"$$p = 2q + 3q$$ $$= 5q$$") == ["p = 2q + 3q = 5q"]


@pytest.mark.parametrize("step", [
    r"So \(\arcsin 1 = 1.57\).",
    r"Then \(\log 100 = 2\).",
    r"We get \(3\frac{1}{2} = 3.5\).",
    r"Define \(a \star b = 2a + b\), so \(1 \star 2 = 4\).",
])
def test_non_numeric_lines(step):
    assert not any(d["numeric"] for d in extract(step))


def test_is_algebra_step():
    assert is_algebra_step(r"Solving, \(4w - 2 = 10\).")
    assert not is_algebra_step("Next we look at how many friends there are.")


# ---------- verdicts inside one line ----------

def test_within_line_arithmetic():
    assert statuses(r"So \(6 \times 7 = 42\).") == [["valid"]]
    assert statuses(r"So \(6 \times 7 = 48\).") == [["error"]]
    assert statuses("The cost is 12 + 9 = 22 dollars.") == [["error"]]


def test_within_line_rounding_and_percent():
    assert statuses(r"So \(\frac{2}{3} = 0.67\).") == [["valid"]]
    assert statuses(r"The share is \(\frac{12}{48} \times 100 = 25\%\).") == [["valid"]]


def test_within_line_symbol_links_skipped():
    assert statuses(r"Then \(u = v = 3\).") == [[]]


def test_negated_error_dropped():
    assert statuses(r"That gives \(1 = 3\), a contradiction.") == [[]]


# ---------- verdicts across lines ----------

def test_cross_line_error_and_valid():
    assert statuses(r"\(5n - 4 = 11\)", r"So \(5n = 15\).") == [[], ["valid"]]
    assert statuses(r"\(5n - 4 = 11\)", r"So \(5n = 7\).") == [[], ["error"]]


def test_cross_line_narrowing_is_unchecked():
    assert statuses(r"\(r^2 = 9\)", r"Since r is positive, \(r = 3\).") == [[], []]


def test_cross_line_skips_hypothesis_and_degree_change():
    assert statuses(r"\(z^2 - 4 = 0\)", r"If \(z = 5\) we get more.") == [[], []]
    assert statuses(r"\(z^3 = 0\)", r"and \(z^2 - z - 6 = 0\)") == [[], []]


def test_predict_first_error_step():
    recs = check_steps(["We start.", r"\(2 + 2 = 4\)", r"\(3 + 3 = 7\)", r"\(1 + 1 = 3\)"])
    assert predict(recs) == 2
    assert predict(check_steps(["Plan only."])) == -1


# ---------- post-evaluation bug fixes (frozen-v6) ----------

def test_leading_product_is_not_an_equation_label():
    assert latex(r"\( (rst) + 3 = -19 \)") == ["(rst) + 3 = -19"]
    assert latex(r"so \( (ab) = 6 \)") == ["(ab) = 6"]
    assert latex(r"\[ (1)\quad 2x+1 = 5 \]") == ["2x+1 = 5"]
    assert statuses(r"\( (2)(3) = 6 \)") == [["valid"]]


def test_integer_quotient_skipped():
    assert statuses(r"\[ 175 \div 119 = 1 \quad \text{(quotient)} \]") == [[]]


def test_degrees_on_symbols_converted():
    assert statuses(r"\(\sum_{k=1}^{44}\cos(8k^\circ) = -1\)") == [["valid"]]
    assert squash(clean_latex(r"x^\circ = 30^\circ")) == r"(x \cdot \frac{\pi}{180}) = (30 \cdot \frac{\pi}{180})"


def test_function_application_not_compared_across_lines():
    assert statuses(r"\(f^{-1}(9) = 5\)", r"so \(f^{-1}(9+4) = 2f^{-1}(9) + 1\)") == [[], []]


def test_span_after_prose_operator_drops_operand():
    step = r"Gain = (expected number) × \( \frac{1}{100} = \frac{101}{100} \times \frac{1}{100} = \frac{101}{10000} \)"
    assert latex(step) == [r"\frac{101}{100} \times \frac{1}{100} = \frac{101}{10000}"]
    assert latex(r"Half of \( 10 = 5 \)") == []
