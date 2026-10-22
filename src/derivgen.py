"""SymPy-checked student-style derivations with exact ground truth (Phase 3.1).

Five types: linear, quadratic_factor, quadratic_formula, rational, inequality; 4-6 LaTeX lines each,
line 0 the problem. Lines are built as strings from integer parameters (so they read like a student
wrote them, e.g. unexpanded brackets) and every step is then verified with `solveset`.

Final-answer convention (one form only): "x = 3", "x = -\\frac{13}{4}", two roots as
"x = 2 \\text{ or } x = 5", inequalities as "x > 4" / "x \\leq -\\frac{1}{2}". Products with a bare
x are written "x \\cdot (x - 3)" (the checker also reads the dictated "x(x - 3)" as a product, see
mathparse.implicit_products; the form is kept so the synthesised audio matches derivations.json).

Error items carry exactly one wrong line; later lines continue correctly from the wrong line, as a
student would, so step_valid is False only there. Parameters are resampled until every line is
"nice" (integer or simple-fraction answers) and the ground truth checks out.
"""

import random
from collections import Counter

import sympy
from sympy.core.function import AppliedUndef

from src.config import SEED, SYMPY_TIMEOUT_S
from src.mathparse import parse_graded, time_limit

TYPES = ["linear", "quadratic_factor", "quadratic_formula", "rational", "inequality"]
ERROR_TYPES = ["sign_flip", "dropped_term", "wrong_distribution", "divide_by_variable",
               "square_both_sides", "missed_flip"]
PREFIX = {"linear": "lin", "quadratic_factor": "qfac", "quadratic_formula": "qform",
          "rational": "rat", "inequality": "ineq"}

# Error items per (type, error type): 24 per type (30% of 80) and 20 per error type (120 / 6),
# with each error type only where it applies (missed_flip needs an inequality, etc.).
PLAN = {
    "linear": {"square_both_sides": 10, "sign_flip": 6, "dropped_term": 4, "wrong_distribution": 4},
    "quadratic_factor": {"divide_by_variable": 20, "sign_flip": 2, "wrong_distribution": 2},
    "quadratic_formula": {"sign_flip": 6, "dropped_term": 10, "wrong_distribution": 8},
    "rational": {"square_both_sides": 10, "sign_flip": 4, "dropped_term": 4, "wrong_distribution": 6},
    "inequality": {"missed_flip": 20, "sign_flip": 2, "dropped_term": 2},
}
PER_TYPE = 80

X = sympy.Symbol("x")
NO_UNIT = [-5, -4, -3, -2, -1, 2, 3, 4, 5, 6]   # b in a = b(x + p): b = 1 reads oddly
FLIP = {"=": "=", "<": ">", ">": "<", r"\leq": r"\geq", r"\geq": r"\leq"}


class Reject(Exception):
    """Parameters gave an ugly or degenerate derivation; resample."""


# ---------- formatting ----------

def num(q):
    q = sympy.Rational(q)
    if q.q == 1:
        return str(q.p)
    return ("-" if q < 0 else "") + rf"\frac{{{abs(q.p)}}}{{{q.q}}}"


def poly(terms):
    """[(coef, power), ...] -> "2x^2 - 5x + 3"; terms kept in the given order, zeros dropped."""
    out = []
    for c, k in terms:
        if c == 0:
            continue
        var = "" if k == 0 else ("x" if k == 1 else f"x^{k}")
        mag = num(abs(c)) if (abs(c) != 1 or k == 0) else ""
        sign = "-" if c < 0 else "+"
        out.append((sign, mag + var))
    if not out:
        return "0"
    first = ("-" if out[0][0] == "-" else "") + out[0][1]
    return " ".join([first] + [f"{s} {t}" for s, t in out[1:]])


def lin(a, b):
    return poly([(a, 1), (b, 0)])


def bracket(a, inner):
    """a(inner) with a = 1 / -1 written as (inner) / -(inner)."""
    return {1: "", -1: "-"}.get(a, str(a)) + f"({inner})"


def frac(a, den):
    """a / den with the sign in front: -\\frac{12}{x + 1}."""
    return ("-" if a < 0 else "") + rf"\frac{{{abs(a)}}}{{{den}}}"


def answer(roots):
    return r" \text{ or } ".join(f"x = {num(r)}" for r in roots)


def factor_line(k, roots):
    """k(x - r1)(x - r2) = 0 with rational roots written as (qx - p)."""
    lead = sympy.Rational(k)
    parts = []
    for r in roots:
        r = sympy.Rational(r)
        lead /= r.q
        parts.append(r.q * X - r.p)
    if lead.q != 1:
        raise Reject
    if roots[0] == 0:   # x(x - r) -> "x \cdot (x - r)"
        body = r"x \cdot " + f"({lin(parts[1].coeff(X), parts[1].subs(X, 0))})"
    else:
        body = "".join(f"({lin(p.coeff(X), p.subs(X, 0))})" for p in parts)
    pre = {1: "", -1: "-"}.get(int(lead), str(lead))
    return f"{pre}{body} = 0"


# ---------- shared tails ----------

def linear_tail(A, C, D, E, rel, err=None):
    """Lines solving Ax + C rel Dx + E; err in {None, "move_x", "move_const", "missed_flip"}."""
    lines, bad = [], None
    if D != 0:
        Ax = A + D if err == "move_x" else A - D
        bad = 0 if err == "move_x" else bad
        lines.append(f"{poly([(Ax, 1), (C, 0)])} {rel} {num(E)}")
    else:
        Ax = A
    if Ax == 0:
        raise Reject
    if C != 0:
        R = E + C if err == "move_const" else E - C
        bad = len(lines) if err == "move_const" else bad
        lines.append(f"{poly([(Ax, 1)])} {rel} {num(R)}")
    else:
        R = E
    if err in ("move_x", "move_const") and bad is None:
        raise Reject
    if Ax != 1:
        flip = Ax < 0 and err != "missed_flip"
        if err == "missed_flip":
            if Ax > 0:
                raise Reject
            bad = len(lines)
        lines.append(f"x {FLIP[rel] if flip else rel} {num(sympy.Rational(R, Ax))}")
    elif err == "missed_flip":
        raise Reject
    return lines, bad


def quad_roots(A, B, C):
    """Two distinct rational roots of Ax^2 + Bx + C, else Reject."""
    disc = B * B - 4 * A * C
    s, exact = sympy.integer_nthroot(disc, 2) if disc > 0 else (0, False)
    if A <= 0 or not exact or s > 20:   # keep \sqrt{D} sayable
        raise Reject
    roots = [sympy.Rational(-B + s, 2 * A), sympy.Rational(-B - s, 2 * A)]
    if any(abs(r.q) > 6 or abs(r) > 12 for r in roots):
        raise Reject
    return roots, s


def quad_tail(A, B, C, route):
    """Lines from Ax^2 + Bx + C = 0 to the roots, via factoring or the formula."""
    roots, s = quad_roots(A, B, C)
    if route == "factor":
        roots.sort(key=lambda r: (r != 0, r))   # a zero root goes first: x \cdot (...)
        return [factor_line(A, roots), answer(roots)]
    top = (lambda sg: rf"{num(-B)} {sg} \sqrt{{{s * s}}}") if B != 0 else \
        (lambda sg: ("" if sg == "+" else "-") + rf"\sqrt{{{s * s}}}")
    form = r" \text{ or } ".join(rf"x = \frac{{{top(sg)}}}{{{2 * A}}}" for sg in "+-")
    return [form, answer(roots)]


def square_tail(k, m, expand=True):
    """(x + k)^2 = m^2 [the error], then correct algebra to two roots."""
    roots = [m - k, -m - k]
    if 0 in roots or len(set(roots)) < 2:
        raise Reject
    sq = f"({lin(1, k)})^2 = {m * m}"
    mid = [f"{poly([(1, 2), (2 * k, 1), (k * k - m * m, 0)])} = 0"] if expand else []
    return [sq] + mid + [factor_line(1, sorted(roots)), answer(sorted(roots))]


def pick_solution(rng, den, lo=-8, hi=8):
    """An integer answer (60%) or a simple fraction with denominator `den`."""
    if rng.random() < 0.6 or abs(den) == 1:
        return sympy.Integer(rng.randint(lo, hi))
    return sympy.Rational(rng.choice([n for n in range(2 * lo, 2 * hi + 1) if n % den]), den)


def nz(rng, lo, hi):
    return rng.choice([v for v in range(lo, hi + 1) if v != 0])


# ---------- types ----------

def gen_linear(rng, err, rel="="):
    """a(x + b) + c rel dx + e; also the inequality type when rel != "="."""
    if err == "square_both_sides":
        a, k, m = rng.randint(2, 6), nz(rng, -6, 6), nz(rng, -6, 6)
        good = [f"{a}({lin(1, k)}) = {a * m}", f"{lin(1, k)} = {m}", f"x = {m - k}"]
        return good[:2] + square_tail(k, m), 2, good
    a, b, c = rng.randint(2, 6), nz(rng, -6, 6), rng.randint(-9, 9)
    d = nz(rng, -4, 5)
    if err == "missed_flip" and a - d >= 0:
        d = a + rng.randint(1, 4)
    if a - d in (0, 1) or (err == "dropped_term" and c == 0):
        raise Reject
    s = pick_solution(rng, a - d)
    e = (a - d) * s + a * b + c
    if e.q != 1 or abs(e) > 40 or a * b + c == 0:
        raise Reject
    lhs = bracket(a, lin(1, b)) + (f" {'+' if c > 0 else '-'} {abs(c)}" if c else "")
    l0 = f"{lhs} {rel} {lin(d, e)}"
    rhs = lin(d, e)
    expand = lambda consts: f"{poly([(a, 1)] + [(v, 0) for v in consts])} {rel} {rhs}"
    good = [l0, expand([a * b, c])] + linear_tail(a, a * b + c, d, e, rel)[0]
    if err is None:
        return good, None, good
    if err == "wrong_distribution":
        return [l0, expand([b, c])] + linear_tail(a, b + c, d, e, rel)[0], 1, good
    if err == "dropped_term":
        return [l0, expand([a * b])] + linear_tail(a, a * b, d, e, rel)[0], 1, good
    tail_err = "missed_flip" if err == "missed_flip" else rng.choice(["move_x", "move_const"])
    tail, bad = linear_tail(a, a * b + c, d, e, rel, tail_err)
    return [l0, good[1]] + tail, 2 + bad, good


def gen_inequality(rng, err):
    return gen_linear(rng, err, rng.choice(["<", ">", r"\leq", r"\geq"]))


def gen_quadratic(rng, err, route):
    """x \\cdot (kx + u) = vx + w, rearranged to standard form, then factor or formula."""
    k = rng.choice([1, 1, 1, 2, 3]) if route == "factor" else rng.randint(2, 6)
    p = 0 if err == "divide_by_variable" else nz(rng, -9, 9)
    r2 = nz(rng, -7, 7)
    u = nz(rng, -6, 6)
    v, w = u + p + k * r2, -p * r2
    if sympy.Rational(p, k) == r2 or (err == "divide_by_variable" and v == 0) or max(abs(v), abs(w)) > 30:
        raise Reject
    l0 = r"x \cdot " + f"({lin(k, u)}) = {lin(v, w)}"
    l1 = lambda lhs, rhs: f"{poly(lhs)} = {lin(*rhs)}"
    std = lambda A, B, C: f"{poly([(A, 2), (B, 1), (C, 0)])} = 0"
    good = [l0, l1([(k, 2), (u, 1)], (v, w)), std(k, u - v, -w)] + quad_tail(k, u - v, -w, route)
    if err is None:
        return good, None, good
    if err == "divide_by_variable":   # x(kx + u - v) = 0 divided by x loses x = 0
        bad = [f"{lin(k, u)} = {v}"]
        rest = [f"{poly([(k, 1)])} = {v - u}"] + ([f"x = {num(sympy.Rational(v - u, k))}"] if k != 1 else [])
        return good[:2] + bad + rest, 2, good
    if err == "wrong_distribution":   # x \cdot u written as u
        return [l0, l1([(k, 2), (u, 0)], (v, w)), std(k, -v, u - w)] + quad_tail(k, -v, u - w, route), 1, good
    if err == "dropped_term":
        if w == 0:
            raise Reject
        return [l0, l1([(k, 2), (u, 1)], (v, 0)), std(k, u - v, 0)] + quad_tail(k, u - v, 0, route), 1, good
    # sign_flip: one moved term keeps its sign
    opts = [o for o, ok in (("v", v != 0), ("w", w != 0)) if ok]
    B, C = (u + v, -w) if rng.choice(opts) == "v" else (u - v, w)
    return good[:2] + [std(k, B, C)] + quad_tail(k, B, C, route), 2, good


def gen_rational(rng, err):
    """a / (x + p) = b, or a / (x + p) = b / (x + q), cleared and solved."""
    if err == "square_both_sides":
        b, p, m = rng.choice(NO_UNIT), nz(rng, -6, 6), nz(rng, -6, 6)
        a = b * m
        good = [f"{frac(a, lin(1, p))} = {b}", f"{a} = {bracket(b, lin(1, p))}",
                f"{lin(1, p)} = {m}", f"x = {m - p}"]
        return good[:3] + square_tail(p, m, expand=False), 3, good
    if rng.random() < 0.5:   # one fraction: a = b(x + p) -> a = bx + K -> bx = a - K
        b, p = rng.choice(NO_UNIT), nz(rng, -6, 6)
        s = pick_solution(rng, b)
        a = b * (s + p)
        if a.q != 1 or a == 0 or abs(a) > 40 or a - b * p == 0:
            raise Reject
        l0, l1 = f"{frac(a, lin(1, p))} = {b}", f"{a} = {bracket(b, lin(1, p))}"

        def tail(K, flip=False):
            R = a + K if flip else a - K
            out = [f"{poly([(b, 1)])} = {R}"]
            sol = sympy.Rational(R, b)
            if sol == -p:
                raise Reject
            return out + ([f"x = {num(sol)}"] if b != 1 else [])
        good = [l0, l1, f"{a} = {lin(b, b * p)}"] + tail(b * p)
        if err is None:
            ok = good
        elif err == "wrong_distribution":
            if abs(b) == 1:
                raise Reject
            ok = [l0, l1, f"{a} = {lin(b, p)}"] + tail(p)
        elif err == "dropped_term":
            ok = [l0, l1, f"{a} = {lin(b, 0)}"] + tail(0)
        else:
            return good[:3] + tail(b * p, flip=True), 3, good
        return ok, (None if err is None else 2), good
    # two fractions: a(x + q) = b(x + p) -> ax + aq = bx + bp -> linear tail
    a, b = rng.randint(1, 6), nz(rng, -5, 6)
    p, q = nz(rng, -6, 6), nz(rng, -6, 6)
    if a == b or p == q:
        raise Reject
    l0 = f"{frac(a, lin(1, p))} = {frac(b, lin(1, q))}"
    l1 = f"{bracket(a, lin(1, q))} = {bracket(b, lin(1, p))}"
    exp = lambda C: f"{poly([(a, 1), (C, 0)])} = {lin(b, b * p)}"
    good = [l0, l1, exp(a * q)] + linear_tail(a, a * q, b, b * p, "=")[0]
    sol = sympy.Rational(b * p - a * q, a - b)
    if sol.q > 6 or abs(sol) > 12:
        raise Reject
    if err is None:
        return good, None, good
    if err == "wrong_distribution":
        if a == 1:
            raise Reject
        return [l0, l1, exp(q)] + linear_tail(a, q, b, b * p, "=")[0], 2, good
    if err == "dropped_term":
        return [l0, l1, exp(0)] + linear_tail(a, 0, b, b * p, "=")[0], 2, good
    tail, bad = linear_tail(a, a * q, b, b * p, "=", "move_const")
    return good[:3] + tail, 3 + bad, good


GEN = {
    "linear": gen_linear,
    "inequality": gen_inequality,
    "quadratic_factor": lambda rng, err: gen_quadratic(rng, err, "factor"),
    "quadratic_formula": lambda rng, err: gen_quadratic(rng, err, "formula"),
    "rational": gen_rational,
}


# ---------- ground truth ----------

def parse_line(latex):
    """parse_graded, asserting a usable result in x alone (no function calls such as x(x - 3))."""
    obj, level = parse_graded(latex)
    if obj is None or obj.atoms(AppliedUndef) or not obj.free_symbols <= {X} | _markers(obj):
        raise ValueError(f"unparseable derivation line: {latex!r}")
    return obj, level


def _markers(obj):
    return {s for s in obj.free_symbols if s.name.startswith("__")}


def solution_set(latex, domain=sympy.S.Reals):
    """Real solution set of one line: a relation, or relations joined by "\\text{ or }"."""
    obj, _ = parse_line(latex)
    rels = obj.args[1:] if isinstance(obj, sympy.Tuple) else [obj]
    if isinstance(obj, sympy.Tuple) and obj.args[0].name != r"__rel__:text:or":
        raise ValueError(f"unsupported line: {latex!r}")
    out = sympy.S.EmptySet
    with time_limit(SYMPY_TIMEOUT_S):
        for r in rels:
            r = r.xreplace({s: X for s in r.free_symbols if s.name == "x"})
            out = out.union(sympy.solveset(r, X, domain))
    return out


def domain_of(latex):
    """Reals minus the poles of the problem statement (the original domain)."""
    obj, _ = parse_line(latex)
    poles = set()
    for d in (sympy.denom(sympy.together(e)) for e in (obj.lhs, obj.rhs) if obj.is_Relational):
        poles |= set(sympy.solveset(d, X, sympy.S.Reals))
    return sympy.S.Reals - sympy.FiniteSet(*poles)


def verify(item):
    """Each step keeps the solution set iff step_valid; `solution` matches the problem."""
    lines, dom = item["lines"], domain_of(item["lines"][0])
    sets = [solution_set(s, dom) for s in lines]
    for i in range(1, len(lines)):
        if (sets[i] == sets[i - 1]) != item["step_valid"][i]:
            return False
    return solution_set(item["solution"], dom) == sets[0]


# ---------- dataset ----------

def make_item(rng, dtype, err, tries=2000):
    for _ in range(tries):
        try:
            lines, bad, good = GEN[dtype](rng, err)
        except Reject:
            continue
        if not 4 <= len(lines) <= 6 or len(set(lines)) < len(lines):
            continue
        item = {"type": dtype, "lines": lines,
                "step_valid": [i != bad for i in range(len(lines))],
                "error": None if err is None else {"type": err, "line": bad},
                "solution": good[-1]}
        try:
            if verify(item):
                return item
        except ValueError:
            continue
    raise RuntimeError(f"could not generate {dtype}/{err}")


def generate(seed=SEED, per_type=PER_TYPE):
    """400 items: 80 per type, 30% with one error, stratified 50/50 calib/test.
    A smaller per_type scales PLAN down (for quick tests)."""
    rng = random.Random(seed)
    items = []
    for dtype in TYPES:
        groups = [(e, n * per_type // PER_TYPE) for e, n in PLAN[dtype].items()]
        groups.append((None, per_type - sum(n for _, n in groups)))
        made = []
        for err, n in groups:
            group = [make_item(rng, dtype, err) for _ in range(n)]
            for j, it in enumerate(group):   # halve each (type, error) stratum
                it["split"] = "calib" if j < n // 2 else "test"
            made += group
        rng.shuffle(made)
        for i, it in enumerate(made):
            items.append({"id": f"{PREFIX[dtype]}-{i:04d}", "split": it["split"], "type": dtype,
                          "lines": it["lines"], "step_valid": it["step_valid"],
                          "error": it["error"], "solution": it["solution"]})
    return items


def summary(items):
    return {"types": Counter(i["type"] for i in items),
            "errors": Counter(i["error"]["type"] for i in items if i["error"]),
            "splits": Counter(i["split"] for i in items),
            "error_rate": sum(bool(i["error"]) for i in items) / len(items)}
