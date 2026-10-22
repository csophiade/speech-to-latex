"""Residual-ratio step checker for algebra derivations (Phase 3.2).

A step from L1 = R1 to L2 = R2 is an equivalence transformation iff g = (L2 - R2) / (L1 - R1) is a
finite nonzero constant; g is estimated at N_CHECK_POINTS seeded random real points. For
inequalities the direction must also flip exactly when g < 0. When g is not constant the step is
judged by comparing real solution sets (solveset over S.Reals, restricted to the domain of the
earlier line and of the problem statement): unchanged -> "caution", changed -> "error". The shape of
g (1/f, f, L + R) explains why the solutions changed.

Unchecked lines: an unparseable line is "unchecked". The next line is then compared against the
last parseable line, but only a "valid" or "caution" verdict is kept (an equivalence across the gap
holds whatever the unreadable line said); an "error" across the gap cannot be pinned to one line,
so that step is "unchecked" too.

Final answers after a non-equivalence (an "error" or "caution" step) are also compared with the
solutions of line 0: the true answer is "valid", and an answer that keeps an extraneous root of the
previous line is an "error".

Budget: one deadline of SYMPY_TIMEOUT_S per step (reading the line plus the check). SIGALRM only
works on the main thread, so threaded callers (Gradio handlers, thread pools) must use
check_derivation_bounded, which runs the check in a worker process with a wall-clock limit.
"""

import math
import multiprocessing
import random
import re
import statistics
import threading
import time

import sympy
from sympy import S
from sympy.calculus.util import continuous_domain
from sympy.core.relational import Relational

from src.config import N_CHECK_POINTS, NUM_TOL, SEED, SYMPY_TIMEOUT_S
from src.calculus import calculus_step, constants, evaluate, has_calculus, parse_bar_line
from src.mathparse import Timeout, implicit_products, parse_graded, time_limit

PM = sympy.Symbol("\\pm")
EMPTY = sympy.Symbol("\\emptyset")
_DIR = {sympy.Gt: 1, sympy.Ge: 1, sympy.Lt: -1, sympy.Le: -1}   # sign of L - R on solutions
_STRICT = {sympy.Gt: True, sympy.Lt: True, sympy.Ge: False, sympy.Le: False}
_SEPS = {",", "or", "and", "text:or", "text:and", "quad", "qquad"}
_SYM = {sympy.Lt: "<", sympy.Le: "≤", sympy.Gt: ">", sympy.Ge: "≥"}
_FLIP = {"<": ">", "≤": "≥", ">": "<", "≥": "≤"}
_NO_SOLUTION = re.compile(r"^[\s$]*(?:\\text\s*\{)?\s*(?:there\s+(?:is|are)\s+)?no\s+(?:real\s+)?solutions?"
                          r"\s*\.?\s*\}?[\s.$]*$", re.I)
MAX_EXP = 10 ** 4   # larger numeric exponents are unreadable: CPython big-int pow ignores SIGALRM


# ---------- reading a line ----------

def _const(e):
    return isinstance(e, sympy.Expr) and not (e.free_symbols - {PM})


def _huge(e):
    """A power with a numeric exponent above MAX_EXP (10^{10^{10}}), which would hang if evaluated."""
    for p in e.atoms(sympy.Pow):
        if p.exp.is_number:
            try:
                if abs(complex(sympy.N(p.exp, 5))) > MAX_EXP:
                    return True
            except (TypeError, ValueError, OverflowError):
                return True
    return False


def _num(e, inf_ok=False):
    """e evaluated, or None if undefined (0/0, 1/0), non-real or (unless inf_ok) infinite."""
    v = e.doit()
    if v.has(S.NaN, S.ComplexInfinity) or v.is_extended_real is False or (not inf_ok and v.is_finite is False):
        return None
    return v


def _values(e):
    """The stated values of a constant expression, expanding ± and evaluating (3/2, \\sqrt{16});
    None if any value is undefined."""
    vals = [_num(v) for v in ((e.subs(PM, 1), e.subs(PM, -1)) if e.has(PM) else (e,))]
    return None if None in vals else set(vals)


def _as_set(obj):
    """A SymPy set from a parsed {..}, interval or \\emptyset, else None."""
    if obj == EMPTY:
        return S.EmptySet
    if not isinstance(obj, sympy.Tuple) or not all(_const(x) for x in obj[1:]):
        return None
    kind, xs = obj[0].name, list(obj[1:])
    if kind == "__set__":
        vals = [_values(x) for x in xs]
        return None if None in vals else sympy.FiniteSet(*[v for vs in vals for v in vs])
    ends = {"__tuple__": (True, True), "__bracket__": (False, False),
            "__interval[)__": (False, True), "__interval(]__": (True, False)}
    if kind in ends and len(xs) == 2:
        lo, hi = (_num(x, inf_ok=True) for x in xs)
        return None if lo is None or hi is None else sympy.Interval(lo, hi, *ends[kind])
    return None


def _compound(obj):
    """(var, interval) for "a < x < b" or "b > x > a", else None."""
    ops = obj[0].name[8:].split(",")
    if len(obj) != 4 or not isinstance(obj[2], sympy.Symbol) or not (_const(obj[1]) and _const(obj[3])):
        return None
    lo, hi = _num(obj[1], inf_ok=True), _num(obj[3], inf_ok=True)
    if lo is None or hi is None:
        return None
    if set(ops) <= {"<", "leq"}:
        return obj[2], sympy.Interval(lo, hi, ops[0] == "<", ops[1] == "<")
    if set(ops) <= {">", "geq"}:
        return obj[2], sympy.Interval(hi, lo, ops[1] == ">", ops[0] == ">")
    return None


def _conjunction(obj):
    """(var, set) for inequalities joined by "and" in one variable ("x > 0 and 1 > 2x"), else None."""
    parts = obj[1:]
    if not parts or not all(type(r) in _DIR for r in parts):
        return None
    syms = set().union(*[r.free_symbols for r in parts])
    if len(syms) != 1:
        return None
    var = syms.pop()
    sets = [_solve(r, var, S.Reals) for r in parts]
    return None if None in sets else (var, sympy.Intersection(*sets))


def _solution(obj):
    """(var, stated set) for "x = 3", "x = 2 or x = 5", "x = a ± b", "x \\in {..}", else None."""
    if isinstance(obj, sympy.Eq) and isinstance(obj.lhs, sympy.Symbol) and _const(obj.rhs):
        vals = _values(obj.rhs)
        return None if vals is None else (obj.lhs, sympy.FiniteSet(*vals))
    if not isinstance(obj, sympy.Tuple) or not isinstance(obj[0], sympy.Symbol):
        return None
    marker = obj[0].name
    if marker == "__rel__:in" and len(obj) == 3 and isinstance(obj[1], sympy.Symbol):
        st = _as_set(obj[2])
        return None if st is None else (obj[1], st)
    if marker == "__list__" or (marker.startswith("__rel__:") and set(marker[8:].split(",")) <= _SEPS):
        parts = [_solution(x) for x in obj[1:]]
        if parts and all(p and p[0] == parts[0][0] for p in parts):
            return parts[0][0], sympy.Union(*[p[1] for p in parts])
    return None


def _same(a, b, syms):
    """a - b vanishes at the sample points (None if it cannot be evaluated)."""
    vals = _ratios(a - b, 1 + sympy.Abs(a) + sympy.Abs(b), syms)
    return None if vals is None else all(abs(v) < NUM_TOL for v in vals)


def _chain(items):
    """A = B = ... = Z: either A = B with B ≡ C ≡ ... (a computation), or A ≡ ... ≡ Y with Y = Z.
    Returns (Eq to check, None) or (None, explanation of the first link that is not an identity)."""
    syms = sorted(set().union(*[x.free_symbols for x in items]), key=str)
    ok = [_same(a, b, syms) for a, b in zip(items, items[1:])]
    if None in ok:
        return None, None   # a link cannot be evaluated (1/0): unreadable, never guessed
    if all(ok[1:]):
        return sympy.Eq(items[0], items[-1], evaluate=False), None
    if all(ok[:-1]):
        return sympy.Eq(items[-2], items[-1], evaluate=False), None
    k = next(i for i, x in enumerate(ok[:-1]) if not x)
    return None, f"{_show(items[k])} and {_show(items[k + 1])} are not equal"


def _left(deadline):
    """Seconds left before `deadline` (at least 0.05)."""
    return max(0.05, deadline - time.monotonic())


def read_line(latex, deadline=None):
    """Parse one line into {"rel": Relational|None, "sol": (var, set)|None, "bad": str|None}, or None.
    "no solution" gives sol (None, EmptySet); the checker fills in the variable."""
    deadline = time.monotonic() + SYMPY_TIMEOUT_S if deadline is None else deadline
    if _NO_SOLUTION.match(latex or ""):
        return {"rel": None, "sol": (None, S.EmptySet), "bad": None}
    obj = parse_bar_line(latex)   # [F]_a^b and F|_a^b: evaluation bars
    if obj is None:
        obj = parse_graded(latex, timeout=math.ceil(_left(deadline) * 2) / 2)[0]   # rounded: keeps the cache useful
    if obj is None:
        return None
    obj = implicit_products(obj)   # x(x - 3) is a product
    if obj.has(S.NaN, S.ComplexInfinity) or _huge(obj) \
            or any(p.base.is_zero and p.exp.is_negative for p in obj.atoms(sympy.Pow)):
        return None   # undefined (1/0, 0/0, kept unevaluated by the parser) or too large to evaluate
    with time_limit(_left(deadline)):
        return _interpret(obj)


def _impossible_and(obj):
    """'x = 2 and x = 3': a variable cannot take two values at once (the speaker meant "or")."""
    if not (isinstance(obj, sympy.Tuple) and isinstance(obj[0], sympy.Symbol)
            and obj[0].name.startswith("__rel__:") and set(obj[0].name[8:].split(",")) <= {"and", "text:and"}):
        return None
    parts = [_solution(x) for x in obj[1:]]
    if len(parts) < 2 or not all(p and p[0] == parts[0][0] and len(p[1]) == 1 for p in parts):
        return None
    vals = sympy.FiniteSet(*[next(iter(p[1])) for p in parts])
    if len(vals) < 2:
        return None
    var, shown = parts[0][0], " and ".join(_show(v) for v in _points(vals))
    return (f"{var} cannot be {shown} at the same time: \"and\" means both must hold. "
            f"Did you mean \"{' or '.join(f'{var} = {_show(v)}' for v in _points(vals))}\"?")


def _interpret(obj):
    bad = _impossible_and(obj)
    if bad:
        return {"rel": None, "sol": None, "bad": bad}
    sol = _solution(obj)
    if sol is not None:
        is_eq = isinstance(obj, sympy.Eq) and not obj.has(PM) and len(sol[1]) == 1
        return {"rel": obj if is_eq else None, "sol": sol, "bad": None}
    if isinstance(obj, Relational) and type(obj) in (sympy.Eq, *_DIR) \
            and all(isinstance(a, sympy.Expr) for a in obj.args) and not obj.has(PM):
        return {"rel": obj, "sol": None, "bad": None}
    if isinstance(obj, sympy.Tuple) and obj[0].name.startswith("__rel__:") \
            and set(obj[0].name[8:].split(",")) == {"="} and all(isinstance(x, sympy.Expr) for x in obj[1:]):
        rel, bad = _chain(list(obj[1:]))
        if rel is None and bad is None:
            return None
        if rel is not None:
            sol = _solution(rel)
            sol = sol if sol and not rel.rhs.has(PM) else None
        return {"rel": rel, "sol": sol, "bad": bad}
    if isinstance(obj, sympy.Tuple) and isinstance(obj[0], sympy.Symbol) \
            and obj[0].name.startswith("__rel__:"):
        ops = set(obj[0].name[8:].split(","))
        sol = _compound(obj) if ops <= {"<", "leq", ">", "geq"} else \
            _conjunction(obj) if ops <= {"and", "text:and"} else None
        if sol is not None:
            return {"rel": None, "sol": sol, "bad": None}
    return None   # sets, systems, \neq: not checked


# ---------- numerics ----------

def _ratios(num, den, syms):
    """num/den at N_CHECK_POINTS seeded real points; undefined points and den ≈ 0 are resampled."""
    rng = random.Random(SEED)
    f = sympy.lambdify(syms, [num, den], "math", dummify=True)
    out, tries = [], 0
    while len(out) < N_CHECK_POINTS and tries < 50 * N_CHECK_POINTS:
        tries += 1
        try:
            a, b = (float(v) for v in f(*[rng.uniform(-10, 10) for _ in syms]))
        except (ValueError, ZeroDivisionError, OverflowError, TypeError):
            continue   # outside the domain (sqrt of a negative, a pole)
        if abs(b) > 1e-9 and abs(a) < 1e12 and abs(b) < 1e12:
            out.append(a / b)
    return out if len(out) == N_CHECK_POINTS else None


def _constant(num, den, syms):
    """The constant value of num/den if it is a finite nonzero constant, else None."""
    vals = _ratios(num, den, syms)
    if vals is None:
        return None
    mid = statistics.median(vals)
    if abs(mid) < 1e-12 or (max(vals) - min(vals)) / abs(mid) >= NUM_TOL:
        return None
    return mid


def _res(rel):
    return rel.lhs - rel.rhs


# ---------- solution sets ----------

def domain(rel, var):
    """Real domain of a line (where both sides are defined)."""
    try:
        return continuous_domain(_res(rel), var, S.Reals)
    except Exception:
        return S.Reals


def _solve(rel, var, dom):
    try:
        s = sympy.solveset(rel, var, S.Reals).intersect(dom)
    except Exception:
        return None
    return s if isinstance(s, (sympy.FiniteSet, sympy.Interval, sympy.Union)) or s is S.EmptySet else None


def _show(e):
    e = e.xreplace({f: sympy.nsimplify(f, rational=True, tolerance=NUM_TOL) for f in e.atoms(sympy.Float)})
    return sympy.sstr(e).replace("**", "^").replace("*", "")


def _fmt(var, s):
    """A solution set in words; sets with no simple form (sin x = 0) get a generic phrase."""
    if s is S.EmptySet:
        return "no solution"
    if isinstance(s, sympy.FiniteSet) and all(v.is_real for v in s):
        return " or ".join(f"{var} = {_show(v)}" for v in sorted(s, key=lambda v: float(v)))
    if s == S.Reals:
        return "every real number"
    if isinstance(s, sympy.Interval):
        lo, hi = ("<" if s.left_open else "≤"), ("<" if s.right_open else "≤")
        if s.end == S.Infinity:
            return f"{var} {'>' if s.left_open else '≥'} {_show(s.start)}"
        if s.start == S.NegativeInfinity:
            return f"{var} {hi} {_show(s.end)}"
        return f"{_show(s.start)} {lo} {var} {hi} {_show(s.end)}"
    if isinstance(s, sympy.Union) and all(isinstance(a, (sympy.Interval, sympy.FiniteSet)) for a in s.args):
        return " or ".join(_fmt(var, a) for a in s.args)
    return "infinitely many solutions" if s.has(sympy.ImageSet) else "a set of values with no simple form"


def _points(s):
    return sorted(s, key=lambda v: float(v)) if isinstance(s, sympy.FiniteSet) else []


# ---------- explanations for a non-constant g ----------

def _terms(rel):
    return [t for side in (rel.lhs, rel.rhs) for t in sympy.Add.make_args(side)]


def _expansion(p, c, syms):
    """(bracket, dropped) for a bracket k(a + b) of p that was expanded wrongly: replacing one wrong
    term v of the expansion (u, -u or -k u instead of k u) by k u repairs the step (dropped None),
    or putting a missing k u back does (dropped = k u). None if no such bracket."""
    rp, rc = _res(p), _res(c)
    cur = set(sympy.preorder_traversal(rc))
    for side, sign in ((p.lhs, 1), (p.rhs, -1)):
        for m in sympy.preorder_traversal(side):
            if not isinstance(m, sympy.Mul) or m in cur:
                continue   # not a product, or not expanded in this step
            adds = [a for a in m.args if isinstance(a, sympy.Add)]
            if len(adds) != 1:
                continue
            k = m / adds[0]
            for u in adds[0].args:
                if any(_constant(rc + sign * (k * u - v), rp, syms) is not None for v in (u, -u, -k * u)):
                    return m, None
                if _constant(rc + sign * k * u, rp, syms) is not None:
                    return m, k * u
    return None


def _hint(p, c, syms):
    """Name the likely slip for a step that changes the solutions without a variable factor.
    A term is blamed only if it is structurally gone from its side (a numeric fix alone can match
    any constant of the right size), and a wrongly expanded bracket is tested first."""
    rp = _res(p)
    slip = _expansion(p, c, syms)
    if slip and slip[1] is None:
        return f"check the expansion of {_show(slip[0])}: multiply every term inside the brackets"
    if slip:
        return f"the term {_show(slip[1])} seems to have been dropped when expanding {_show(slip[0])}"
    terms = {side: set(sympy.Add.make_args(side)) for side in (p.lhs, p.rhs, c.lhs, c.rhs)}
    for side, other, sign, before in ((c.lhs, c.rhs, 1, p.rhs), (c.rhs, c.lhs, -1, p.lhs)):
        for t in sympy.Add.make_args(side):   # moved to this side with its sign unchanged
            if not t.is_zero and t in terms[before] and _constant(sign * (side - 2 * t - other), rp, syms) is not None:
                return f"check the sign of {_show(t)}: moving a term to the other side changes its sign"
    for t, same in [(t, c.lhs) for t in terms[p.lhs]] + [(t, c.rhs) for t in terms[p.rhs]]:
        if t in terms[same]:
            continue   # still where it was
        if any(_constant(_res(c) + 2 * s * t, rp, syms) is not None for s in (1, -1)):
            return f"check the sign of {_show(t)}: moving a term to the other side changes its sign"
    gone = [t for t, same in [(t, c.lhs) for t in terms[p.lhs]] + [(t, c.rhs) for t in terms[p.rhs]]
            if t not in terms[same]]   # no longer on its own side
    for t in gone:
        if any(_constant(_res(c) + s * t, rp, syms) is not None for s in (1, -1)):
            return f"the term {_show(t)} seems to have been dropped"
    return None


def _operation(p, c, syms):
    """'subtracting 50 from both sides gives x = 40, not x = 30': one side of the step is a clean
    add/subtract or divide/multiply by a constant, and the other side got the wrong result."""
    for (pa, ca), (pb, cb) in (((p.lhs, c.lhs), (p.rhs, c.rhs)), ((p.rhs, c.rhs), (p.lhs, c.lhs))):
        d = sympy.simplify(pa - ca)
        moved = d in sympy.Add.make_args(pa) and set(sympy.Add.make_args(ca)) == set(sympy.Add.make_args(pa)) - {d}
        if d != 0 and d.is_number and moved:   # a term literally taken off this side
            want = sympy.simplify(pb - d)
            if not _same(want, cb, syms):
                verb = f"subtracting {_show(d)} from" if d > 0 else f"adding {_show(-d)} to"
                return f"{verb} both sides gives {_show(ca)} = {_show(want)}, not {_show(ca)} = {_show(cb)}" \
                    if pa is p.lhs else f"{verb} both sides gives {_show(want)} = {_show(ca)}, not {_show(cb)} = {_show(ca)}"
        k = sympy.simplify(pa / ca) if ca != 0 else None
        coeff = isinstance(pa, sympy.Mul) and k in pa.args and sympy.Mul(*[a for a in pa.args if a != k]) == ca
        if k is not None and k.is_number and k not in (0, 1) and coeff:   # this side literally divided by k
            want = sympy.simplify(pb / k)
            if not _same(want, cb, syms):
                verb = f"dividing both sides by {_show(k)}" if abs(k) > 1 or not k.is_rational \
                    else f"multiplying both sides by {_show(1 / k)}"
                return f"{verb} gives {_show(ca)} = {_show(want)}, not {_show(ca)} = {_show(cb)}" \
                    if pa is p.lhs else f"{verb} gives {_show(want)} = {_show(ca)}, not {_show(cb)} = {_show(ca)}"
    return None


def _explain(p, c, var, sp, sc, syms, ineq):
    """(status, explanation) for a step whose ratio g is not constant."""
    rp, rc = _res(p), _res(c)
    lost, added = _points(sp - sc), _points(sc - sp)
    changed = sp != sc
    if not ineq and _constant(rc, rp * (p.lhs + p.rhs), syms) is not None:
        if added:
            return "error", (f"squaring both sides can add solutions where L = -R: "
                             f"{var} = {_show(added[0])} solves the new line but not the previous one")
        return ("error", "squaring both sides changed the solutions") if changed else \
            ("caution", "squaring both sides can add solutions where L = -R; none appear here")
    f = sympy.cancel(sympy.together(rp / rc))   # dividing both sides by f
    if f.is_polynomial(var) and sympy.degree(f, var) > 0:
        fs = _show(sympy.factor(f))
        if lost:
            return "error", f"dividing by {fs} may lose the solution {var} = {_show(lost[0])}"
        if not changed:
            return "caution", f"dividing by {fs} is only safe because {fs} ≠ 0 at the solutions"
    f = sympy.cancel(sympy.together(rc / rp))   # multiplying both sides by f
    if f.is_polynomial(var) and sympy.degree(f, var) > 0:
        fs = _show(sympy.factor(f))
        if ineq and changed:
            return "error", (f"multiplying an inequality by {fs} needs its sign: "
                             "where it is negative the inequality must flip")
        if added:
            return "error", f"multiplying by {fs} adds the solution {var} = {_show(added[0])}, where {fs} = 0"
        if not changed:
            zeros = _points(_solve(sympy.Eq(f, 0), var, S.Reals) or S.EmptySet)
            note = " and ".join(f"{var} ≠ {_show(z)}" for z in zeros)
            note = f" (so {note})" if zeros and f != var else ""
            return "caution", f"multiplying by {fs} is only allowed because {fs} ≠ 0 here{note}"
    if not changed:
        return "caution", "this step is not a simple equivalence, but the solutions are unchanged"
    op, hint = _operation(p, c, syms), _hint(p, c, syms)
    if op:
        return "error", f"{op} ({hint})" if hint else op
    moved = f"the solutions change from {_fmt(var, sp)} to {_fmt(var, sc)}"
    return "error", f"{hint}; {moved}" if hint else moved


# ---------- the step check ----------

def _out(status, g=None, why=None):
    return {"status": status, "g": g, "explanation": why}


def _compare_sets(var, before, stated, dom):
    """A stated solution set against the solutions of the previous line."""
    outside = _points(stated - dom)
    if outside:
        return _out("error", why=f"{var} = {_show(outside[0])} is outside the domain of the problem "
                                 "(a denominator is zero or a root is undefined there)")
    if before == stated:
        return _out("valid")
    missing, extra = _points(before - stated), _points(stated - before)
    if extra:
        return _out("error", why=f"{var} = {_show(extra[0])} does not satisfy the previous line")
    if missing:
        return _out("error", why=f"the solution {var} = {_show(missing[0])} is missing")
    return _out("error", why=f"the previous line gives {_fmt(var, before)}, not {_fmt(var, stated)}")


def _swapped(pr, cr, syms):
    """cr is pr with its two sides exchanged (7 < 2x + 3 from 2x + 3 < 7)."""
    return bool(_same(cr.lhs, pr.rhs, syms)) and bool(_same(cr.rhs, pr.lhs, syms))


def _extraneous(var, stated, before, orig):
    """A final answer after a non-equivalence, judged against the original problem (orig = (true
    solutions, cause, "equation"|"inequality")). None if the original solutions do not decide it."""
    true, cause, kind = orig
    if stated == true:
        return _out("valid")
    extra = _points(stated - true)
    if extra and true.is_subset(stated) and before is not None and stated.is_subset(before):
        return _out("error", why=f"{var} = {_show(extra[0])} does not satisfy the original {kind} "
                                 f"(an extraneous root from {cause})")
    return None


def _calculus(p, c):
    """Calculus steps: side-by-side verdict, or both lines with their operators evaluated (and new
    integration constants set to 0) for the algebra check below. Returns (verdict | None, p, c)."""
    pr, cr = p["rel"], c["rel"]
    res = calculus_step(pr, cr)
    if res is not None:
        return _out(res["status"], why=res["explanation"]), p, c
    if cr is not None and pr is not None and pr.atoms(sympy.Integral):
        cr = cr.subs({k: 0 for k in constants(cr, pr.free_symbols)})
    ev = [type(r)(evaluate(r.lhs), evaluate(r.rhs), evaluate=False) if r is not None else None for r in (pr, cr)]
    if all(r is not None and not r.free_symbols for r in ev):   # numbers only: compare the values
        vp, vc = (complex(sympy.N(_res(r))) for r in ev)
        if abs(vp - vc) < NUM_TOL * max(1.0, abs(vp)):
            return _out("valid"), p, c
        return _out("error", why=f"this changes the value: left minus right was {vp.real:.4g} and is now {vc.real:.4g}"), p, c
    new = [_interpret(r) if r is not None else x for r, x in zip(ev, (p, c))]
    if None in new:
        return _out("unchecked", why="could not evaluate the calculus in this step"), p, c
    return None, new[0], new[1]


def _check(p, c, base=S.Reals, orig=None):
    if c["bad"]:
        return _out("error", why=c["bad"])
    if has_calculus(p["rel"]) or has_calculus(c["rel"]):
        verdict, p, c = _calculus(p, c)
        if verdict is not None:
            return verdict
    pr, cr = p["rel"], c["rel"]
    syms = sorted(set().union(*[x.free_symbols for x in (pr, cr) if x is not None],
                              *[{x["sol"][0]} for x in (p, c) if x["sol"] and x["sol"][0] is not None]), key=str)
    var = syms[0] if len(syms) == 1 else None
    if var is not None:   # "no solution" lines take the variable of the other line
        p, c = ({**x, "sol": (var, x["sol"][1])} if x["sol"] and x["sol"][0] is None else x for x in (p, c))
    dom = base if var is None or pr is None else base.intersect(domain(pr, var))

    if pr is not None and cr is not None and type(pr) is type(cr) is sympy.Eq \
            and _same(pr.lhs, pr.rhs, syms) and _same(cr.lhs, cr.rhs, syms):
        return _out("valid", why="both sides are always equal: every real number is a solution")
    if pr is not None and cr is not None and (type(pr) is sympy.Eq) == (type(cr) is sympy.Eq):
        ineq = type(pr) is not sympy.Eq
        g = _constant(_res(cr), _res(pr), syms)
        if g is not None:
            gs = _show(sympy.nsimplify(g, rational=True, tolerance=NUM_TOL))
            if ineq and _DIR[type(pr)] * _DIR[type(cr)] * g < 0:
                why = ("multiplying or dividing by a negative number must flip the inequality" if g < 0
                       else "the inequality was flipped without multiplying by a negative number")
                if g < 0 and _swapped(pr, cr, syms):
                    op = _SYM[type(cr)]
                    why = (f"swapping the two sides reverses the inequality sign ({_show(cr.lhs)} {op} "
                           f"{_show(cr.rhs)} means {_show(cr.rhs)} {_FLIP[op]} {_show(cr.lhs)})")
                return _out("error", gs, why)
            if ineq and _STRICT[type(pr)] != _STRICT[type(cr)]:
                return _out("error", gs, "changing between < and ≤ changes whether the boundary is a solution")
            if c["sol"] and var is not None:
                bad = _points(c["sol"][1] - dom)
                if bad:
                    return _out("error", gs, f"{var} = {_show(bad[0])} is outside the domain of the problem "
                                             "(a denominator is zero or a root is undefined there)")
            return _out("valid", gs)
    if var is None:
        return _out("unchecked", why="could not compare solutions (several variables or none)")

    before = p["sol"][1].intersect(dom) if pr is None else _solve(pr, var, dom)
    if c["sol"] is not None and (cr is None or c["sol"][0] == var):
        if orig is not None:
            r = _extraneous(var, c["sol"][1], before, orig)
            if r is not None:
                return r
        if before is None:
            return _out("unchecked", why="could not solve the previous line")
        if cr is None or pr is None:
            return _compare_sets(var, before, c["sol"][1], dom)
    after = c["sol"][1].intersect(dom) if cr is None else _solve(cr, var, dom)
    if before is None or after is None:
        return _out("unchecked", why="could not solve the lines to compare their solutions")
    if pr is None or cr is None or (type(pr) is sympy.Eq) != (type(cr) is sympy.Eq):
        return _out("valid") if before == after else \
            _out("error", why=f"the solutions change from {_fmt(var, before)} to {_fmt(var, after)}")
    if c["sol"] is not None and _points(c["sol"][1] - dom):
        return _compare_sets(var, before, c["sol"][1], dom)   # an answer outside the domain
    g = sympy.cancel(sympy.together(_res(cr) / _res(pr)))
    g = None if g.has(S.NaN, S.ComplexInfinity) else _show(g)
    status, why = _explain(pr, cr, var, before, after, syms, type(pr) is not sympy.Eq)
    if c["sol"] is not None and status == "caution":
        return _out("valid")   # a solution statement that matches the solutions (g is not constant)
    return _out(status, g, why)


def _safe_check(p, c, base=S.Reals, orig=None, deadline=None):
    deadline = time.monotonic() + SYMPY_TIMEOUT_S if deadline is None else deadline
    try:
        with time_limit(_left(deadline)):
            return _check(p, c, base, orig)
    except Timeout:
        return _out("unchecked", why="the check took too long")
    except Exception as e:   # SymPy internals: never guess
        return _out("unchecked", why=f"could not check this step ({type(e).__name__})")


def _read(latex, deadline=None):
    """read_line, with None for anything that fails (never guess)."""
    try:
        return read_line(latex, deadline)
    except (Timeout, Exception):
        return None


def check_step(prev_latex, cur_latex):
    """{"status": "valid"|"caution"|"error"|"unchecked", "g": str|None, "explanation": str|None}.
    One SYMPY_TIMEOUT_S deadline covers reading both lines and the check (main thread only)."""
    deadline = time.monotonic() + SYMPY_TIMEOUT_S
    p, c = _read(prev_latex, deadline), _read(cur_latex, deadline)
    if p is None or c is None or p["rel"] is None and p["sol"] is None:
        return _out("unchecked", why="this line could not be read" if c is None else "the previous line could not be read")
    return _safe_check(p, c, deadline=deadline)


def _problem(first):
    """(domain, solutions or None) of the problem statement (line 0); final answers must lie in the domain."""
    rel = first["rel"] if first else None
    syms = rel.free_symbols if rel is not None else set()
    if len(syms) != 1:
        return S.Reals, None
    var = next(iter(syms))
    try:
        with time_limit(SYMPY_TIMEOUT_S):
            dom = domain(rel, var)
            return dom, _solve(rel, var, dom)
    except Timeout:
        return S.Reals, None


def check_derivation(lines):
    """Per-line [{"line": i, "status": ..., "explanation": ...}]; line 0 is "given".
    Main thread only (SIGALRM); from other threads use check_derivation_bounded."""
    if not lines:
        return []
    first = _read(lines[0])
    first = first if first and (first["rel"] is not None or first["sol"] is not None) and not first["bad"] else None
    base, true = _problem(first)
    kind = "equation" if first and type(first["rel"]) is sympy.Eq else "inequality"
    out = [{"line": 0, "status": "given", "explanation": None}]
    prev, last, gap, cause = first, 0, False, None   # cause: the first non-equivalence step, if any
    for i in range(1, len(lines)):
        deadline = time.monotonic() + SYMPY_TIMEOUT_S   # reading the line and checking the step
        c = _read(lines[i], deadline)
        if c is None or (c["rel"] is None and c["sol"] is None and not c["bad"]):
            out.append({"line": i, "status": "unchecked", "explanation": "this line could not be read"})
            gap = True
            continue
        if prev is None:
            r = _out("unchecked", why="no earlier line could be read to compare with")
        else:
            orig = (true, cause, kind) if cause and true is not None else None
            r = _safe_check(prev, c, base, orig, deadline)
            if gap and r["status"] == "error":
                r = _out("unchecked", why=f"this line does not follow from line {last}, "
                                          "but the unreadable line in between may hold the mistake")
        if r["status"] in ("error", "caution"):   # squaring is the usual source of extraneous roots
            cause = "squaring both sides" if "squaring" in (r["explanation"] or "") else cause or "an earlier step"
        out.append({"line": i, "status": r["status"], "explanation": r["explanation"]})
        if not c["bad"]:
            prev, last, gap = c, i, False
    return out


# ---------- a wall-clock bound from any thread ----------

_POOL, _POOL_LOCK = None, threading.Lock()


def _ping():
    return True


def check_derivation_bounded(lines, timeout=None):
    """check_derivation with a wall-clock limit that also holds off the main thread (Gradio handlers,
    thread pools): it runs in a worker process, where SIGALRM works; on expiry the worker is killed
    and every line after line 0 is "unchecked". Calls are serialised (one worker). The worker is
    spawned, so a calling script needs the usual `if __name__ == "__main__":` guard."""
    global _POOL
    lines = list(lines)
    timeout = SYMPY_TIMEOUT_S * (len(lines) + 1) if timeout is None else timeout
    with _POOL_LOCK:
        if _POOL is None:
            _POOL = multiprocessing.get_context("spawn").Pool(1)
            _POOL.apply(_ping)   # wait for the worker's imports outside the limit
        try:
            return _POOL.apply_async(check_derivation, (lines,)).get(timeout)
        except multiprocessing.TimeoutError:
            _POOL.terminate()
            _POOL = None
    return [{"line": 0, "status": "given", "explanation": None}][:len(lines)] + \
        [{"line": i, "status": "unchecked", "explanation": "the check took too long"} for i in range(1, len(lines))]
