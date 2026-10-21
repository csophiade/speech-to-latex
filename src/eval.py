"""Metrics shared by every experiment: CER, NEM, SymEq, latency, results logging."""

import json
import random
import re
import subprocess
import time
from contextlib import contextmanager
from datetime import datetime, timezone

import jiwer
import sympy
from sympy.core.relational import Relational

from src import config
from src.mathparse import Timeout, parse_graded, time_limit, try_parse

# ---------- normalisation ----------

_TOKEN = re.compile(r"\\[A-Za-z]+|\\.|\S")
_DROP = {r"\left", r"\right", r"\bigl", r"\bigr", r"\Bigl", r"\Bigr", r"\big", r"\Big",
         r"\bigg", r"\Bigg", r"\,", r"\;", r"\:", r"\!", r"\quad", r"\qquad", r"\ ",
         r"\displaystyle", "~", "$"}
_ALIAS = {r"\dfrac": r"\frac", r"\tfrac": r"\frac", r"\le": r"\leq", r"\ge": r"\geq",
          r"\ne": r"\neq", r"\rightarrow": r"\to", r"\lbrace": r"\{", r"\rbrace": r"\}",
          r"\cdots": r"\dots", r"\ldots": r"\dots"}


def _tokens(latex):
    s = re.sub(r"\\\s+([{}])", r"\\\1", latex or "")   # Speech2Latex writes "\ {" for "\{"
    toks = [_ALIAS.get(t, t) for t in _TOKEN.findall(s)]
    return [t for t in toks if t not in _DROP]


def _unbrace(toks):
    """Parse into nested groups and drop braces around groups holding a single token."""
    def parse(i):
        out = []
        while i < len(toks):
            t = toks[i]
            if t == "{":
                group, i = parse(i + 1)
                out.append(group)
            elif t == "}":
                return out, i + 1
            else:
                out.append(t)
                i += 1
        return out, i

    def flat(node):
        if isinstance(node, str):
            return node
        while len(node) == 1 and isinstance(node[0], list):
            node = node[0]
        inner = "".join(flat(n) for n in node)
        return inner if len(node) == 1 else "{" + inner + "}"

    tree, _ = parse(0)
    return "".join(flat(n) for n in tree)


def normalise(latex):
    s = _unbrace(_tokens(latex))
    return s.rstrip(".,;")


# ---------- metrics ----------

def cer(refs, hyps):
    """Corpus-level character error rate on normalised LaTeX."""
    refs = [normalise(r) or " " for r in refs]
    hyps = [normalise(h) for h in hyps]
    return jiwer.cer(refs, hyps)


def nem(ref, hyp):
    return normalise(ref) == normalise(hyp)


def _side_pairs(a, b):
    """Ways to match the two sides of relations a and b: [[(x1, y1), (x2, y2)], ...]; None if incomparable."""
    if type(a) is type(b):
        same = [[(a.lhs, b.lhs), (a.rhs, b.rhs)]]
        # equations may also be written with sides swapped
        return same + [[(a.lhs, b.rhs), (a.rhs, b.lhs)]] if isinstance(a, (sympy.Eq, sympy.Ne)) else same
    if a.reversed.func is b.func:   # x < 1  vs  1 > x
        return [[(a.lhs, b.rhs), (a.rhs, b.lhs)]]
    return None


def _comparisons(a, b):
    """(x, y) pairs such that a and b are the same maths iff x == y for some pair; None if incomparable.

    Only for relations whose difference lhs - rhs is not constant (see _equal): every true identity
    has difference 0, so differences must never be compared for identities or numeric facts.
    """
    if isinstance(a, Relational) != isinstance(b, Relational):
        return None
    if not isinstance(a, Relational):
        return [(a, b)]
    if _side_pairs(a, b) is None:
        return None
    da, db = a.lhs - a.rhs, b.lhs - b.rhs
    if type(a) is type(b):
        return [(da, db), (da, -db)] if isinstance(a, (sympy.Eq, sympy.Ne)) else [(da, db)]
    return [(da, -db)]


_CALCULUS = (sympy.Limit, sympy.Integral, sympy.Derivative, sympy.Sum, sympy.Product)


def _functionise(obj):
    """Derivative(y, x) with y a plain Symbol would evaluate to 0: make y an unknown function y(x, ...).

    Every Symbol that is differentiated but is not itself a differentiation variable becomes
    Function(name)(v1, v2, ...) over all differentiation variables of the object (sorted), so both
    sides of a comparison get the same functions.
    """
    derivs = [d for d in sympy.preorder_traversal(obj) if isinstance(d, sympy.Derivative)]
    if not derivs:
        return obj
    dvars = sorted({v for d in derivs for v in d.variables if isinstance(v, sympy.Symbol)}, key=str)
    if not dvars:
        return obj
    reps = {}
    for d in derivs:
        own = set(d.variables)
        for sym in d.expr.free_symbols:
            if isinstance(sym, sympy.Symbol) and sym not in own and sym not in dvars:
                reps[sym] = sympy.Function(sym.name)(*dvars)
    if not reps:
        return obj
    return obj.replace(lambda e: isinstance(e, sympy.Derivative),
                       lambda e: sympy.Derivative(e.expr.xreplace(reps), *e.variable_count))


def _constant(d, rng):
    """True if d (a difference lhs - rhs) is constant, i.e. the relation is an identity/numeric fact."""
    if not d.free_symbols:
        return True
    probe = _numeric_equal(d, d.xreplace({s: sympy.Float(1.2345 + k) for k, s in enumerate(
        sorted(d.free_symbols, key=str))}), rng)
    if probe is not None:
        return probe
    return not sympy.simplify(d, doit=False).free_symbols


def _numeric_equal(x, y, rng, n_points=config.N_CHECK_POINTS, tol=config.NUM_TOL):
    """True/False if x == y at random points (relative tol); None if they cannot be evaluated."""
    both = x - y
    if both.atoms(sympy.core.function.AppliedUndef) or both.has(sympy.Limit, sympy.Integral, sympy.Derivative):
        return None
    syms = sorted(both.free_symbols, key=str)
    good = 0
    for _ in range(n_points * 3):
        vals = {s: rng.choice([-1, 1]) * rng.uniform(0.5, 3.0) for s in syms}
        try:
            vx, vy = complex(x.evalf(subs=vals)), complex(y.evalf(subs=vals))
        except Timeout:
            raise   # never swallow the alarm: later work would run unbounded
        except Exception:
            continue
        if any(v != v or abs(v) == float("inf") for v in (vx, vy)):
            continue   # pole or undefined point: resample
        if abs(vx - vy) > tol * max(1.0, abs(vx), abs(vy)):
            return False
        good += 1
        if good >= n_points:
            return True
    return None


def _is_struct(e):
    """A marker Tuple from mathparse (relation chain, tuple, set, ...)."""
    return (isinstance(e, sympy.Tuple) and len(e.args) > 0 and isinstance(e.args[0], sympy.Symbol)
            and e.args[0].name.startswith("__"))


def _equal(a, b, rng):
    """Same maths? Marker Tuples compare elementwise (sets as multisets, nothing else reordered)."""
    if _is_struct(a) or _is_struct(b):
        if not (_is_struct(a) and _is_struct(b)):
            return False
        if a.args[0].name != b.args[0].name:
            return False
        if len(a.args) != len(b.args) and a.args[0].name != "__set__":
            return False
        xs, ys = a.args[1:], b.args[1:]
        if a.args[0].name == "__set__":   # sets: same elements, order and repetition ignored
            return (all(any(_equal(x, y, rng) for y in ys) for x in xs)
                    and all(any(_equal(x, y, rng) for x in xs) for y in ys))
        return all(_equal(x, y, rng) for x, y in zip(xs, ys))
    if isinstance(a, Relational) and isinstance(b, Relational):
        sides = _side_pairs(a, b)
        if sides is None:
            return False
        if any(all(_same(x, y, rng) for x, y in way) for way in sides):
            return True
        # identities (x^2-1 = (x-1)(x+1)) and numeric facts (2+3 = 5) have a constant difference,
        # so all of them would look alike: those are compared side by side only
        if _constant(a.lhs - a.rhs, rng) or _constant(b.lhs - b.rhs, rng):
            return False
    pairs = _comparisons(a, b)
    if pairs is None:
        return False
    return any(_same(x, y, rng) for x, y in pairs)


def _same(x, y, rng):
    """Two expressions (not relations) equal? Calculus objects are never evaluated (doit)."""
    if x == y:
        return True
    verdict = _numeric_equal(x, y, rng)
    if verdict is None:
        diff = x - y
        if diff.has(*_CALCULUS):
            verdict = sympy.simplify(diff, doit=False) == 0
        else:
            verdict = sympy.simplify(diff) == 0
    return bool(verdict)


def symeq(ref, hyp, timeout=config.SYMPY_TIMEOUT_S):
    """'equal' | 'not_equal' | 'unparseable' | 'timeout'. Parses at any level (clean or cleaned)."""
    try:
        if nem(ref, hyp):
            return "equal"
    except RecursionError:   # deeply nested braces: normalise() recurses
        pass
    a, b = try_parse(ref), try_parse(hyp)   # outside time_limit: parsing has its own
    if a is None or b is None:
        return "unparseable"
    try:
        with time_limit(timeout):
            a, b = _functionise(a), _functionise(b)
            return "equal" if _equal(a, b, random.Random(config.SEED)) else "not_equal"
    except Timeout:
        return "timeout"
    except Exception:
        return "not_equal"


# ---------- latency ----------

class _Timer:
    seconds = 0.0


@contextmanager
def latency():
    t = _Timer()
    start = time.perf_counter()
    try:
        yield t
    finally:
        t.seconds = time.perf_counter() - start


# ---------- reporting ----------

def _nem_safe(ref, hyp):
    try:
        return nem(ref, hyp)
    except RecursionError:
        return False


def symeq_report(refs, verdicts):
    """SymEq on the subset whose reference parses (any level), plus that subset's coverage."""
    ref_ok = [parse_graded(r)[1] != "none" for r in refs]
    k = sum(ref_ok)
    hits = sum(ok and v == "equal" for ok, v in zip(ref_ok, verdicts))
    return {"symeq_coverage": round(k / len(refs), 4) if refs else 0.0,
            "symeq_on_parseable": round(hits / k, 4) if k else 0.0}


def summarise(refs, hyps, latencies):
    """Standard metric block for one system on one dataset.

    "symeq" stays equal/n (comparable with old rows); "symeq_on_parseable" is the reported SymEq
    from Phase 2 on, with "symeq_coverage" the fraction of references that parse.
    """
    verdicts = [symeq(r, h) for r, h in zip(refs, hyps)]
    n = len(refs)
    lat = sorted(latencies)
    return {
        "n": n,
        "cer": round(cer(refs, hyps), 4),
        "nem": round(sum(_nem_safe(r, h) for r, h in zip(refs, hyps)) / n, 4),
        "symeq": round(verdicts.count("equal") / n, 4),
        "symeq_unparseable": round(verdicts.count("unparseable") / n, 4),
        "symeq_timeout": verdicts.count("timeout"),
        **symeq_report(refs, verdicts),
        "latency_median_s": round(lat[n // 2] if n % 2 else (lat[n // 2 - 1] + lat[n // 2]) / 2, 3),
    }, verdicts


def git_hash():
    try:
        return subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], cwd=config.ROOT,
                                       text=True, stderr=subprocess.DEVNULL).strip()
    except Exception:
        return "unknown"


def append_result(system, dataset, metrics, **extra):
    config.RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    path = config.RESULTS_JSON
    rows = json.loads(path.read_text()) if path.exists() else []
    rows.append({"system": system, "dataset": dataset, "metrics": metrics, "n": metrics.get("n"),
                 "timestamp": datetime.now(timezone.utc).isoformat(timespec="seconds"),
                 "git": git_hash(), **extra})
    path.write_text(json.dumps(rows, indent=2, ensure_ascii=False) + "\n")
