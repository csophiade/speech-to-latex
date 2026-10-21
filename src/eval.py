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
from src.mathparse import Timeout, time_limit, try_parse

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


def _comparisons(a, b):
    """(x, y) pairs such that a and b are the same maths iff x == y for some pair; None if incomparable."""
    if isinstance(a, Relational) != isinstance(b, Relational):
        return None
    if not isinstance(a, Relational):
        return [(a, b)]
    da, db = a.lhs - a.rhs, b.lhs - b.rhs
    if type(a) is type(b):
        # equations may also be written with sides swapped
        return [(da, db), (da, -db)] if isinstance(a, sympy.Eq) else [(da, db)]
    if a.reversed.func is b.func:   # x < 1  vs  1 > x
        return [(da, -db)]
    return None


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


def symeq(ref, hyp, timeout=config.SYMPY_TIMEOUT_S):
    """'equal' | 'not_equal' | 'unparseable' | 'timeout'."""
    if nem(ref, hyp):
        return "equal"
    a, b = try_parse(ref), try_parse(hyp)
    if a is None or b is None:
        return "unparseable"
    try:
        with time_limit(timeout):
            pairs = _comparisons(a, b)
            if pairs is None:
                return "not_equal"
            rng = random.Random(config.SEED)
            for x, y in pairs:
                verdict = _numeric_equal(x, y, rng)
                if verdict is None:
                    verdict = sympy.simplify(x - y) == 0
                if verdict:
                    return "equal"
            return "not_equal"
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

def summarise(refs, hyps, latencies):
    """Standard metric block for one system on one dataset."""
    verdicts = [symeq(r, h) for r, h in zip(refs, hyps)]
    n = len(refs)
    lat = sorted(latencies)
    return {
        "n": n,
        "cer": round(cer(refs, hyps), 4),
        "nem": round(sum(nem(r, h) for r, h in zip(refs, hyps)) / n, 4),
        "symeq": round(verdicts.count("equal") / n, 4),
        "symeq_unparseable": round(verdicts.count("unparseable") / n, 4),
        "symeq_timeout": verdicts.count("timeout"),
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
