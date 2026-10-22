"""Calculus steps for the checker: derivatives, indefinite and definite integrals, evaluation bars.

The residual-ratio test samples lines at random points through lambdify, which cannot print an
unevaluated Derivative or Integral (PrintMethodNotImplementedError). So calculus is handled here
first, side by side: when one side of the previous line holds an operator, the same side of the
current line must be its correct result. Anything left over is evaluated and handed back to the
ordinary algebra check.

- d/dx(...)            : the side must equal the evaluated derivative.
- \\int f dx            : the side must differentiate back to f (a missing + C is a caution note).
- \\int_a^b f dx        : [F]_a^b must use an antiderivative of f (else an antiderivative error) and
                         the same limits; a number must equal the integral (else an evaluation error).
- [F]_a^b, F|_a^b       : parsed into EvalBar(F, x, a, b) = F(b) - F(a).
"""

import re

import sympy

from src.mathparse import parse_graded

X = sympy.Symbol("x")
CONSTANT_NAMES = {"C", "c", "K", "C_1", "C_{1}", "C_0", "C_{0}"}

# [F]_a^b and F\Big|_a^b (also \left[...\right], \right|, \bigg|, plain |)
_BAR = re.compile(
    r"^\s*(?:\\left\s*\[\s*(?P<f1>.+?)\s*\\right\s*\]|\[\s*(?P<f2>.+?)\s*\]"
    r"|(?:\\left\s*\.)?\s*(?P<f3>.+?)\s*(?:\\right|\\Big|\\big|\\bigg|\\Bigg)?\s*\|)"
    r"\s*_\s*(?:\{(?P<a1>[^{}]+)\}|(?P<a2>[^\s^{}]))\s*\^\s*(?:\{(?P<b1>[^{}]+)\}|(?P<b2>[^\s_{}]))\s*$")


class EvalBar(sympy.Function):
    """[F]_a^b: kept unevaluated until doit(), so the antiderivative F stays checkable."""
    nargs = 4

    def doit(self, **hints):
        f, x, a, b = self.args
        return f.subs(x, b) - f.subs(x, a)


def _parse(latex):
    return parse_graded(latex)[0]


def parse_side(latex):
    """A SymPy object for one side of a line, with evaluation bars understood (None if unreadable)."""
    m = _BAR.match(latex)
    if not m:
        return _parse(latex)
    f_latex = re.sub(r"\\(?:right|Big|big|bigg|Bigg)\s*$", "", (m.group("f1") or m.group("f2") or m.group("f3")).strip())
    f = _parse(f_latex)
    a, b = (_parse(m.group(k + "1") or m.group(k + "2")) for k in ("a", "b"))
    if f is None or a is None or b is None or not isinstance(f, sympy.Expr):
        return None
    syms = sorted(f.free_symbols, key=str)
    var = X if X in syms or not syms else syms[0]
    return EvalBar(f, var, a, b)


def parse_bar_line(latex):
    """Eq(lhs, rhs) for a line with an evaluation bar on a side, else None."""
    s = (latex or "").strip().strip("$")
    if "|" not in s and "]_" not in s.replace(" ", ""):
        return None
    parts = re.split(r"(?<![<>\\!])=(?!=)", s)
    if len(parts) != 2:
        return None
    sides = [parse_side(p) for p in parts]
    if any(x is None or not isinstance(x, sympy.Expr) for x in sides) or not any(x.has(EvalBar) for x in sides):
        return None
    return sympy.Eq(*sides, evaluate=False)


def has_calculus(e):
    return e is not None and e.has(sympy.Derivative, sympy.Integral, EvalBar)


def _indefinite(e):
    return [i for i in e.atoms(sympy.Integral) if any(len(lim) == 1 for lim in i.limits)]


def constants(e, before):
    """Integration-constant symbols (C, c, K, C_1) in e that were not in the previous line."""
    return {s for s in e.free_symbols if s.name in CONSTANT_NAMES and s not in before}


def evaluate(e):
    """Calculus operators evaluated: derivatives, integrals (indefinite: no constant) and bars."""
    return e.replace(lambda x: isinstance(x, EvalBar), lambda x: x.doit()).doit()


def _zero(e, x):
    """e is identically zero (exact simplify; callers hold the checker's time limit)."""
    return sympy.simplify(e) == 0


def _show(e):
    return sympy.sstr(e).replace("**", "^").replace("*", "")


def _side(p_side, c_side, before):
    """(status, why) for one side holding a calculus operator in the previous line, or None if
    this side has none. status is "ok", "caution" or "error"."""
    if not has_calculus(p_side):
        return None
    ders = p_side.atoms(sympy.Derivative)
    indef = _indefinite(p_side)
    defin = [i for i in p_side.atoms(sympy.Integral) if i not in indef]
    if indef and p_side == indef[0]:   # the side is one indefinite integral
        integral = indef[0]
        f, x = integral.function, integral.limits[0][0]
        consts = constants(c_side, before)
        anti = c_side.subs({k: 0 for k in consts})
        if has_calculus(anti):
            return None
        d = sympy.diff(anti, x)
        if not _zero(d - f, x):
            return "error", (f"{_show(anti)} is not an antiderivative of {_show(f)}: "
                             f"its derivative is {_show(sympy.simplify(d))}, not {_show(f)}")
        if not consts:
            return "caution", "correct antiderivative, but add the constant of integration (+ C)"
        return "ok", None
    if defin and p_side == defin[0] and isinstance(c_side, EvalBar):
        integral = defin[0]
        f, (x, a, b) = integral.function, integral.limits[0]
        F, var, ca, cb = c_side.args
        d = sympy.diff(F.subs(var, x), x)
        if not _zero(d - f, x):
            return "error", (f"incorrect antiderivative: the derivative of {_show(F)} is "
                             f"{_show(sympy.simplify(d))}, not {_show(f)}")
        if not (_zero(ca - a, x) and _zero(cb - b, x)):
            return "error", f"the limits changed: the integral runs from {_show(a)} to {_show(b)}"
        return "ok", None
    if has_calculus(c_side):
        return None   # e.g. an integral split into two: left to the evaluated comparison
    value = evaluate(p_side)
    if isinstance(p_side, EvalBar) and not _zero(value - c_side, X):
        F, var, a, b = p_side.args
        return "error", (f"incorrect evaluation: F({_show(b)}) - F({_show(a)}) with F = {_show(F)} is "
                         f"{_show(sympy.nsimplify(value))}, not {_show(c_side)} (the antiderivative itself is fine)")
    if _zero(value - c_side, X):
        return "ok", None
    if ders and p_side == next(iter(ders)):
        der = next(iter(ders))
        return "error", (f"the derivative of {_show(der.expr)} is {_show(sympy.simplify(value))}, "
                         f"not {_show(c_side)}")
    if defin and p_side == defin[0]:
        return "error", (f"incorrect evaluation: the integral equals {_show(sympy.nsimplify(value))}, "
                         f"not {_show(c_side)}")
    return None


def calculus_step(pr, cr):
    """{"status", "explanation"} for a step whose previous line holds calculus, side by side;
    None when the step is not of that shape (the caller then evaluates and checks as algebra)."""
    if pr is None or cr is None or not isinstance(pr, sympy.Eq) or not isinstance(cr, sympy.Eq):
        return None
    before = pr.free_symbols
    for (ps, cs), (po, co) in (((pr.lhs, cr.lhs), (pr.rhs, cr.rhs)), ((pr.rhs, cr.rhs), (pr.lhs, cr.lhs))):
        res = _side(ps, cs, before)
        if res is None:
            continue
        status, why = res
        if status == "error":
            return {"status": "error", "explanation": why}
        other = evaluate(po) - evaluate(co)
        if not _zero(other, X):
            return None   # the other side changed too: an algebra step, checked after evaluation
        if isinstance(cs, EvalBar):
            return {"status": "valid", "explanation": None}
        return {"status": "caution" if status == "caution" else "valid", "explanation": why}
    return None
