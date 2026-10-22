"""Safe LaTeX -> SymPy parsing shared by the eval metrics, System B and the checker.

`parse_graded` returns (obj, level): "clean" if the raw string parses, "cleaned" if it parses only
after the cosmetic cleanup below, "none" otherwise. Things SymPy cannot represent (relations such
as \\in or \\to, chains a = b = c, tuples, sets, intervals, prose separators) become a
`sympy.Tuple` whose first element is a marker Symbol ("__rel__:in", "__tuple__", "__set__", ...),
so two objects can be compared elementwise. Decorated or font-changed atoms (\\bar{x},
\\mathcal{N}, \\alpha', G^{++}) become Symbols with distinct names that no parser can produce
from plain input, so a cleanup never makes different maths compare equal.
"""

import functools
import logging
import re
import signal
import threading
import warnings
from contextlib import contextmanager

import sympy
from sympy.core.function import AppliedUndef
from sympy.core.relational import Relational
from sympy.parsing.latex import parse_latex

from src.config import SYMPY_TIMEOUT_S

log = logging.getLogger(__name__)

_CMD = re.compile(r"\\([A-Za-z]+)")
_NL = r"(?![A-Za-z])"   # a command ends here (so \le does not match \left)
# Raw ('clean') attempt: only what HEAD already stripped (\left/\right, \, \; \: \!, \displaystyle).
_STRIP_RAW = re.compile(r"\\(?:left|right)" + _NL + r"\s*\.?|\\[,;:!](?![A-Za-z])"
                        + r"|\\displaystyle" + _NL)
# Cleanup pass: also sizing commands, ~, \limits, \textstyle, "\ ". \quad is NOT stripped here: it may
# separate two statements (see _quads).
_STRIP = re.compile(r"\\(?:left|right)" + _NL + r"\s*\.?|\\[Bb]igg?[lrm]?" + _NL
                    + r"|\\[,;:!>](?![A-Za-z])|\\(?:display|text)style" + _NL
                    + r"|\\(?:no)?limits" + _NL + r"|~|\\ ")
# evaluation bar "\left. F \right|_{x=0}" / "\Big|_0^1": the evaluation point would be lost
_EVAL_BAR = re.compile(r"\\(?:right|[Bb]igg?r?)" + _NL + r"\s*(?:\||\\[rl]?vert" + _NL + r")\s*_"
                       r"|\\left\s*\..*\\right\s*(?:\||\\[rl]?vert" + _NL + r")\s*\^")
MAX_LEN = 1000   # longer inputs are 'none' (en_dev refs are <= 142 chars); bounds work in threads
_LB = "\x00LB\x00"


class Timeout(BaseException):
    """BaseException so that no generic `except Exception` handler can swallow it."""


@contextmanager
def time_limit(seconds):
    """Raise Timeout after `seconds` (SIGALRM; a no-op outside the main thread). Do not nest."""
    if threading.current_thread() is not threading.main_thread():
        yield
        return

    def handler(signum, frame):
        raise Timeout()

    old = signal.signal(signal.SIGALRM, handler)
    signal.setitimer(signal.ITIMER_REAL, seconds)
    try:
        yield
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
        signal.signal(signal.SIGALRM, old)


def _strip_cosmetic(s, pat=_STRIP):
    """Line breaks protected, "\\ {" -> "\\{", then sizing/spacing commands removed."""
    s = s.replace("\\\\", _LB)
    s = re.sub(r"\\\s+\{", r"\\{", s)
    s = re.sub(r"\\\s+\}", r"\\}", s)
    s = pat.sub(" ", s)
    return s.replace(_LB, "\\\\")


def clean_for_parse(latex):
    """Minimal cleanup for the 'clean' (raw) parse attempt ('' if the string must not be parsed)."""
    s = (latex or "").strip().strip("$").strip()
    if len(s) > MAX_LEN or _EVAL_BAR.search(s):
        return ""
    s = _strip_cosmetic(s, _STRIP_RAW)
    s = s.rstrip(".,; ")
    return s.strip()


# ---------- vocabularies ----------

GREEK = set("alpha beta gamma delta epsilon zeta eta theta iota kappa lambda mu nu xi omicron pi rho "
            "sigma tau upsilon phi chi psi omega Gamma Delta Theta Lambda Xi Pi Sigma Upsilon Phi Psi "
            "Omega".split())
FUNCS = set("sin cos tan sec csc cot sinh cosh tanh arcsin arccos arctan log ln exp det max min".split())
CONNECTIVES = set("for all where if when with then and or is are by since so hence thus therefore such "
                  "that given provided as of on at the we let find solve solving gives which implies".split())
SYM_OK = GREEK | {"hbar", "ell", "infty"} | FUNCS | set(
    "frac sqrt lim sum prod int cdot times div binom mathit".split())
_TEXT = {"text", "textrm", "textit", "textbf", "textsf", "mathrm", "mathit", "operatorname", "mbox"}
_DECOR = {"bar": "bar", "overline": "bar", "hat": "hat", "widehat": "hat", "tilde": "tilde",
          "widetilde": "tilde", "vec": "vec", "overrightarrow": "vec", "dot": "dot", "ddot": "ddot",
          "check": "check", "breve": "breve", "underline": "underline", "acute": "acute",
          "grave": "grave", "mathbf": "mathbf", "boldsymbol": "mathbf", "bm": "mathbf",
          "mathcal": "mathcal", "mathbb": "mathbb", "mathfrak": "mathfrak", "mathscr": "mathscr",
          "mathsf": "mathsf"}
_OPNAMES = {"deg", "dim", "ker", "Tr", "tr", "rank", "sgn", "supp", "arg", "emptyset", "varnothing"}
_REL_CMDS = ["Rightarrow", "iff", "mapsto", "to", "subseteq", "subset", "supseteq", "supset", "notin",
             "in", "approx", "equiv", "simeq", "sim", "cong", "propto", "perp", "ll", "gg", "prec",
             "succ", "gtrsim", "lesssim", "mid", "leq", "geq", "neq", "forall", "exists"]
_REL_TOKS = {"\\" + c for c in _REL_CMDS} | {":=", ":", "<", ">", "="}
_BINOPS = {"\\" + c for c in "cup cap setminus oplus otimes circ wedge vee".split()}
_PH = re.compile(r"\\?zzq([a-z]{3})")
_TOK = re.compile(r"\\zzq[a-z]{3}|\\[A-Za-z]+|\\.|:=|\S")

_SYNONYMS = [(r"\\[dtc]frac" + _NL, r"\\frac"), (r"\\leqslant" + _NL, r"\\leq"), (r"\\le" + _NL, r"\\leq"),
             (r"\\geqslant" + _NL, r"\\geq"), (r"\\ge" + _NL, r"\\geq"), (r"\\not\s*\\?\s*=", r"\\neq "),
             (r"\\ne" + _NL, r"\\neq"), (r"\\not\s*\\in" + _NL, r"\\notin"),
             (r"\\[lr]?vert" + _NL, "|"), (r"\\[lr]?Vert" + _NL, r"\\|"),
             (r"\\var(phi|epsilon|theta|rho|sigma|pi)" + _NL, r"\\\1"),
             (r"\\(?:cdots|ldots|vdots|dotsc|dotsb|dotsm|dotso)" + _NL, r"\\dots"),
             (r"\\colon" + _NL, ":"), (r":\s*=", ":="), (r"\\implies" + _NL, r"\\Rightarrow"),
             (r"\\Leftrightarrow" + _NL, r"\\iff"), (r"\\rightarrow" + _NL, r"\\to"),
             (r"\\lbrace" + _NL, r"\\{"), (r"\\rbrace" + _NL, r"\\}")]


# ---------- brace / argument helpers ----------

def _match(s, i):
    """Index of the '}' matching the '{' at s[i] (ignoring escaped braces), or -1."""
    depth = 0
    j = i
    while j < len(s):
        c = s[j]
        if c == "\\":
            j += 2
            continue
        if c == "{":
            depth += 1
        elif c == "}":
            depth -= 1
            if depth == 0:
                return j
        j += 1
    return -1


def _skip_ws(s, i):
    while i < len(s) and s[i].isspace():
        i += 1
    return i


def _arg(s, i):
    """(text, inner, end) of a macro argument at i: a braced group or one token; None if absent."""
    i = _skip_ws(s, i)
    if i >= len(s):
        return None
    if s[i] == "{":
        j = _match(s, i)
        if j < 0:
            return None
        return s[i:j + 1], s[i + 1:j], j + 1
    m = re.match(r"\\zzq[a-z]{3}|\\[A-Za-z]+|[A-Za-z0-9]", s[i:])
    if not m:
        return None
    return m.group(0), m.group(0), i + m.end()


class _Ctx:
    """Per-call placeholder table: id -> (kind, canonical name)."""

    def __init__(self):
        self.table, self.ids, self.multi, self.text = {}, {}, set(), set()

    def ph(self, kind, name):
        key = (kind, name)
        if key not in self.ids:
            n = len(self.ids)
            pid = "".join(chr(97 + (n // 26 ** k) % 26) for k in (2, 1, 0))
            self.ids[key] = pid
            self.table[pid] = key
        return "\\zzq" + self.ids[key] + " "

    def name(self, pid):
        kind, name = self.table[pid]
        if kind == "prose":
            return name[len("text:"):]
        return name


def _canon(t, ctx):
    """Source text without whitespace, placeholders resolved, single-token scripts braced."""
    toks = _TOK.findall(t)
    out = []
    for k, tok in enumerate(toks):
        if k > 0 and toks[k - 1] in "_^" and tok != "{":
            tok = "{" + tok + "}"
        out.append(tok)
    s = "".join(out)
    return _PH.sub(lambda m: ctx.name(m.group(1)), s) if "zzq" in s else s


# ---------- A: token-level normalisation ----------

def _collapse_double_braces(s):
    changed = True
    while changed:
        changed = False
        for m in re.finditer(r"(?<!\\)\{\s*\{", s):
            i = m.start()
            j = _match(s, i)
            k = m.end() - 1
            kk = _match(s, k)
            if j > 0 and kk > 0 and s[kk + 1:j].strip() == "":
                s = s[:i] + s[k:kk + 1] + s[j + 1:]
                changed = True
                break
    return s


def _brace_call(s, cmd, ctx):
    r"""\max\{...\} -> \max(...)."""
    while True:
        m = re.search(r"\\" + cmd + _NL + r"\s*\\\{", s)
        if not m:
            return s
        depth, j = 0, m.end() - 2
        while j < len(s):
            if s.startswith(r"\{", j):
                depth += 1
                j += 2
            elif s.startswith(r"\}", j):
                depth -= 1
                if depth == 0:
                    break
                j += 2
            else:
                j += 1
        if j >= len(s):
            return s
        s = s[:m.start()] + "\\" + cmd + "(" + s[m.end():j] + ")" + s[j + 2:]


def _higher_derivs(s):
    r"""\frac{d^{N} X}{d v^{N}} -> \frac{d}{dv} ... \frac{dX}{dv} (N a digit)."""
    pat = re.compile(r"\\frac\s*\{\s*(d|\\partial)\s*\^\s*\{?\s*(\d)\s*\}?\s*((?:[^{}]|\{[^{}]*\})*)\}\s*"
                     r"\{\s*(?:d|\\partial)\s*([A-Za-z])\s*\^\s*\{?\s*(\d)\s*\}?\s*\}")

    def rep(m):
        d, n, x, v, n2 = m.group(1), int(m.group(2)), m.group(3).strip(), m.group(4), int(m.group(5))
        if n != n2 or n < 2:
            return m.group(0)
        sp = " " if d.startswith("\\") else ""
        one = "\\frac{%s}{%s%s%s} " % (d, d, sp, v)
        return one * (n - 1) + "\\frac{%s%s%s}{%s%s%s} " % (d, sp, x, d, sp, v) if x else one * n

    return pat.sub(rep, s)


_QUAD = re.compile(r"\s*(?:\\q?quad" + _NL + r"\s*)+")
_QUAD_SEP_AFTER = re.compile(r"(?:[=<>,(\[{+\-*/:]|\\\{|\\(?:" + "|".join(_REL_CMDS) + r"|cdot|times|div|cup|cap"
                             r"|setminus|oplus|otimes|circ|wedge|vee)(?![A-Za-z]))\s*$")
_QUAD_SEP_BEFORE = re.compile(r"\s*(?:[=<>,)\]}+\-*/:]|\\\}|\\(?:" + "|".join(_REL_CMDS)
                              + r"|cdot|times|div)(?![A-Za-z]))")


def _quads(s, ctx):
    r"""\quad between two operands separates two statements ("a=1 \quad b=2"): a 'gap' placeholder
    that _parse_structured splits on. Next to an operator / relation / comma it is only spacing."""
    out, last = "", 0
    for m in _QUAD.finditer(s):
        before, after = s[:m.start()], s[m.end():]
        out += s[last:m.start()]
        last = m.end()
        if not before.strip() or not after.strip() or _QUAD_SEP_AFTER.search(before) \
                or _QUAD_SEP_BEFORE.match(after):
            out += " "
        else:
            out += " " + ctx.ph("gap", "quad")
    return out + s[last:]


def _substack(s):
    r"""\substack{i=1} -> i=1. A multi-line \substack (several conditions) is rejected."""
    while True:
        m = re.search(r"\\substack" + _NL + r"\s*\{", s)
        if not m:
            return s
        j = _match(s, m.end() - 1)
        inner = s[m.end():j] if j > 0 else ""
        if j < 0 or _LB in inner or "\\\\" in inner or not inner.strip():
            raise ValueError("unsupported \\substack")
        s = s[:m.start()] + "{" + inner + "}" + s[j + 1:]


_G = r"(\{(?:[^{}]|\{[^{}]*\})*\}|\\[A-Za-z]+|[^\s{}\\_^])"


def _normalise_tokens(s, ctx):
    s = s.strip().strip("$").strip()
    if s.startswith(r"\[") and s.endswith(r"\]"):
        s = s[2:-2]
    s = _strip_cosmetic(s).replace("\\\\", _LB)
    s = s.strip().rstrip(".,; ").strip()
    while s.startswith("{") and _match(s, 0) == len(s) - 1:
        s = s[1:-1].strip()   # redundant braces around the whole string
    for pat, rep in _SYNONYMS:
        s = re.sub(pat, rep, s)
    s = _quads(s, ctx)
    s = _substack(s)
    s = re.sub(r"\^\s*\{\s*\\prime\s*\\prime\s*\}", "''", s)
    s = re.sub(r"\^\s*(?:\{\s*\\prime\s*\}|\\prime" + _NL + ")", "'", s)
    s = re.sub(r"\{\s*\\(sf|rm|bf|cal|it)" + _NL + r"\s*([^{}]*)\}", lambda m: "\\math%s{%s}" % m.groups(), s)
    s = _brace_call(_brace_call(s, "max", ctx), "min", ctx)
    sep = ctx.ph("sep", "mod")
    s = re.sub(r"\\pmod\s*(?:\{([^{}]*)\}|(\\[A-Za-z]+|[A-Za-z0-9]+))",
               lambda m: " " + sep + "(" + (m.group(1) or m.group(2)) + ")", s)
    s = re.sub(r"\(\s*\\(?:text|mathrm|operatorname)\s*\{\s*mod\s*\}\s*([^()]*)\)",
               lambda m: " " + sep + "(" + m.group(1) + ")", s)
    s = re.sub(r"\\(?:bmod|mod)" + _NL, lambda m: " " + sep, s)
    deg = " \\cdot " + ctx.ph("sym", r"\degree")
    s = re.sub(r"\^\s*(?:\{\s*\\circ\s*\}|\\circ" + _NL + ")", lambda m: deg, s)
    s = re.sub(r"\\degree" + _NL, lambda m: deg, s)
    s = _collapse_double_braces(s)
    s = re.sub(r"\^\s*" + _G + r"\s*_\s*" + _G, r"_\2^\1", s)
    return _higher_derivs(s)


# ---------- B: atom placeholders ----------

_LABEL = re.compile(r"^(?:\+\+|--|\+|-|\*|\\ast|\\dagger|\\star|\\pm|\\mp|\(.*\))$")


def _tail(s, j, ctx):
    """Parse sub/primes/label superscript after an atom. Returns dict with end, text, flags."""
    start, sub, primes, label = j, None, "", None
    k = _skip_ws(s, j)
    m = re.match(r"'+", s[k:])
    if m:
        primes += m.group(0)
        j = k = k + m.end()
        k = _skip_ws(s, k)
    if k < len(s) and s[k] == "_":
        k2 = _skip_ws(s, k + 1)
        a = (s[k2], s[k2], k2 + 1) if s[k2:k2 + 1] in ("+", "-", "*") else _arg(s, k + 1)
        if a:
            sub = a[1]
            j = k = a[2]
            k = _skip_ws(s, k)
    m = re.match(r"'+", s[k:])
    if m:
        primes += m.group(0)
        j = k = k + m.end()
        k = _skip_ws(s, k)
    if k < len(s) and s[k] == "^":
        k2 = _skip_ws(s, k + 1)
        rest = s[k2:]
        if rest.startswith("{"):
            e = _match(s, k2)
            inner = re.sub(r"\s+", "", s[k2 + 1:e]) if e > 0 else ""
            if e > 0 and _LABEL.match(inner) and (not inner.startswith("(") or _paren_whole(inner)):
                label, j = inner, e + 1
        else:
            m = re.match(r"\\(?:ast|dagger|star|pm|mp)" + _NL + r"|[+\-*]", rest)
            if m and not (m.group(0) in "+-" and re.match(r"[A-Za-z0-9\\]", rest[1:2])):
                label, j = m.group(0), k2 + m.end()
    return {"end": j, "text": s[start:j], "sub": sub, "primes": primes, "label": label}


def _paren_whole(t):
    depth = 0
    for i, c in enumerate(t):
        depth += c == "("
        depth -= c == ")"
        if depth == 0 and i < len(t) - 1:
            return False
    return depth == 0


def _sub_has_comma(sub):
    if sub is None:
        return False
    depth = 0
    for c in sub:
        depth += c in "{(["
        depth -= c in "})]"
        if c == "," and depth == 0:
            return True
    return False


def _name_like_sub(sub):
    if sub is None:
        return False
    toks = _TOK.findall(sub)
    ok = all(re.fullmatch(r"[A-Za-z0-9]", t) or (t[1:] in GREEK and t.startswith("\\")) for t in toks)
    return ok and len(toks) >= 2 and not all(t.isdigit() for t in toks)


def _text_rule(w_raw, ctx):
    """Only whitespace is removed: '/', '<', '-' etc. stay part of the opaque symbol name."""
    w = re.sub(r"\s+", "", w_raw)
    words = w_raw.split()
    if not w:
        return " ", False
    if w.lower() in FUNCS:
        return "\\" + w.lower() + " ", False
    if len(w) == 1 and w.isalnum():
        return w, w == "d"
    if any(x.lower().strip(",.;:") in CONNECTIVES for x in words):
        return ctx.ph("prose", "text:" + w), False
    out = ctx.ph("sym", w)
    ctx.text.add(w)
    if len(words) >= 2:
        ctx.multi.add(w)
    return out, False


def _pm_prefix(out):
    o = out.rstrip()
    if not o or o[-1] in "([{=,<>:":
        return True
    m = re.search(r"(\\[A-Za-z]+|\\\{)\s*$", o)
    return bool(m and (m.group(1) in _REL_TOKS or m.group(1) in _BINOPS or m.group(1) == r"\{"))


def _atoms(s, ctx):
    """Replace decorated/font/text/label atoms by placeholders (left to right, innermost first)."""
    out, i = "", 0
    while i < len(s):
        m = re.match(r"\\(zzq[a-z]{3}|[A-Za-z]+\*?)", s[i:])
        if m:
            cmd = m.group(1)
            j = i + m.end()
            base_cmd = cmd.rstrip("*")
            if base_cmd in _TEXT and (cmd == base_cmd or base_cmd == "operatorname"):
                a = _arg(s, j)
                if a is None:
                    return None
                rep, eat_ws = _text_rule(a[1], ctx)
                rest = s[a[2]:]
                if eat_ws:
                    rest = rest.lstrip()
                s = s[:i] + rep + rest
                continue        # rescan: the replacement may carry a tail
            if cmd in _DECOR:
                a = _arg(s, j)
                if a is None:
                    return None
                if not a[1].strip():
                    return None   # \bar{} / \mathcal{}: nothing to decorate
                inner = _atoms(a[1], ctx) if a[0].startswith("{") else a[1]
                if inner is None:
                    return None
                t = _tail(s, a[2], ctx)
                tail_text = _atoms(t["text"], ctx)
                name = "\\%s{%s}%s" % (_DECOR[cmd], _canon(inner, ctx), _canon(tail_text or "", ctx))
                out += ctx.ph("sym", name)
                i = t["end"]
                continue
            if cmd in _OPNAMES:
                out += ctx.ph("sym", "\\emptyset" if cmd == "varnothing" else "\\" + cmd)
                i = j
                continue
            if cmd in ("pm", "mp"):
                pm = ctx.ph("sym", "\\" + cmd) + "\\cdot "
                out += (" " + pm) if _pm_prefix(out) else (" + " + pm)
                i = j
                continue
            if cmd in GREEK or cmd in ("hbar", "ell") or cmd.startswith("zzq"):
                r = _label_atom(s, i, j, m.group(0), ctx, cmd_base=True)
                if r is None:
                    return None
                out, i = out + r[0], r[1]
                continue
            out += m.group(0)
            i = j
            continue
        if s.startswith(r"\|", i):
            depth, k = 0, i + 2
            while k < len(s):
                if s.startswith(r"\|", k) and depth == 0:
                    break
                if s[k] == "\\":
                    k += 2
                    continue
                depth += s[k] in "{(["
                depth -= s[k] in "})]"
                k += 1
            if k >= len(s):
                return None
            s = s[:i] + ctx.ph("func", "__norm__") + "(" + s[i + 2:k] + ")" + s[k + 2:]
            continue
        if s[i] == "\\":
            out += s[i:i + 2]
            i += 2
            continue
        if s[i].isalpha():
            r = _label_atom(s, i, i + 1, s[i], ctx, cmd_base=False)
            if r is None:
                return None
            out, i = out + r[0], r[1]
            continue
        out += s[i]
        i += 1
    return out


def _label_atom(s, i, j, base, ctx, cmd_base):
    """A letter / Greek / placeholder base with an optional tail (B4, B5)."""
    t = _tail(s, j, ctx)
    if t["text"].strip() == "":
        return base, j
    sub_atoms = _atoms(t["sub"], ctx) if t["sub"] is not None else None
    if t["sub"] is not None and sub_atoms is None:
        return None
    sign_sub = t["sub"] is not None and _LABEL.match(re.sub(r"\s+", "", t["sub"]))
    if (cmd_base and t["primes"]) or t["label"] or sign_sub or _sub_has_comma(t["sub"]):
        tail = (("_{" + sub_atoms + "}") if t["sub"] is not None else "") + t["primes"]
        tail += ("^{" + t["label"] + "}") if t["label"] else ""
        return ctx.ph("sym", _canon(base + tail, ctx)), t["end"]
    if _name_like_sub(t["sub"]):
        k = s.index("_", j)
        a = _arg(s, k + 1)
        return base + s[j:k] + "_{" + ctx.ph("sym", _canon(t["sub"], ctx)).strip() + " }", a[2]
    if t["sub"] is not None and sub_atoms != t["sub"]:
        k = s.index("_", j)
        a = _arg(s, k + 1)
        return base + s[j:k] + "_{" + sub_atoms + "}", a[2]
    return base, j


def _cleanup(s):
    """(cleaned string, ctx) or (None, None)."""
    if "zzq" in s:
        return None, None
    ctx = _Ctx()
    s = _normalise_tokens(s, ctx)
    s = _atoms(s, ctx)
    if s is None:
        return None, None
    s = _nabla(s, ctx)
    if s is None:
        return None, None
    return s.replace(_LB, "\\\\"), ctx


_NABLA = re.compile(r"\\nabla" + _NL + r"\s*(\\cdot" + _NL + r"|\\times" + _NL + r"|\^\s*(?:\{\s*2\s*\}|2))?\s*")
_NABLA_KIND = {None: "__grad__", "cdot": "__div__", "times": "__curl__", "2": "__lap__"}


def _operand(s, i):
    """End of the operand starting at s[i] (a (...) group or one atom with optional subscript), or -1."""
    if i < len(s) and s[i] == "(":
        depth = 0
        for k in range(i, len(s)):
            depth += s[k] == "("
            depth -= s[k] == ")"
            if depth == 0:
                return k + 1
        return -1
    m = re.match(r"\\zzq[a-z]{3}\s*|\\[A-Za-z]+\s*|[A-Za-z]", s[i:])
    if not m or (m.group(0).startswith("\\") and not m.group(0).startswith("\\zzq")
                 and m.group(0).strip()[1:] not in GREEK | {"nabla"}):
        return -1
    j = i + m.end()
    k = _skip_ws(s, j)
    if k < len(s) and s[k] == "_":
        a = _arg(s, k + 1)
        if a is None:
            return -1
        j = a[2]
    return j


def _nabla(s, ctx):
    r"""\nabla \cdot X -> __div__(X), \nabla \times X -> __curl__(X), \nabla^2 X -> __lap__(X),
    \nabla X -> __grad__(X) (innermost operand first). A bare \nabla (e.g. a label) is a distinct symbol;
    an operand whose extent is unclear makes the string unparseable."""
    while True:
        m = None
        for m in _NABLA.finditer(s):
            pass   # last occurrence first, so an operand never contains an unconverted \nabla
        if m is None:
            return s
        rest = s[m.end():]
        if not rest.strip() or rest.lstrip()[0] in "}])" and m.group(1) is None:
            s = s[:m.start()] + ctx.ph("sym", r"\nabla") + rest
            continue
        kind = m.group(1)
        kind = None if kind is None else ("2" if kind.startswith("^") else kind.lstrip("\\"))
        end = _operand(s, m.end())
        if end < 0:
            return None
        arg = s[m.end():end].strip()
        if arg.startswith("(") and arg.endswith(")"):
            arg = arg[1:-1]
        s = s[:m.start()] + ctx.ph("func", _NABLA_KIND[kind]) + "(" + arg + ")" + s[end:]


# ---------- D/E: hazards and validity ----------

_HAZARDS = [re.compile(p) for p in (
    r"\\[A-Za-z]+\s*(?:_\s*(?:\{[^{}]*\}|\\[A-Za-z]+|\w))?\s*'",
    r"(?<!\\int)(?<!\\sum)(?<!\\prod)(?<!\\oint)\^\s*(?:\{(?:[^{}]|\{[^{}]*\})*\}|\\?\w+)\s*_",
    r"\^\s*\{\s*\([^{}()]*\)\s*\}",
    r"\\\{",
    r"\\frac\s*\{\s*(?:d|\\partial)\s*\^")]


def _hazards(s):
    """True if `s` is safe to hand to a backend (no known silent mis-parse)."""
    return not any(p.search(s) for p in _HAZARDS)


_LIMITED = (sympy.Derivative, sympy.Integral, sympy.Sum, sympy.Product, sympy.Limit)


def _has_tuple(expr):
    """A Tuple anywhere except as the limits/variables of a calculus object."""
    if isinstance(expr, sympy.Tuple):
        return True
    args = expr.args[:1] if isinstance(expr, _LIMITED) else expr.args
    return any(_has_tuple(a) for a in args if isinstance(a, sympy.Basic))


def _valid(expr, s):
    if not isinstance(expr, sympy.Basic) or isinstance(expr, sympy.logic.boolalg.BooleanAtom):
        return False   # lark returns a Tree for ambiguous input
    if isinstance(expr, Relational) and any(isinstance(a, Relational) for a in expr.args):
        return False   # "a = b = c" parses as Eq(Eq(a, b), c)
    if _has_tuple(expr) or expr.has(sympy.conjugate) or expr.has(sympy.Symbol("derivative")):
        return False
    letters = set(re.findall(r"[A-Za-z]", _CMD.sub(" ", s))) | set(_CMD.findall(s))
    for integral in expr.atoms(sympy.Integral):
        if any(str(v) not in letters for v in integral.variables):
            return False   # ANTLR invents x for "\int f"
    # ANTLR turns unknown commands like \text into a Symbol named "text"
    bad = set(_CMD.findall(s)) - SYM_OK - {m for m in _CMD.findall(s) if m.startswith("zzq")}
    names = [str(x.name) for x in expr.free_symbols] + [a.func.__name__ for a in expr.atoms(AppliedUndef)]
    return not any(tok in bad for n in names for tok in re.findall(r"[A-Za-z]+", n))


def _backend(s, ctx=None):
    """Parse one atom string: ANTLR, then lark (lark cannot read placeholders)."""
    for kwargs in ({"strict": True}, {"backend": "lark"}):
        if "backend" in kwargs and re.search(r"zzq|\\(?:left|right|[Bb]ig)[A-Za-z]", s):
            break   # lark reads \rightarrow as \right + "arrow"
        try:
            expr = parse_latex(s, **kwargs)
        except Timeout:
            raise
        except Exception:
            continue
        if _valid(expr, s):
            return expr
    return None


def _rename(expr, ctx, s):
    def fix(name):
        name = re.sub(r"[\s*]", "", str(name))
        return _PH.sub(lambda m: ctx.name(m.group(1)), name) if ctx else name

    reps = {x: sympy.Symbol(fix(x.name)) for x in expr.free_symbols if isinstance(x, sympy.Symbol)
            and (fix(x.name) != x.name or type(x.name) is not str)}
    expr = expr.xreplace(reps)
    if ctx:
        expr = expr.replace(lambda e: isinstance(e, AppliedUndef) and _PH.search(e.func.__name__),
                            lambda e: sympy.Function(fix(e.func.__name__))(*e.args))
    if re.search(r"\\pi" + _NL, s):
        expr = expr.xreplace({sympy.Symbol("pi"): sympy.pi})
    return expr


# ---------- section 3: structured parse ----------

_OPENERS = {"{", "(", "[", r"\{", r"\langle"}
_CLOSERS = {"}", ")", "]", r"\}", r"\rangle"}


def _tokens(s):
    return [(m.start(), m.group(0)) for m in _TOK.finditer(s)]


def _split(s, is_op):
    """Split `s` at depth-0 tokens for which is_op(tok) gives a name. Returns (parts, ops)."""
    toks = _tokens(s)
    parts, ops, depth, last = [], [], 0, 0
    for k, (pos, tok) in enumerate(toks):
        if tok in _OPENERS:
            depth += 1
        elif tok in _CLOSERS:
            depth -= 1
        elif depth == 0 and not (k > 0 and toks[k - 1][1] in ("_", "^")):
            name = is_op(tok, toks[k + 1][1] if k + 1 < len(toks) else "")
            if name is not None:
                parts.append(s[last:pos])
                ops.append(name)
                last = pos + len(tok)
    parts.append(s[last:])
    return parts, ops


def _wrapper(s):
    """(opener, inner, closer, suffix) if the whole of s is one bracketed group, else None."""
    toks = _tokens(s)
    if not toks or toks[0][1] not in _OPENERS or toks[0][1] == "{":
        return None
    depth = 0
    for k, (pos, tok) in enumerate(toks):
        depth += tok in _OPENERS
        depth -= tok in _CLOSERS
        if depth == 0:
            inner = s[len(toks[0][1]) + toks[0][0]:pos]
            return toks[0][1], inner, tok, s[pos + len(tok):].strip()
    return None


_NATIVE = {"=": sympy.Eq, "<": sympy.Lt, ">": sympy.Gt, "leq": sympy.Le, "geq": sympy.Ge, "neq": sympy.Ne}


def _mk(marker, items):
    return sympy.Tuple(sympy.Symbol(marker), *items)


def _parse_structured(s, ctx):
    s = s.strip()
    if not s:
        return None
    if s == r"\dots":
        return sympy.Symbol("__dots__")
    parts, _ = _split(s, lambda t, n: "" if t == "\\\\" else None)
    if len(parts) > 1:
        parts = [p for p in parts if p.strip()]
        items = [_parse_structured(p, ctx) for p in parts]
        return None if any(x is None for x in items) else (_mk("__lines__", items) if len(items) > 1 else items[0])

    def prose(t, n):
        m = _PH.fullmatch(t[1:]) if t.startswith("\\zzq") else None
        return ctx.table[m.group(1)][1] if m and ctx.table[m.group(1)][0] in ("prose", "gap") else None

    parts, ops = _split(s, prose)
    if ops:
        items = [_parse_structured(p, ctx) for p in parts if p.strip()]
        if not items or any(x is None for x in items):
            return None
        return _mk("__rel__:" + ",".join(ops), items)

    if re.fullmatch(r"-?\d{1,3}(?:,\d{3})+(?:\.\d+)?", s):
        s = s.replace(",", "")   # thousands separators: 1,000
    parts, ops = _split(s, lambda t, n: "," if t == "," else None)
    if ops:
        if any(re.fullmatch(r"0\d+(?:\.\d*)?", p.strip()) for p in parts):
            return None   # "1,000,5" / "3,05": a leading-zero element is a mangled number, not a list
        items = [_parse_structured(p, ctx) for p in parts]
        return None if any(x is None for x in items) else _mk("__list__", items)

    def rel(t, n):
        if t in _REL_TOKS:
            return t.lstrip("\\")
        m = _PH.fullmatch(t[1:]) if t.startswith("\\zzq") else None
        return ctx.table[m.group(1)][1] if m and ctx.table[m.group(1)][0] == "sep" else None

    parts, ops = _split(s, rel)
    if ops:
        keep = [p for k, p in enumerate(parts)
                if p.strip() or not (k < len(ops) and ops[k] in ("forall", "exists"))]
        items = [_parse_structured_part(p, ctx) for p in keep]
        if any(x is None for x in items):
            return None
        if len(ops) == 1 and ops[0] in _NATIVE and len(items) == 2 and all(
                isinstance(x, sympy.Expr) for x in items):
            return _NATIVE[ops[0]](items[0], items[1], evaluate=False)
        return _mk("__rel__:" + ",".join(ops), items)
    return _parse_structured_part(s, ctx)


def _parse_structured_part(s, ctx):
    s = s.strip()
    if not s:
        return None
    if s == r"\dots":
        return sympy.Symbol("__dots__")
    parts, ops = _split(s, lambda t, n: t.lstrip("\\") if t in _BINOPS and n not in ("_", "^") else None)
    if ops:
        items = [_parse_structured_part(p, ctx) for p in parts]
        return None if any(x is None for x in items) else _mk("__op__:" + ",".join(ops), items)
    w = _wrapper(s)
    if w:
        o, inner, c, suffix = w
        elems, commas = _split(inner, lambda t, n: "," if t == "," else None)
        items = lambda: [_parse_structured(e, ctx) for e in elems]  # noqa: E731
        marker = None
        if o == r"\{" and c == r"\}":
            if suffix:
                m = re.fullmatch(r"_\s*(\{.*\}|\S)", suffix)
                if not m:
                    return None
                seq = [_parse_structured(inner, ctx), _parse_structured(_unbrace(m.group(1)), ctx)]
                return None if any(x is None for x in seq) else _mk("__seq__", seq)
            marker = "__set__"
        elif suffix:
            marker = None
        elif o == r"\langle" and c == r"\rangle":
            marker = "__angle__"
        elif o == "(" and c == ")":
            marker = "__tuple__" if commas else None
        elif o == "[" and c == "]":
            marker = "__bracket__" if commas else None
        elif o == "[" and c == ")":
            marker = "__interval[)__"
        elif o == "(" and c == "]":
            marker = "__interval(]__"
        if marker:
            xs = items()
            return None if any(x is None for x in xs) else _mk(marker, xs)
    return _atom(s, ctx)


def _unbrace(t):
    t = t.strip()
    return t[1:-1] if t.startswith("{") and t.endswith("}") else t


def _atom(s, ctx):
    if any(ctx.table[m.group(1)][0] == "gap" for m in _PH.finditer(s)):
        return None   # a \quad gap inside a group: statements would fuse by multiplication
    if r"\dots" in s:
        s = re.sub(r"\\dots" + _NL, lambda m: ctx.ph("sym", r"\dots"), s)
    if not _hazards(s):
        return None
    expr = _backend(s, ctx)
    return None if expr is None else _rename(expr, ctx, s)


# ---------- public API ----------

def parse_graded(latex, timeout=SYMPY_TIMEOUT_S):
    """(obj, level), level in {"clean", "cleaned", "none"}; obj is None iff level == "none".

    "clean": the raw string parses with only $ / trailing punctuation removed and HEAD's minimal
    stripping (\\left, \\right, \\, \\; \\: \\!, \\displaystyle). Everything else (\\quad, \\big..., ~,
    \\limits, fonts, \\text, structure splitting) counts as cleanup -> "cleaned".

    The object always comes from the cleaned pipeline (so ref and candidate take the same path);
    the level records whether the raw string parses too. Must not be called inside time_limit.
    """
    try:
        return _parse_graded_cached(latex or "", timeout)
    except Timeout:
        return None, "none"


def _names(obj):
    out = set()
    for e in sympy.preorder_traversal(obj):
        if isinstance(e, sympy.Symbol):
            out.add(e.name)
        elif isinstance(e, AppliedUndef):
            out.add(e.func.__name__)
    return out


@functools.lru_cache(maxsize=65536)
def _parse_graded_cached(latex, timeout):
    with time_limit(timeout), warnings.catch_warnings():
        warnings.simplefilter("ignore")
        try:
            return _parse_graded_body(latex)
        except Exception:   # RecursionError etc. (Timeout is a BaseException and propagates)
            log.debug("parse failed for %r", latex, exc_info=True)
            return None, "none"


def _parse_graded_body(latex):
    raw = clean_for_parse(latex)
    if not raw:
        return None, "none"
    raw_expr = None
    if _hazards(raw) and not re.search(r"\\q?quad" + _NL, raw):
        raw_expr = _backend(raw)
        if raw_expr is not None:
            raw_expr = _rename(raw_expr, None, raw)
    ctx = None
    try:
        cleaned, ctx = _cleanup(raw)
        obj = _parse_structured(cleaned, ctx) if cleaned is not None else None
    except Exception:
        log.debug("cleanup failed for %r", latex, exc_info=True)
        obj = None
    if obj is not None and isinstance(obj, sympy.Symbol) and obj.name in ctx.multi | {"__dots__"}:
        return None, "none"   # bare prose such as \text{hello world}, or a lone \dots
    if obj is not None and ctx.text and not all(any(w in n for n in _names(obj)) for w in ctx.text):
        return None, "none"   # a \text word vanished (e.g. multiplied into a 0): meaning lost
    if obj is not None:
        return obj, ("clean" if raw_expr is not None else "cleaned")
    if raw_expr is not None:
        log.debug("raw parse only: %r", latex)
        return raw_expr, "clean"
    return None, "none"


def try_parse(latex, timeout=SYMPY_TIMEOUT_S):
    """A SymPy object (any level), or None if `latex` is not parseable."""
    return parse_graded(latex, timeout)[0]


def parses(latex):
    return parse_graded(latex)[1] != "none"


def implicit_products(obj):
    """x(x - 3) as a product, not a call: single-argument applications of a one-letter name other
    than f, g, h become multiplication. Opt-in (used by the checker) so parse_graded, and every
    metric built on it, is unchanged."""
    return obj.replace(
        lambda e: isinstance(e, AppliedUndef) and len(e.args) == 1 and len(e.func.__name__) == 1
        and e.func.__name__ not in "fghFGH",
        lambda e: sympy.Symbol(e.func.__name__) * e.args[0])
