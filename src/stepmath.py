"""ProcessBench (3.4): maths lines from natural-language solution steps, and a step policy on top
of the residual-ratio checker.

extract(step) turns one prose step into ordered line records {latex, latex_lit, pos, hyp, neg, alt,
numeric, src}: delimited maths ($..$, \\(..\\), \\[..\\], environments) split at connectives, and
plain-text arithmetic ("48/2 = 24", "$400 - $80 = $320"). Lines are normalised for the parser
(units, percentages, degrees, number commas, \\boxed); unit conversions and remainders are skipped.

check_steps(steps) gives each step its verdicts: constant links inside one line ("3 + 4 = 7"), and
conservative two-line checks (same variables, same degree, no hypothesis/negation/disjunction cue).
The rules were written from step text alone (labels never read) and are frozen: see RULES_VERSION.
Every SymPy call runs under time_limit on the main thread; parse_graded never inside one.
"""

import re

import sympy

from src import checker
from src.mathparse import Timeout, parse_graded, time_limit

RULES_VERSION = "frozen-v6"   # v6: post-evaluation bug fixes (see results/reports/processbench.md)
SYMPY_TIMEOUT_S = checker.SYMPY_TIMEOUT_S
VERDICTS = ("valid", "caution", "error")

# ---------- delimited maths ----------

REL = re.compile(r"=|<|>|\\le\b|\\leq\b|\\ge\b|\\geq\b|\\lt\b|\\gt\b|\\neq\b|\\ne\b|≤|≥|≠|\\in\b")
APPROX = re.compile(r"\\approx|≈|\\simeq|\\sim\b|\\cong")
SPLIT = re.compile(r"\\implies|\\Longrightarrow|\\Rightarrow|\\iff|\\Leftrightarrow|\\Longleftrightarrow"
                   r"|\\therefore|\\because|\\\\|;|\\text\s*\{\s*(?:and|so|thus|hence|then)\s*\}"
                   r"|\\quad\s*\\text\s*\{[^{}]*\}|\\qquad|\\quad|\\text\s*\{\s*\(?(?:equation|eq\.?)[^{}]*\}")
NEG = re.compile(r"\b(not|no|cannot|can't|isn't|aren't|doesn't|don't|never|except|extraneous|invalid|undefined|"
                 r"impossible|contradiction|incorrect|wrong|mistake|absurd|false)\b|\\neq|≠", re.I)
FUNCSLASH = re.compile(r"\\(?:log|ln|lg|sin|cos|tan|cot|sec|csc)(?:_\{[^{}]*\}|_\w)?\s*(?!\(|\\left|\{|\s)"
                       r"[^=<>(){}]*?/")
SKIP = re.compile(r"remainder|remaining|left over|quotient|integer part|goes into|\\text\s*\{\s*R\s*\}|\\pmod|\\bmod|\\mod\b|\\equiv|\bmod\b", re.I)
HYP = re.compile(r"\b(if|for|when|whenever|suppose|supposing|assum\w*|let|let's|case|test\w*|try\w*|consider\w*|"
                 r"where|with|since|given|because|as|plug\w*|substitut\w*|setting|put\w*|check\w*|verify\w*)\b", re.I)
NUMCOMMA = re.compile(r"(?<![\d.])(\d{1,3}(?:(?:,|\{,\})\d{3})+)(?![\d])")
DELIMS = [("$$", "$$"), ("\\[", "\\]"), ("\\(", "\\)")]
ENVS = re.compile(r"\\begin\{(equation|align|aligned|eqnarray|gather|gathered|split|cases)\*?\}")
CONNECT = r"(?!\s*(?:or|and|if|for|where|when|since|so|then|thus|hence|with|but)\s*\})"
UNIT = r"\\(?:text|mathrm|textrm|mbox)\s*\{" + CONNECT + r"([^{}]*)\}"
AFTER_NUM = r"(?<=[\d}\)])\s*(?:\\[,;! ]\s*)*"
EQLABEL = re.compile(r"^\s*\((?:\d{1,2}[a-z]?|\w{1,3}(?=\)\s*&))\)\s*&?"
                     r"(?!\s*(?:[-+*/=<>^_(×÷·]|\\(?:times|cdot|div|le|ge|lt|gt|neq|ne)\b))")
DEG = r"\s*(?:\^\s*\{?\\circ\}?|°|\\degree)"
DEGATOM = re.compile(r"(\d+(?:\.\d+)?|\\[A-Za-z]+|(?<![A-Za-z\\])[A-Za-z]|\{[^{}]*\}|\([^()]*\))" + DEG)


def _currency(s, i):
    """The "$" at s[i] is a currency sign, not a maths delimiter."""
    if i + 1 >= len(s) or not s[i + 1].isdigit():
        return False
    j = s.find("$", i + 1)
    while j != -1 and s[j - 1] == "\\":
        j = s.find("$", j + 1)
    if j == -1 or (j + 1 < len(s) and s[j + 1].isdigit()):
        return True
    return bool(re.search(r"(?<!\\)\b[A-Za-z]{3,}\b", re.sub(r"\\text\s*\{[^{}]*\}", "", s[i + 1:j])))


def _next_dollar(s, i):
    """Index of the next single "$" that opens maths, or -1."""
    k = s.find("$", i)
    while k != -1 and (k > 0 and s[k - 1] == "\\" or s.startswith("$$", k) or _currency(s, k)):
        k = s.find("$", k + (2 if s.startswith("$$", k) else 1))
    return k


def _opener(s, i):
    """(start, opener, closer) of the earliest maths span from s[i], or None."""
    cands = [(k, o, c) for o, c in DELIMS for k in [s.find(o, i)] if k != -1]
    k = _next_dollar(s, i)
    if k != -1:
        cands.append((k, "$", "$"))
    m = ENVS.search(s, i)
    if m:
        cands.append((m.start(), m.group(0), "\\end{" + m.group(1)))
    return min(cands, key=lambda b: b[0]) if cands else None


def spans(s):
    """[(start, end, inner)] of delimited maths, left to right."""
    out, i = [], 0
    while i < len(s):
        b = _opener(s, i)
        if b is None:
            break
        k, o, c = b
        if o.startswith("\\begin"):
            e = s.find(c, k + len(o))
            e2 = s.find("}", e) if e != -1 else -1
            if e2 == -1:
                break
            out.append((k, e2 + 1, s[k + len(o):e]))
            i = e2 + 1
            continue
        e = s.find(c, k + len(o))
        while e != -1 and c == "$" and s[e - 1] == "\\":
            e = s.find(c, e + 1)
        if e == -1:
            break
        out.append((k, e + len(c), s[k + len(o):e]))
        i = e + len(c)
    return out


def _unwrap(s, cmd):
    """\\cmd{X} -> X (balanced braces)."""
    while True:
        k = s.find(cmd + "{")
        if k == -1:
            return s
        d, j = 0, k + len(cmd)
        for j in range(k + len(cmd), len(s)):
            d += {"{": 1, "}": -1}.get(s[j], 0)
            if d == 0:
                break
        if d != 0:
            return s
        s = s[:k] + s[k + len(cmd) + 1:j] + s[j + 1:]


def _quantity_unit(m):
    """The unit of a member that is one quantity with one unit ('6 \\text{ inches}'), else None."""
    units = re.findall(AFTER_NUM + UNIT, m)
    if len(units) != 1 or re.search(r"\\times|\\cdot|\*|×|\+|(?<=[\w})])\s*-", m):
        return None
    return units[0].strip().lower().rstrip("s") or None


def _conversion(t):
    """Two adjacent "=" members are single quantities in different units (6 in = 1/2 ft)."""
    units = [_quantity_unit(m) for m in t.split("=")]
    return any(a and b and a != b for a, b in zip(units, units[1:]))


def clean_latex(t, literal=False):
    """One maths line normalised for the parser; '' for a unit conversion. literal=True reads
    10\\% as 10 and 30^\\circ as 30 (a percentage or angle written as a plain number)."""
    if _conversion(t):
        return ""
    t = _unwrap(t, "\\boxed")
    t = re.sub(r"\\[dt]frac", r"\\frac", t)
    t = re.sub(r"\\[dt]binom", r"\\binom", t)
    t = re.sub(r"\\tag\*?\{[^{}]*\}|\\label\{[^{}]*\}|\\nonumber|\\notag", "", t)
    t = EQLABEL.sub("", t)   # (1) / (1a) equation labels, not a leading factor like (rst) + 3
    t = t.replace("&", " ")
    if not literal:   # 30^\circ, x^\circ, (8k)^\circ: degrees to radians
        t = DEGATOM.sub(lambda m: "(" + m.group(1) + r" \cdot \frac{\pi}{180})", t)
    t = re.sub(DEG, "", t)
    t = NUMCOMMA.sub(lambda m: m.group(1).replace("{,}", "").replace(",", ""), t)
    t = re.sub(AFTER_NUM + UNIT + r"(?:\^\s*\{?[23]\}?)?", " ", t)   # units, with an optional ^2/^3
    t = re.sub(r"(\d+(?:\.\d+)?)\s*\\?%", r"\1" if literal else r"(\\frac{\1}{100})", t)
    for a, b in (("*", " \\times "), ("×", " \\times "), ("÷", " \\div "), ("·", " \\cdot "), ("−", "-")):
        t = t.replace(a, b)
    t = re.sub(r"\\(?:,|;|!|quad|qquad)|~", " ", t)
    return t.strip().rstrip(".,;:").strip()


def _drop_named(t):
    """Drop "=" members that are only words (\\text{Total cost} = 3 + 4 = 7 -> 3 + 4 = 7)."""
    if re.search(r"[<>]", t) or "=" not in t:
        return t
    keep = [p for p in t.split("=") if not re.fullmatch(r"\s*(\\(?:text|mathrm|textbf|textit)\s*\{[^{}]*\}\s*)+", p)]
    return "=".join(keep) if len(keep) >= 2 else ""


def math_lines(inner):
    """One delimited span -> [(latex, latex_lit or None)] for its parts that state a relation."""
    inner = re.sub(r"(?:\\q?quad|,)?\s*\\text\s*\{\s*(or|and)\s*\}\s*(?:\\q?quad)?", r" \\text{ \1 } ", inner)
    inner = re.sub(r"\\end\{\w+\*?\}|\\begin\{\w+\*?\}(\{[^{}]*\})?", " \\\\ ", inner)
    out = []
    for raw in SPLIT.split(inner):
        raw = APPROX.split(raw)[0]   # only the exact part before ≈
        part, lit = (_drop_named(clean_latex(raw, literal)) for literal in (False, True))
        if FUNCSLASH.search(part):
            continue   # \log_{10} 8 / \log_{10} 2: the parser reads log(8 / log 2)
        if part and REL.search(part) and re.search(r"[0-9A-Za-z]", part):
            out.append((part, lit if lit != part else None))
    return out


# ---------- plain-text arithmetic ----------

CONV = re.compile(r"(?<![\d.,])\d[\d.,]*\s*([A-Za-z]{2,}?)s?\s*=\s*[\d.,/ ()]+?\s*([A-Za-z]{2,}?)s?\b")
STOP = r"(?:of|and|or|is|are|was|to|by|from|for|in|with|times|per|than|the|a|an)"
OPWORDS = {"of", "by", "from", "times", "per", "than", "plus", "minus", "over", "divided", "multiplied", "into"}
TOKEN = re.compile(r"^[\d\s.+\-*/×÷·^()!=<>≤≥xX%]*$")
CLAUSE = re.compile(r"[^;:!?\n⟦⟧]+?(?=(?<!\d)\.(?!\d)|\.\s|[;:!?\n⟦⟧]|,\s|$)")
BARE_NUM = re.compile(r"\s*[\d.]+%?\s*")


def _unconvert(m):
    """Blank the "=" of "2 hours = 120 minutes" when its left side is a single quantity."""
    before = m.string[:m.start()].rstrip()
    if m.group(1).lower() == m.group(2).lower() or (before and before[-1] in "*×/+-÷x") \
            or re.search(r"\b(?:of|by|times)$", before, re.I):
        return m.group(0)
    return m.group(0).replace("=", " ")


def _arith(x):
    """x is plain arithmetic (digits, operators, single letters)."""
    return bool(x.strip()) and bool(TOKEN.match(re.sub(r"(?<![A-Za-z])[A-Za-z](?![A-Za-z])", "0", x))) \
        and bool(re.search(r"[\dA-Za-z]", x))


def _first_member(m):
    """The longest arithmetic suffix of a chain's first member, unless prose makes it an operand."""
    toks = re.split(r"(\s+)", m)
    for k in range(len(toks)):
        cand = "".join(toks[k:])
        if _arith(cand):
            before = "".join(toks[:k]).split()
            if re.match(r"\s*[-*/×÷+^=]", cand) or (before and before[-1].lower().strip(",") in OPWORDS) \
                    or re.fullmatch(r"\s*[\d.]+\s*", cand):
                return ""
            return cand
    return ""


def _bare_member(m, side):
    """The maths part of a plain-text chain member; side is 'first'|'mid'|'last'. '' if none."""
    m = m.strip()
    if re.search(r"\d\s+[\d(]|(?<![A-Za-z])[A-Za-z]\(", m) or (side != "first" and re.search(r"[A-Za-z]", m)):
        return ""   # "9 choose 3" -> "9 3", P(A), letters outside the first member
    if _arith(m):
        return "" if side == "first" and re.fullmatch(r"\s*[\d.]+\s*", m) else m
    if side == "first":
        return _first_member(m)
    if side == "last":
        mm = re.match(r"^\s*[-(]?[\d.]+%?(?:\s*[+\-*/×÷^]\s*[\d.()]+%?)*\)?", m)
        return mm.group(0) if mm and _arith(mm.group(0)) else ""
    return ""


def _normalise_prose(p):
    """Bullets, currency, unit conversions, number commas, units and word labels out of prose."""
    p = re.sub(r"(?m)^\s*(?:[-*•]|\d+\.)\s+", " ", p)
    p = re.sub(r"(?<!\\)\$(?=\s?\d)", " ", p)
    p = CONV.sub(_unconvert, p)
    p = NUMCOMMA.sub(lambda m: m.group(1).replace(",", ""), p)
    p = re.sub(r"\((?:[^()\d]*[A-Za-z]{2,}[^()\d]*)\)", " ", p)                 # (red)
    p = re.sub(r"\b[A-Za-z]{2,}\s*/\s*[A-Za-z]{2,}\b", " ", p)                   # pages/inch
    p = re.sub(r"(?<=[\d.])\s*/\s*[A-Za-z]{2,}\b", " ", p)                       # $50/hour
    p = re.sub(r"(?<=\d)((?:\s+(?!" + STOP + r"\b)[A-Za-z]{2,}\b)+)", " ", p)    # 3 hours
    return re.sub(r"(?<=\d)\s*[xX]\s*(?=\d)", " × ", p)


def _chains(c):
    """Arithmetic chains (strings) in one clause; a rejected member cuts the chain."""
    parts = re.split(r"(=|≤|≥|<|>)", c)
    mems, ops = parts[0::2], parts[1::2]
    out, chain = [], []
    for k, m in enumerate(mems):
        side = "first" if k == 0 else "last" if k == len(mems) - 1 else "mid"
        b = _bare_member(m, side)
        if not b:
            if len(chain) >= 3:
                out.append("".join(chain))
            chain = []
            continue
        if not chain and (BARE_NUM.fullmatch(b) or re.match(r"\s*[-+*/×÷^]", b)):
            continue   # a chain never starts at a bare number or an operator (prose operand)
        if chain:
            chain.append(ops[k - 1])
        chain.append(b)
    if len(chain) >= 3:
        out.append("".join(chain))
    return out


def bare_lines(prose):
    """Plain-text arithmetic ("48/2 = 24") -> [(start, end, raw clause, latex, latex_lit or None)]."""
    res = []
    for cl in CLAUSE.finditer(_normalise_prose(prose)):
        if not re.search(r"[=<>≤≥]", cl.group(0)):
            continue
        for t in _chains(cl.group(0)):
            t, lit = clean_latex(t), clean_latex(t, literal=True)
            if re.search(r"\d", t) and REL.search(t):
                res.append((cl.start(), cl.end(), cl.group(0), t, lit if lit != t else None))
    return res


# ---------- line records ----------

JOINGAP = re.compile(r"\s*,?\s*(?:and|or)?\s*,?\s*")
NONUM = re.compile(r"\d\s+\d|\d\s*\\frac\{\d+\}\{\d+\}|\\(?:star|ast|diamond|oplus|otimes|odot|bullet|circ|triangle|"
                   r"dagger|heartsuit|spadesuit|clubsuit)\b|[#@]|\\arc\w+|\\(?:sin|cos|tan|cot|sec|csc)\s*\^\s*\{?-1"
                   r"|\\log(?!\s*_)")
TRIG = re.compile(r"\\(?:sin|cos|tan|cot|sec|csc)\b")
DEGREE = re.compile(r"\\circ|°|\\degree|degree", re.I)
LEADREL = re.compile(r"^\s*(=|\\leq?\b|\\geq?\b|<|>)")


def _solvar(t):
    """'x' for a solution statement 'x = <no other letters>' (one '='), else None."""
    m = re.fullmatch(r"\s*([A-Za-z](?:_\{?\w+\}?)?)\s*=\s*([^=<>]*?)(?:\s*\\text\{ or \}.*)?", t)
    if not m:
        return None
    rhs = m.group(2).replace("\\frac", "").replace("\\sqrt", "").replace("\\pm", "")
    return None if re.search(r"(?<![a-z\\])[A-Za-z](?![a-z])", rhs) else m.group(1)


def _sentence(s, k, e):
    """(text before the line since the last sentence boundary, text after it to the sentence end)."""
    before = re.split(r"(?<!\d)[.:;!?](?!\d)", s[max(0, k - 200):k])[-1]
    after = re.split(r"(?<!\d)[.;!?](?!\d)|:\s", s[e:e + 200])[0]
    return before, after


def _line(t, pos, end, s, src, raw, lit=None):
    """A line record, or None for remainder / modular arithmetic."""
    before, after = _sentence(s, pos, end)
    if SKIP.search(raw) or SKIP.search(after[:40]):
        return None
    numeric = not NONUM.search(t) and not (TRIG.search(t) and DEGREE.search(s) and not DEGREE.search(raw))
    return {"latex": t, "latex_lit": lit, "pos": pos, "hyp": bool(HYP.search(before)),
            "neg": bool(NEG.search(before + " " + after)), "alt": False, "numeric": numeric, "src": src}


def _joined(lines, new, gap):
    """Merge "x = 7" and "x = -5" separated by a comma/and/or into one solution set; True if merged."""
    if len(new) != 1 or not new[0] or not lines or not lines[-1] or gap is None or not JOINGAP.fullmatch(gap):
        return False
    v = _solvar(lines[-1]["latex"])
    if not v or v != _solvar(new[0]["latex"]):
        return False
    lines[-1] = {**lines[-1], "latex": lines[-1]["latex"] + " \\text{ or } " + new[0]["latex"],
                 "neg": lines[-1]["neg"] or new[0]["neg"]}
    return True


def _mark_alt(lines, new, gap):
    """Adjacent spans joined by , / and / or are a list or a disjunction: never compared across lines."""
    if new and lines and lines[-1] and gap is not None and JOINGAP.fullmatch(gap) \
            and re.search(r",|\b(?:or|and)\b", gap):
        for d in [lines[-1], *new]:
            if d:
                d["alt"] = True


PROSEOP = re.compile(r"(?:[×*/+÷·]|\s-|\\times|\\cdot|\b(?:of|times|by|plus|minus|over)\b)\s*$", re.I)


def _drop_operand(inner, before):
    """A span after "... ×" or "half of" starts with an operand of prose: drop its first member."""
    if not PROSEOP.search(before):
        return inner
    m = REL.search(SPLIT.split(inner)[0])
    return inner[m.end():] if m else inner


def _join_continuations(lines):
    """A line that starts with a relation ("= 10x - 1") continues the previous one, else is dropped."""
    out = []
    for d in lines:
        if LEADREL.match(d["latex"]):
            if out:
                out[-1] = {**out[-1], "latex": out[-1]["latex"] + " " + d["latex"]}
        else:
            out.append(d)
    return out


def extract(step):
    """Ordered line records of one step: {latex, latex_lit, pos, hyp, neg, alt, numeric, src}."""
    s = step.replace("\\$", " ")
    lines, prose, last, prev_e = [], [], 0, None
    for k, e, inner in spans(s):
        inner = _drop_operand(inner, s[last:k])
        new = [_line(t, k, e, s, "math", inner, lit) for t, lit in math_lines(inner)]
        gap = None if prev_e is None else s[prev_e:k]
        if not _joined(lines, new, gap):
            _mark_alt(lines, new, gap)
            lines.extend(new)
        prev_e = e
        prose += [s[last:k], "⟦" + " " * max(0, e - k - 2) + "⟧"]
        last = e
    ptxt = "".join(prose) + s[last:]
    lines += [_line(t, a, b, ptxt, "bare", raw, lit) for a, b, raw, t, lit in bare_lines(ptxt)]
    return _join_continuations(sorted([d for d in lines if d], key=lambda d: d["pos"]))


def is_algebra_step(step):
    """The step yields at least one line the parser reads."""
    return any(parse_graded(d["latex"])[1] != "none" for d in extract(step))


# ---------- checks inside one line ----------

DEC = re.compile(r"^\s*-?\d+\.\d+\s*$")


def _approx_ok(a, b, ta, tb):
    """A failing constant link that is only rounding: a decimal literal with d places the other side
    rounds/truncates to, or a decimal with >= 3 places within relative 10^-(d-1)."""
    try:
        va, vb = float(sympy.N(a, 30)), float(sympy.N(b, 30))
    except (TypeError, ValueError, OverflowError):
        return False
    for v, t in ((va, tb), (vb, ta)):
        if t is not None and DEC.match(t) and abs(v - float(t)) < 10 ** -len(t.strip().split(".")[1]) * (1 + 1e-9):
            return True
    ds = [len(x) for t in (ta, tb) if t for x in re.findall(r"\d\.(\d{3,})", t)]
    return bool(ds) and abs(va - vb) <= 10 ** -(min(ds) - 1) * max(1.0, abs(va), abs(vb))


def _items(obj):
    """Members of Eq(A, B) or an A = B = C chain, else None."""
    if isinstance(obj, sympy.Eq):
        items = [obj.lhs, obj.rhs]
    elif isinstance(obj, sympy.Tuple) and str(obj[0]).startswith("__rel__:") \
            and set(str(obj[0])[8:].split(",")) == {"="}:
        items = list(obj[1:])
    else:
        return None
    return items if all(isinstance(x, sympy.Expr) for x in items) else None


def _show(e):
    try:
        with time_limit(SYMPY_TIMEOUT_S):
            return sympy.sstr(e).replace("**", "^")
    except (Timeout, Exception):
        return str(e)


def within(latex, neg=False):
    """(verdict or None, parse level, explanation): adjacent constant members must be equal; links with
    a symbol (an equation, a substitution, a system) are skipped; an error in a negated sentence is dropped."""
    obj, level = parse_graded(latex)
    if obj is None or checker._read(latex) is None:
        return None, level, None
    items = _items(obj)
    if items is None:
        return None, level, None
    mem = latex.split("=") if not re.search(r"<|>|\\le|\\ge|\\neq|≤|≥", latex) else None
    texts = mem if mem and len(mem) == len(items) else [None] * len(items)
    links = [(a, b, ta, tb) for a, b, ta, tb in zip(items, items[1:], texts, texts[1:])
             if not (a.free_symbols or b.free_symbols)]
    if not links:
        return None, level, None
    bad = None
    try:
        with time_limit(SYMPY_TIMEOUT_S):
            for a, b, ta, tb in links:
                s = checker._same(a, b, [])
                if s is None:
                    return None, level, None
                if not s and not _approx_ok(a, b, ta, tb):
                    bad = (a, b)
                    break
    except (Timeout, Exception):
        return None, level, None
    if bad is None:
        return "valid", level, None
    return (None if neg else "error"), level, f"{_show(bad[0])} and {_show(bad[1])} are not equal"


# ---------- checks across two lines ----------

def _identity(rel):
    """Both sides agree everywhere (a rewrite of an expression, not an equation)."""
    try:
        with time_limit(SYMPY_TIMEOUT_S):
            return bool(checker._same(rel.lhs, rel.rhs, sorted(rel.free_symbols, key=str)))
    except (Timeout, Exception):
        return True   # unknown: never compare across lines


def _is_chain(latex):
    """A = B = C (three or more members): a computation or substitution, not an equation to transform."""
    obj = parse_graded(latex)[0]
    return isinstance(obj, sympy.Tuple) and str(obj[0]).startswith("__rel__:") and len(obj) > 3


def _degree(rel):
    """Total degree of the numerator of L - R (a change marks factors or cases, not a step)."""
    try:
        with time_limit(SYMPY_TIMEOUT_S):
            num = sympy.fraction(sympy.together(rel.lhs - rel.rhs))[0]
            return sympy.Poly(num, *sorted(rel.free_symbols, key=str)).total_degree()
    except (Timeout, Exception):
        return None


FUNCAPP = re.compile(r"(?<![A-Za-z\\])(?:[fgh](?:_\{?\w+\}?)?(?:\^\{?-1\}?)?|[A-Za-z](?:_\{?\w+\}?)?\^\{?-1\}?)"
                     r"\s*(?:\\left)?\(")


def _gate(p, c, pd, cd):
    """A cross-line check is allowed (p, c: checker.read_line dicts; pd, cd: line records); never on
    function application (f(x), f^{-1}(9)), which the parser reads as a product."""
    if FUNCAPP.search(pd["latex"]) or FUNCAPP.search(cd["latex"]):
        return False
    if p is None or c is None or p["bad"] or c["bad"] or cd["hyp"] or pd["neg"] or cd["neg"] \
            or pd["alt"] or cd["alt"] or _is_chain(pd["latex"]) or _is_chain(cd["latex"]):
        return False
    pr, cr = p["rel"], c["rel"]
    if pr is None or p["sol"] is not None:
        return False
    if cr is None:   # a joined solution set "x = 7 or x = -5"
        return type(pr) is sympy.Eq and c["sol"] is not None and pr.free_symbols == {c["sol"][0]} \
            and not _identity(pr)
    if not pr.free_symbols or pr.free_symbols != cr.free_symbols:
        return False
    peq, ceq = type(pr) is sympy.Eq, type(cr) is sympy.Eq
    if peq != ceq or peq and _identity(pr) or ceq and c["sol"] is None and _identity(cr):
        return False
    return c["sol"] is not None or _degree(pr) == _degree(cr)


def _num_g(g):
    """The checker's ratio g is a number (inequality flip, strictness, answer outside the domain)."""
    try:
        with time_limit(SYMPY_TIMEOUT_S):
            return g is not None and bool(sympy.sympify(g).is_number)
    except (Timeout, Exception):
        return False


def _cross_verdict(p, c, r):
    """Map a check_step error: a constant ratio stays an error; inequalities with a varying ratio are
    unchecked; equations compare solution sets (narrower: unchecked, wider: caution, else error)."""
    if r["status"] != "error" or _num_g(r.get("g")):
        return r["status"]
    pr = p["rel"]
    if type(pr) is not sympy.Eq or len(pr.free_symbols) != 1:
        return "unchecked"
    var = next(iter(pr.free_symbols))
    try:
        with time_limit(SYMPY_TIMEOUT_S):
            before = checker._solve(pr, var, sympy.S.Reals)
            after = c["sol"][1] if c["sol"] is not None else checker._solve(c["rel"], var, sympy.S.Reals)
            if before is None or after is None or after.is_subset(before) is not False:
                return "unchecked"
            return "caution" if before.is_subset(after) is not False else "error"
    except (Timeout, Exception):
        return "unchecked"


def _approx_answer(prev_latex, c, latex):
    """"y = -18.83" after "92y = -1732": a stated decimal that a solution of the previous line rounds to."""
    m = re.search(r"=\s*(-?\d+\.\d+)\s*$", latex)
    if not m or c["sol"] is None or c["sol"][0] is None:
        return False
    p = checker._read(prev_latex)
    if p is None or p["rel"] is None:
        return False
    d = len(m.group(1).split(".")[1])
    try:
        with time_limit(SYMPY_TIMEOUT_S):
            sols = checker._solve(p["rel"], c["sol"][0], sympy.S.Reals)
            return isinstance(sols, sympy.FiniteSet) and any(
                abs(float(sympy.N(v, 30)) - float(m.group(1))) < 10 ** -d * (1 + 1e-9) for v in sols)
    except (Timeout, Exception):
        return False


def cross(prev, cur, p, c):
    """(verdict, explanation) of line record cur after prev (p, c their read_line dicts)."""
    r = checker.check_step(prev["latex"], cur["latex"])
    if r["status"] == "error" and _approx_answer(prev["latex"], c, cur["latex"]):
        return "valid", None
    return _cross_verdict(p, c, r), r.get("explanation")


# ---------- a whole solution ----------

def _step_lines(step, i, prev):
    """Check one step's lines; returns (step record, last line state)."""
    st = {"lines": extract(step), "levels": [], "statuses": [], "cross": [], "why": []}
    for d in st["lines"]:
        if d["numeric"]:
            status, level, why = within(d["latex"], d["neg"])
        else:
            status, level, why = None, parse_graded(d["latex"])[1], None
        if status == "error" and d["latex_lit"] and within(d["latex_lit"], d["neg"])[0] == "valid":
            status = "valid"   # 4/40 × 100 = 10%: a percentage written as a number
        st["levels"].append(level)
        if status:
            st["statuses"].append(status)
            if status == "error":
                st["why"].append({"kind": "within", "line": d["latex"], "explanation": why})
        c = checker._read(d["latex"])
        if prev is not None and prev[0] >= i - 1 and _gate(prev[2], c, prev[1], d):
            status, why = cross(prev[1], d, prev[2], c)
            st["cross"].append(status)
            if status == "error":
                st["why"].append({"kind": "cross", "prev": prev[1]["latex"], "line": d["latex"], "explanation": why})
            if status in VERDICTS:
                st["statuses"].append(status)
        prev = (i, d, c)
    return st, prev


def check_steps(steps):
    """Per-step records {lines, levels, statuses, cross, why} for one solution (main thread only)."""
    out, prev = [], None
    for i, s in enumerate(steps):
        st, prev = _step_lines(s, i, prev)
        out.append(st)
    return out


def step_flags(st):
    """(parses, checkable, error) for one step record."""
    return (any(lv != "none" for lv in st["levels"]), any(x in VERDICTS for x in st["statuses"]),
            "error" in st["statuses"])


def predict(records):
    """Index of the first step with an error verdict, or -1."""
    return next((i for i, st in enumerate(records) if step_flags(st)[2]), -1)
