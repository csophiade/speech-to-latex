"""Rule-based 'flat' reading of LaTeX: read left to right, never voicing grouping.

Expressions that differ only in grouping, e.g. (x+3)^2 and x+3^2, get exactly the same
reading, so the acoustic score cannot favour either unless the speaker marked the grouping.
"""

import re

from src.eval import _tokens

WORDS = {
    "+": "plus", "-": "minus", "=": "equals", "<": "less than", ">": "greater than",
    "/": "over", "!": "factorial", "'": "prime", ",": ",", ":": ":", ";": ",",
    r"\leq": "less than or equal to", r"\geq": "greater than or equal to", r"\neq": "not equal to",
    r"\approx": "approximately", r"\cdot": "times", r"\times": "times", r"\div": "divided by",
    r"\pm": "plus or minus", r"\mp": "minus or plus", r"\infty": "infinity", r"\to": "to",
    r"\in": "in", r"\notin": "not in", r"\subset": "subset of", r"\subseteq": "subset of",
    r"\cup": "union", r"\cap": "intersection", r"\sum": "sum", r"\prod": "product",
    r"\int": "integral", r"\lim": "limit", r"\partial": "partial", r"\nabla": "nabla",
    r"\sin": "sine", r"\cos": "cosine", r"\tan": "tangent", r"\ln": "natural log", r"\log": "log",
    r"\exp": "exp", r"\dots": "dot dot dot", r"\equiv": "equivalent to", r"\sim": "tilde",
    r"\propto": "proportional to", r"\forall": "for all", r"\exists": "there exists",
    r"\emptyset": "empty set", r"\perp": "perpendicular", r"\{": "", r"\}": "", r"\|": "norm",
}
SILENT = {"(", ")", "[", "]", r"\left", r"\right", r"\displaystyle", r"\limits", "&", "\\\\"}
FONTS = {r"\mathcal", r"\mathbb", r"\mathbf", r"\mathrm", r"\mathit", r"\boldsymbol", r"\bm",
         r"\mathscr", r"\mathfrak", r"\mathsf", r"\operatorname", r"\text", r"\textrm", r"\textbf"}
ACCENTS = {r"\bar": "bar", r"\overline": "bar", r"\hat": "hat", r"\widehat": "hat", r"\tilde": "tilde",
           r"\widetilde": "tilde", r"\vec": "vector", r"\dot": "dot", r"\ddot": "double dot"}
POWERS = {"2": "squared", "3": "cubed"}


def _tree(toks):
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
    return parse(0)[0]


def _read(nodes):
    words, i, in_abs = [], 0, False

    def arg():
        nonlocal i
        if i >= len(nodes):
            return []
        node = nodes[i]
        i += 1
        return _read(node) if isinstance(node, list) else _read([node])

    while i < len(nodes):
        t = nodes[i]
        i += 1
        if isinstance(t, list):
            words += _read(t)
        elif t == r"\frac":
            a = arg()
            words += a + ["over"] + arg()
        elif t == r"\sqrt":
            words += ["square root of"] + arg()
        elif t == "^":
            a = arg()
            words += [POWERS[a[0]]] if len(a) == 1 and a[0] in POWERS else ["to the power of"] + a
        elif t == "_":
            words += ["sub"] + arg()
        elif t in FONTS:
            words += arg()
        elif t in ACCENTS:
            words += arg() + [ACCENTS[t]]
        elif t == "|":
            words += [] if in_abs else ["absolute value of"]
            in_abs = not in_abs
        elif t in SILENT:
            continue
        elif t in WORDS:
            words.append(WORDS[t])
        elif t.startswith("\\"):
            words.append(t[1:])
        else:
            words.append(t)
    return words


def flat_reading(latex):
    """Plain left-to-right reading of a LaTeX string ('' if nothing readable)."""
    toks = _tokens((latex or "").strip().strip("$"))
    # join digit runs back into numbers: 1 7 . 1 2 4 -> 17.124
    merged = []
    for t in toks:
        if merged and re.fullmatch(r"[\d.]", t) and re.fullmatch(r"\d[\d.]*|\.", merged[-1]):
            merged[-1] += t
        else:
            merged.append(t)
    text = " ".join(w for w in _read(_tree(merged)) if w)
    return re.sub(r"\s+([,:])", r"\1", re.sub(r"\s+", " ", text)).strip()
