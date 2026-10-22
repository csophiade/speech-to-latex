"""Per-line decision rule (Phase 4.1): verified / corrected / ask_back / math_error / unchecked.

Candidates are dicts {latex, delta, status, explanation[, cls]}: delta = D(ĉ) − D(c) ≥ 0 under the
frozen System D score, status from checker.check_step(prev_out, c). ĉ is the smallest-delta
candidate (first one on ties). A candidate is valid iff its status is "valid" or "caution";
"unchecked" is never treated as valid. Rules, in order (line 0 is always "given"):
  1. ĉ valid -> "verified" (a caution explanation is kept as a note; caution is never a flag).
  2. valid candidates with delta < δ form exactly one SymEq class -> "corrected" (smallest delta).
  4. they form ≥ 2 classes -> "ask_back" with the smallest-delta member of the two best classes.
  3. otherwise keep ĉ: "math_error" (checker explanation) if ĉ is an error, else "unchecked".
For ask_back the output is ĉ until the user picks an option. Flags: math_error and ask_back.

Class-conditional (Mondrian) thresholds: a candidate carrying "hom_class": True (a single
confusable-number swap of ĉ, see is_homophone_of / tag_homophone) is compared with delta_hom
instead of δ when delta_hom is given. With delta_hom=None the rule is unchanged.
"""

from src.augment import latex_swaps
from src.eval import normalise, symeq

VALID = {"valid", "caution"}
FLAGS = {"math_error", "ask_back"}


def is_valid(status):
    return status in VALID


def classes(latexes):
    """SymEq class id per latex (union of 'equal' pairs; identical normal forms always merge)."""
    ids = list(range(len(latexes)))

    def root(i):
        while ids[i] != i:
            i = ids[i]
        return i

    for i in range(len(latexes)):
        for j in range(i):
            if root(i) == root(j):
                continue
            if normalise(latexes[i]) == normalise(latexes[j]) or symeq(latexes[j], latexes[i]) == "equal":
                ids[root(i)] = root(j)
    return [root(i) for i in range(len(latexes))]


def _with_classes(cands):
    if all("cls" in c for c in cands):
        return cands
    return [dict(c, cls=k) for c, k in zip(cands, classes([c["latex"] for c in cands]))]


def _out(status, output, heard, explanation=None, note=None, options=None):
    return {"status": status, "output": output, "heard": heard, "options": options or [],
            "explanation": explanation, "note": note}


def _note(c):
    return c.get("explanation") if c["status"] == "caution" else None


def is_homophone_of(c_hat_latex, c_latex):
    """True iff c is one confusable-number swap of ĉ (src/augment rules), compared by normalise()."""
    target = normalise(c_latex)
    return any(normalise(new) == target for new, *_ in latex_swaps(c_hat_latex))


def tag_homophone(cands):
    """Copies of cands with "hom_class" set by the runtime class rule against ĉ (smallest delta)."""
    hat = min(cands, key=lambda c: c["delta"])["latex"]
    return [dict(c, hom_class=is_homophone_of(hat, c["latex"])) for c in cands]


def _threshold(c, delta, delta_hom):
    return delta_hom if delta_hom is not None and c.get("hom_class") else delta


def decide(cands, delta, line_index, delta_hom=None):
    """Apply the decision rule to one line's candidate pool at threshold δ (delta_hom for
    homophone-class candidates when given)."""
    order = sorted(range(len(cands)), key=lambda i: cands[i]["delta"])   # stable: ties keep pool order
    hat = cands[order[0]]
    heard = hat["latex"]
    if line_index == 0:
        return _out("given", heard, heard)
    if is_valid(hat["status"]):
        return _out("verified", heard, heard, note=_note(hat))
    close = _with_classes([cands[i] for i in order
                           if is_valid(cands[i]["status"])
                           and cands[i]["delta"] < _threshold(cands[i], delta, delta_hom)])
    best = {}
    for c in close:                       # sorted by delta: first member of each class is its best
        best.setdefault(c["cls"], c)
    if len(best) == 1:
        c = close[0]
        return _out("corrected", c["latex"], heard, explanation=hat.get("explanation"), note=_note(c))
    if len(best) >= 2:
        a, b = list(best.values())[:2]
        return _out("ask_back", heard, heard, explanation=hat.get("explanation"),
                    options=[a["latex"], b["latex"]])
    status = "math_error" if hat["status"] == "error" else "unchecked"
    return _out(status, heard, heard, explanation=hat.get("explanation"))
