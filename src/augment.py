"""Confusable-number pool augmentation (Phase 4, part of the shipped system).

Whisper and the LLM tend to agree on one reading of a "-teen"/"-ty" number, so the candidate
pool rarely holds the other one and the decision rule has nothing valid to switch to. For
every pool candidate we add variants that swap ONE number at a time through CONFUSION
(13<->30, ..., 19<->90). Variants are ordinary candidates, scored like any other:

  - src_rank / src_logprob inherited from the parent; cand_rank = parent cand_rank + 1, so the
    LLM's own reading wins an exact acoustic tie;
  - augmented=True, aug_parent (parent LaTeX), aug_swap ("15->50");
  - deduplicated against the pool (and earlier variants) by src.eval.normalise; parents are
    visited in pool order, so a duplicate keeps the best parent's ranks;
  - spoken forms without an LLM call: the rule-based flat_reading of the new LaTeX, then the
    parent's spoken forms with the corresponding number swapped (forms where the
    correspondence is ambiguous are dropped), capped at SPOKEN_K like candidates.to_spoken.
  - level is NOT set here (parse timeouts only work on the main thread): callers run
    rerank.fill_levels, as for every other pool entry.

Boundary rules (LaTeX), a number is a maximal digit run that is
  - not part of a longer number: no digit, '.', or ',' + digit on either side (so 150, 1.5,
    0.15, 15.5 and 1,500 are never touched);
  - not a subscript index: inside _{...} or directly after _ (x_{15}, a_1 are labels, not
    quantities; swapping them would rename a variable, which is not the acoustic confusion);
  - not an unbraced single-digit argument: after ^, \\frac, \\dfrac, \\tfrac, \\sqrt with no brace
    (\\frac15 is 1/5 and x^15 is x^1 5 in TeX, so the digits are not one number).
  Exponents in braces (x^{13}, 10^{15}) ARE swapped: they are said as numbers.

Spoken text: digit tokens (same boundary rule, "15x" allowed) and number words thirteen..nineteen,
thirty..ninety, plus ordinals (fifteenth<->fiftieth). A word is skipped when it is part of a
compound: hyphenated ("fifty-five"), followed by a ones word / hundred / thousand / point, or
preceded by hundred / thousand / and / point ("one hundred fifty", "fifteen hundred"). Limits:
compounds are never swapped (none is a member of the map), "a fifty" vs "eight" is not handled.
Occurrences are matched by index: the k-th whole-number occurrence of N in the LaTeX
(subscripts included, so "x sub 15" keeps the alignment) maps to the k-th occurrence of N in a
spoken form; a form whose count of N differs from the LaTeX's is dropped.

"a" <-> 8 is NOT included: "a" is also an article and a common parameter name, and swapping a
letter for a digit changes the variable set, so it is neither cheap nor safe.
"""

import re

from src.eval import normalise
from src.verbalise import flat_reading

_PAIRS = [(13, 30), (14, 40), (15, 50), (16, 60), (17, 70), (18, 80), (19, 90)]
CONFUSION = {**{a: b for a, b in _PAIRS}, **{b: a for a, b in _PAIRS}}
SPOKEN_K = 3

_TEENS = "thirteen fourteen fifteen sixteen seventeen eighteen nineteen".split()
_TENS = "thirty forty fifty sixty seventy eighty ninety".split()
_TEENTH = [w + "th" for w in _TEENS]
_TIETH = "thirtieth fortieth fiftieth sixtieth seventieth eightieth ninetieth".split()
WORD_VALUE = {}
for _i, (_a, _b) in enumerate(_PAIRS):
    WORD_VALUE[_TEENS[_i]] = (_a, "card")
    WORD_VALUE[_TENS[_i]] = (_b, "card")
    WORD_VALUE[_TEENTH[_i]] = (_a, "ord")
    WORD_VALUE[_TIETH[_i]] = (_b, "ord")
VALUE_WORD = {(v, kind): w for w, (v, kind) in WORD_VALUE.items()}

_ONES = set("one two three four five six seven eight nine".split())
_AFTER_BLOCK = _ONES | {"hundred", "thousand", "million", "point", "hundredth", "thousandth"}
_BEFORE_BLOCK = {"hundred", "thousand", "million", "and", "point"}

_NUM = re.compile(r"(?<![\d.])(?<!\d,)\d+(?!\d|[.,]\d)")
_WORD = re.compile(r"[a-z]+", re.I)


# ---------- LaTeX ----------

def _in_subscript(latex, pos):
    """True if position pos is inside a _{...} group (any depth) or right after a bare _."""
    stack = []   # for each open brace: is it a subscript group?
    i = 0
    while i < pos:
        ch = latex[i]
        if ch == "\\":
            i += 2
            continue
        if ch == "{":
            j = i - 1
            while j >= 0 and latex[j].isspace():
                j -= 1
            stack.append(j >= 0 and latex[j] == "_")
        elif ch == "}" and stack:
            stack.pop()
        i += 1
    if any(stack):
        return True
    return latex[:pos].rstrip().endswith("_")


def _bare_argument(latex, pos):
    """Digits directly after ^ / \\frac / \\sqrt without a brace are single-token arguments."""
    before = latex[:pos].rstrip()
    return bool(re.search(r"(\^|\\[dt]?frac|\\sqrt)$", before))


def latex_numbers(latex):
    """Every whole-number occurrence: list of (start, end, value, swappable)."""
    out = []
    for m in _NUM.finditer(latex):
        value = int(m.group())
        ok = (m.group()[0] != "0" or m.group() == "0") and value in CONFUSION \
            and not _in_subscript(latex, m.start()) and not _bare_argument(latex, m.start())
        out.append((m.start(), m.end(), value, ok))
    return out


def latex_swaps(latex):
    """[(new_latex, value, partner, k)]: one swap per swappable occurrence; k = index of that
    occurrence among all whole-number occurrences of `value` (for aligning spoken forms)."""
    nums = latex_numbers(latex)
    out = []
    for idx, (s, e, v, ok) in enumerate(nums):
        if not ok:
            continue
        k = sum(1 for x in nums[:idx] if x[2] == v)
        out.append((latex[:s] + str(CONFUSION[v]) + latex[e:], v, CONFUSION[v], k))
    return out


# ---------- spoken text ----------

def spoken_numbers(text):
    """Every standalone occurrence of a confusable number in spoken text:
    list of (start, end, value, form) with form 'digit', 'card' or 'ord'.
    Digit runs of any value are listed (so non-map numbers keep their place); words only
    when not part of a compound."""
    out = []
    for m in _NUM.finditer(text):
        out.append((m.start(), m.end(), int(m.group()), "digit"))
    words = [(m.start(), m.end(), m.group().lower()) for m in _WORD.finditer(text)]
    for i, (s, e, w) in enumerate(words):
        if w not in WORD_VALUE:
            continue
        if (s > 0 and text[s - 1] == "-") or (e < len(text) and text[e] == "-" and e + 1 < len(text)
                                               and text[e + 1].isalpha()):
            continue
        nxt = words[i + 1][2] if i + 1 < len(words) else None
        prv = words[i - 1][2] if i > 0 else None
        if nxt in _AFTER_BLOCK or prv in _BEFORE_BLOCK:
            continue
        v, kind = WORD_VALUE[w]
        out.append((s, e, v, kind))
    return sorted(out)


def _match_case(src, word):
    if src.isupper():
        return word.upper()
    if src[:1].isupper():
        return word[:1].upper() + word[1:]
    return word


def swap_spoken(text, value, partner, k):
    """Swap the k-th occurrence of `value` in `text` for `partner` (digits stay digits, words
    stay words). None if the text does not contain exactly the expected occurrence."""
    occ = [x for x in spoken_numbers(text) if x[2] == value]
    if k >= len(occ):
        return None
    s, e, _, form = occ[k]
    new = str(partner) if form == "digit" else _match_case(text[s:e], VALUE_WORD[(partner, form)])
    return text[:s] + new + text[e:]


def count_value(latex, value):
    return sum(1 for x in latex_numbers(latex) if x[2] == value)


# ---------- pools ----------

def variant_forms(parent_latex, parent_spoken, new_latex, value, partner, k, limit=SPOKEN_K):
    """Spoken forms of a variant: flat reading of new_latex, then the parent's forms swapped.
    A parent form is used only if it holds as many occurrences of `value` as the LaTeX."""
    n = count_value(parent_latex, value)
    forms = []
    flat = flat_reading(new_latex)
    if flat:
        forms.append(flat)
    for f in parent_spoken or []:
        if sum(1 for x in spoken_numbers(f) if x[2] == value) != n:
            continue
        g = swap_spoken(f, value, partner, k)
        if g:
            forms.append(g)
    return list(dict.fromkeys(forms))[:limit]


def augment_pool(pool):
    """New candidate dicts (not yet in pool; no level, no acoustic scores) for every
    single-number confusion swap of every non-augmented pool candidate."""
    seen = {normalise(p["latex"]) for p in pool}
    new = []
    for p in pool:
        if p.get("augmented"):
            continue   # one swap per variant: never augment a variant
        for latex, v, partner, k in latex_swaps(p["latex"]):
            key = normalise(latex)
            if key in seen:
                continue
            seen.add(key)
            new.append({"latex": latex, "level": None, "src_rank": p["src_rank"],
                        "src_logprob": p["src_logprob"], "cand_rank": p["cand_rank"] + 1,
                        "spoken": variant_forms(p["latex"], p.get("spoken"), latex, v, partner, k),
                        "augmented": True, "aug_parent": p["latex"], "aug_swap": f"{v}->{partner}"})
    return new
