"""Homophone stress set (Phase 4 follow-up): the spoken error is the sound-alike of the correct number.

Each item is a VALID derivation from src.derivgen (same verified builders) in which one confusable
number on line t >= 1 is replaced by its confusion partner (13<->30, ..., 19<->90): the student SAYS
"fifteen" where "fifty" is right. The LaTeX ground truth is what was said, so line t is a genuine
wrong step (its solution set differs from line t-1's, checked with solveset) while the original
line t is valid. The derivation ends at line t: no "follows_error" lines, so the item tests exactly
one decision (correct silently vs flag) and nothing downstream depends on the swapped number.

Swap sites are whole-number tokens only: not part of a longer number or a decimal, and not a
subscript or exponent (x_{13}, x^{13}); a number inside \\frac{..}{..} or \\sqrt{..} counts (it is
read aloud as a number). "a"<->8 is not used here (letters are not confusable numbers in this data).
"""

import random
import re
from collections import Counter

from src.config import SEED
from src.derivgen import PREFIX, TYPES, Reject, make_item, solution_set, domain_of, step_labels, verify  # noqa: F401
from src.spoken_norm import number_words

TEENS = {13: 30, 14: 40, 15: 50, 16: 60, 17: 70, 18: 80, 19: 90}
CONFUSION = {**TEENS, **{v: k for k, v in TEENS.items()}}
# a run of digits not preceded by a digit, '.', '^', '_' or '^{' / '_{', and not followed by a digit or '.'
_TOKEN = re.compile(r"(?<![\d.^_])(?<!\^\{)(?<!_\{)(\d+)(?![\d.])")
PER_TYPE_SPLIT = 20   # 5 types x 20 = 100 calib + 100 test


def swap_sites(latex):
    """[(start, end, value)] for each whole-number token with a confusion partner."""
    return [(m.start(1), m.end(1), int(m.group(1))) for m in _TOKEN.finditer(latex)
            if int(m.group(1)) in CONFUSION]


def swap(latex, site):
    s, e, v = site
    return latex[:s] + str(CONFUSION[v]) + latex[e:]


def homophone_from(base, t, site):
    """The homophone item for base derivation `base` (valid), swap `site` on line t; None if the
    swapped line is not a genuine wrong step (solution set unchanged) or cannot be read."""
    lines = base["lines"]
    said = swap(lines[t], site)
    try:
        dom = domain_of(lines[0])
        prev, orig, new = (solution_set(s, dom) for s in (lines[t - 1], lines[t], said))
    except (ValueError, Exception):   # unreadable / sympy timeout: skip this site
        return None
    if orig != prev or new == prev:
        return None
    correct = site[2]
    out = lines[:t] + [said]
    return {"type": base["type"], "lines": out,
            "step_valid": [k != t for k in range(t + 1)],
            "step_label": step_labels(t + 1, t),
            "error": {"type": "homophone", "line": t},
            "solution": base["solution"], "set": "homophone",
            "homophone": {"correct": correct, "said": CONFUSION[correct],
                          "correct_word": number_words(correct), "said_word": number_words(CONFUSION[correct]),
                          "correct_line": lines[t],
                          "position": "final" if t == len(lines) - 1 else "intermediate",
                          "orig_n_lines": len(lines),
                          "direction": "teen_said_for_ty" if correct >= 30 else "ty_said_for_teen"}}


def generate(seed=SEED, per_type_split=PER_TYPE_SPLIT, exclude_line0=(), max_bases=6000):
    """~2 * per_type_split items per type, stratified over (type, position) and split calib/test.

    For each type, valid base derivations are drawn (rng seeded from `seed`) until there are enough
    swap candidates; each base contributes at most one item, its problem (line 0) is new (not in
    `exclude_line0`, e.g. the 400 Phase 3 derivations). Within a type we aim for half final-line and
    half intermediate-line swaps, and half each direction, filling from what exists when a cell is short."""
    rng = random.Random(f"homophone-{seed}")
    seen0 = set(exclude_line0)
    items = []
    need = 2 * per_type_split
    for dtype in TYPES:
        cands = {}   # (position, direction) -> [item]
        bases = 0
        while bases < max_bases:
            have = Counter({k: len(v) for k, v in cands.items()})
            if _enough(have, need, bases):
                break
            base = make_item(rng, dtype, None)
            bases += 1
            if base["lines"][0] in seen0:
                continue
            sites = [(t, s) for t in range(1, len(base["lines"])) for s in swap_sites(base["lines"][t])]
            rng.shuffle(sites)
            for t, s in sites:
                it = homophone_from(base, t, s)
                if it:
                    key = (it["homophone"]["position"], it["homophone"]["direction"])
                    if len(cands.get(key, [])) < need:   # keep the scarce cells, cap the common ones
                        cands.setdefault(key, []).append(it)
                        seen0.add(base["lines"][0])
                    break
        print(f"{dtype}: {bases} bases, candidates {dict(Counter({k: len(v) for k, v in cands.items()}))}", flush=True)
        chosen = _balance(cands, need, rng)
        for stratum in _group(chosen, lambda it: it["homophone"]["position"]):
            rng.shuffle(stratum)
            for j, it in enumerate(stratum):
                it["split"] = "calib" if j % 2 == 0 else "test"
        # even split sizes per type: move items between splits if the position strata were odd
        _even_splits(chosen)
        items += chosen
    rng.shuffle(items)
    out = []
    for i, it in enumerate(sorted(items, key=lambda it: (it["split"], it["type"]))):
        out.append({"id": f"hom-{i:04d}", "split": it["split"], "type": it["type"], "lines": it["lines"],
                    "step_valid": it["step_valid"], "step_label": it["step_label"], "error": it["error"],
                    "solution": it["solution"], "set": "homophone", "homophone": it["homophone"]})
    return out


def _enough(have, need, bases):
    """Stop drawing once every (position, direction) cell has a quarter of `need`, or the pool is
    generous (3x need; cells are capped at `need`), or after 1500 bases with at least `need` candidates
    (some cells barely occur, e.g. a final answer of 30+ in a quadratic)."""
    q = need // 4
    full = all(have[(p, d)] >= q for p in ("final", "intermediate")
               for d in ("teen_said_for_ty", "ty_said_for_teen"))
    total = sum(have.values())
    return full or total >= 3 * need or (bases >= 1500 and total >= need)


def _balance(cands, need, rng):
    """Pick `need` items round-robin over the (position, direction) cells (scarce cells first)."""
    pools = {k: list(v) for k, v in cands.items()}
    for v in pools.values():
        rng.shuffle(v)
    order = sorted(pools, key=lambda k: (len(pools[k]), k))
    chosen = []
    while len(chosen) < need and any(pools.values()):
        for k in order:
            if pools[k] and len(chosen) < need:
                chosen.append(pools[k].pop())
    if len(chosen) < need:
        raise RuntimeError(f"only {len(chosen)} homophone candidates for {chosen and chosen[0]['type']}")
    return chosen


def _group(items, key):
    groups = {}
    for it in items:
        groups.setdefault(key(it), []).append(it)
    return [groups[k] for k in sorted(groups)]


def _even_splits(items):
    calib = [it for it in items if it["split"] == "calib"]
    test = [it for it in items if it["split"] == "test"]
    while len(calib) > len(test) + 1:
        it = calib.pop()
        it["split"] = "test"
        test.append(it)
    while len(test) > len(calib) + 1:
        it = test.pop()
        it["split"] = "calib"
        calib.append(it)


def check(item):
    """Ground-truth checks for one homophone item (used by tests and the generator script)."""
    h, t, lines = item["homophone"], item["error"]["line"], item["lines"]
    assert t >= 1 and t == len(lines) - 1
    assert CONFUSION[h["correct"]] == h["said"]
    dom = domain_of(lines[0])
    prev, wrong, orig = (solution_set(s, dom) for s in (lines[t - 1], lines[t], h["correct_line"]))
    assert orig == prev, "original line t must be a valid step"
    assert wrong != prev, "said line t must be a wrong step"
    # the said line is the original with exactly one confusable token swapped to its partner
    sites = [s for s in swap_sites(h["correct_line"]) if s[2] == h["correct"] and swap(h["correct_line"], s) == lines[t]]
    assert sites, "said line must differ from the original by one homophone swap"
    assert verify({"lines": lines, "step_valid": item["step_valid"], "solution": item["solution"]})
    return True
