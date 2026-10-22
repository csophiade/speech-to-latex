"""Candidate pools and the selection rules for Systems A-D (single equations).

A pool is built once per utterance (LLM calls and acoustic scores are cached); selection for
any (λ, β) is then a cheap pass over the stored scores, so tuning never re-runs models.
"""

import re
import threading
import time
from concurrent.futures import ThreadPoolExecutor

from src import config
from src.acoustic import mix_scores, phoneme_scores, whisper_scores
from src.candidates import to_latex_timed, to_spoken_timed
from src.eval import normalise
from src.mathparse import parse_graded

LEVEL_BONUS = {"clean": 1.0, "cleaned": 0.5, "none": 0.0}   # × λ


def _transcript_key(text):
    return re.sub(r"[^\w]+", " ", text.lower()).strip()


def fill_levels(pool):
    """(Re)compute each entry's graded validity on the calling (main) thread, where the timeout works."""
    for p in pool:
        p["level"] = parse_graded(p["latex"])[1]
    return pool


def build_pool(nbest, llm=config.LLM_MODEL, workers=8, context_vars=None, augment=None):
    """LaTeX candidates (with spoken forms) from every distinct transcript in the n-best list.

    Returns (pool, timing). Each pool entry: latex, level, src_rank (best beam that produced
    it), src_logprob, cand_rank (position in that beam's LLM answer), spoken (verbalisations).
    timing: seconds of the top-beam LLM call, the slowest to_latex call, and the slowest
    to_spoken call; calls within a stage run concurrently, so a real system waits for the max.
    context_vars: variables already in use (derivation mode), passed to the LLM prompt.
    augment: add confusable-number variants (src/augment.py; no LLM calls); None means
    config.AUGMENT_CONFUSABLE. Variants carry augmented=True.
    """
    unique, seen = [], set()
    for rank, (text, lp) in enumerate(nbest):
        key = _transcript_key(text)
        if key and key not in seen:
            seen.add(key)
            unique.append((rank, text, lp))
    with ThreadPoolExecutor(workers) as ex:
        answers = list(ex.map(lambda u: to_latex_timed(u[1], context_vars, model=llm), unique))

    pool, index = [], {}
    for (rank, _, lp), (cands, _) in zip(unique, answers):
        for j, latex in enumerate(cands):
            key = normalise(latex)
            if key in index:
                continue
            index[key] = len(pool)
            # parse_graded's timeout only works on the main thread: in a worker, leave the level
            # for fill_levels (called by the caller on the main thread)
            level = parse_graded(latex)[1] if threading.current_thread() is threading.main_thread() else None
            pool.append({"latex": latex, "level": level, "src_rank": rank,
                         "src_logprob": lp, "cand_rank": j})

    with ThreadPoolExecutor(workers) as ex:
        spoken = list(ex.map(lambda p: to_spoken_timed(p["latex"], model=llm), pool))
    for p, (forms, _) in zip(pool, spoken):
        p["spoken"] = forms
    if config.AUGMENT_CONFUSABLE if augment is None else augment:
        add_variants(pool)
    timing = {"llm_top_s": answers[0][1] if answers else 0.0,
              "llm_latex_s": max((s for _, s in answers), default=0.0),
              "llm_spoken_s": max((s for _, s in spoken), default=0.0)}
    return pool, timing


def add_variants(pool):
    """Append confusable-number variants to pool in place (level filled on the main thread
    only, as in build_pool); returns the new entries."""
    from src.augment import augment_pool
    new = augment_pool(pool)
    if threading.current_thread() is threading.main_thread():
        for p in new:
            p["level"] = parse_graded(p["latex"])[1]
    pool.extend(new)
    return new


def score_pool(audio, pool, only_missing=False):
    """Attach raw Whisper/phoneme scores for every spoken form (in place).

    only_missing: score only entries without stored scores (migration of cached pools; raw
    scores are per form, independent of the rest of the pool).
    Returns {"whisper_s", "phoneme_s"}: GPU seconds for each scorer (System D uses phoneme only).
    """
    if only_missing:
        pool = [p for p in pool if "whisper" not in p or "phoneme" not in p]
    flat = [s for p in pool for s in p["spoken"]]
    start = time.perf_counter()
    w = whisper_scores(audio, flat) if flat else []
    mid = time.perf_counter()
    ph = phoneme_scores(audio, flat) if flat else []
    end = time.perf_counter()
    i = 0
    for p in pool:
        n = len(p["spoken"])
        p["whisper"], p["phoneme"] = w[i:i + n], ph[i:i + n]
        i += n
    return {"whisper_s": mid - start, "phoneme_s": end - mid}


def acoustic_scores(pool, beta=config.BETA):
    """Per-candidate acoustic score from the raw scores stored by score_acoustics."""
    return mix_scores([s for p in pool for s in p["whisper"]], [s for p in pool for s in p["phoneme"]],
                      [len(p["whisper"]) for p in pool], beta)


def _bonus(pool, lam):
    """Graded validity bonus, applied only if at least one pooled candidate parses."""
    if not any(p["level"] != "none" for p in pool):
        return [0.0] * len(pool)
    return [lam * LEVEL_BONUS[p["level"]] for p in pool]


def system_scores(system, pool, lam=1.0, gamma=0.25, eps=0.05):
    """Per-candidate score of System C or D (see select); select's argmax is over these."""
    base = [p["src_logprob"] for p in pool]
    if system == "D":
        phon = acoustic_scores(pool, beta=0.0)
        base = [b + gamma * a - eps * p["cand_rank"] for b, a, p in zip(base, phon, pool)]
    elif system != "C":
        raise ValueError(system)
    return [b + x for b, x in zip(base, _bonus(pool, lam))]


def select(system, pool, lam=1.0, gamma=0.25, eps=0.05):
    """Index of the chosen candidate in `pool` (None if the pool is empty).

    C: beam log-prob of the source transcript + graded validity bonus (ties: the LLM's ranking).
    D: C + gamma * phoneme-CTC score (z-normalised across the pool's spoken forms)
       - eps * LLM candidate rank. Plain additive: the rank prior restores the LLM's ordering
       among candidates from the same transcript (equal beam scores) without a separate gate;
       the only gate is the δ decision rule (src/decide.py).
    """
    if not pool:
        return None
    top = [i for i, p in enumerate(pool) if p["src_rank"] == min(q["src_rank"] for q in pool)]
    if system == "A":
        return min(top, key=lambda i: pool[i]["cand_rank"])
    if system == "B":
        valid = [i for i in top if pool[i]["level"] != "none"]
        return min(valid or top, key=lambda i: pool[i]["cand_rank"])
    scores = system_scores(system, pool, lam, gamma, eps)
    # ties: better beam, then the LLM's own ordering
    return max(range(len(pool)), key=lambda i: (scores[i], -pool[i]["src_rank"], -pool[i]["cand_rank"]))
