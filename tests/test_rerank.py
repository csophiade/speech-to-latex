from src.rerank import select


def cand(latex, level, src_rank, src_logprob, cand_rank, whisper=(0.0,), phoneme=(0.0,)):
    return {"latex": latex, "level": level, "src_rank": src_rank, "src_logprob": src_logprob,
            "cand_rank": cand_rank, "spoken": [latex] * len(whisper),
            "whisper": list(whisper), "phoneme": list(phoneme)}


POOL = [
    cand(r"\text{bitter}", "none", 0, -0.2, 0, whisper=(-3.0,), phoneme=(-3.0,)),
    cand("b", "clean", 0, -0.2, 1, whisper=(-2.0,), phoneme=(-2.5,)),
    cand(r"\beta", "cleaned", 1, -0.5, 0, whisper=(-0.5,), phoneme=(-0.4,)),
]


def test_a_takes_top_beam_first_candidate():
    assert select("A", POOL) == 0


def test_b_takes_first_parseable_from_top_beam():
    assert select("B", POOL) == 1
    assert select("B", [POOL[0]]) == 0          # nothing parses: fall back to A


def test_c_graded_bonus():
    # with λ = 1: b scores -0.2 + 1, \beta scores -0.5 + 0.5
    assert select("C", POOL, lam=1.0) == 1


def test_d_is_c_plus_phoneme():
    # strong phoneme evidence for \beta overturns C's choice; weak weight leaves C's choice
    assert select("D", POOL, lam=0.5, gamma=1.0, eps=0.01) == 2
    assert select("D", POOL, lam=1.0, gamma=0.0, eps=0.01) == select("C", POOL, lam=1.0)


def test_d_rank_prior_keeps_llm_order_against_weak_acoustics():
    # same transcript (equal beam scores), slightly better phoneme score for the LLM's 2nd choice
    pool = [cand("a", "clean", 0, -0.2, 0, phoneme=(-1.00,)), cand("b", "clean", 0, -0.2, 1, phoneme=(-0.98,)),
            cand("c", "clean", 0, -0.2, 2, phoneme=(-3.0,))]
    assert select("D", pool, lam=1.0, gamma=0.1, eps=0.25) == 0
    assert select("D", pool, lam=1.0, gamma=0.1, eps=0.0) == 1


def test_d_shared_plain_reading_leaves_tie_to_c():
    # (x+3)^2 vs x+3^2: same plain reading, the speaker did not mark grouping -> acoustics tie,
    # so D falls back to C's evidence (the beam/LLM ordering)
    plain = -0.5
    pool = [cand("(x+3)^2", "clean", 0, -0.2, 0, whisper=(plain, -2.0), phoneme=(plain, -2.0)),
            cand("x+3^2", "clean", 0, -0.2, 1, whisper=(plain, -2.5), phoneme=(plain, -2.5))]
    assert select("D", pool, lam=1.0, gamma=1.0, eps=0.01) == select("C", pool, lam=1.0) == 0


def test_bonus_only_when_something_parses():
    pool = [cand("a", "none", 0, -0.1, 0), cand("b", "none", 1, -0.9, 0)]
    assert select("C", pool, lam=4.0) == 0


def test_empty_pool():
    assert select("D", []) is None
