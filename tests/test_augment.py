"""Confusable-number pool augmentation (src/augment.py, rerank integration)."""

from src import config
from src.augment import CONFUSION, augment_pool, latex_swaps, spoken_numbers, swap_spoken, variant_forms
from src.eval import normalise
from src.rerank import add_variants, build_pool


def cand(latex, spoken, src_rank=0, lp=-0.3, cand_rank=0, **kw):
    return {"latex": latex, "level": "clean", "src_rank": src_rank, "src_logprob": lp,
            "cand_rank": cand_rank, "spoken": spoken, **kw}


def swapped(latex):
    return sorted(s[0] for s in latex_swaps(latex))


def test_map_is_symmetric_and_complete():
    assert all(CONFUSION[CONFUSION[k]] == k for k in CONFUSION)
    assert {k: CONFUSION[k] for k in range(13, 20)} == {13: 30, 14: 40, 15: 50, 16: 60, 17: 70, 18: 80, 19: 90}
    assert len(CONFUSION) == 14


def test_whole_number_tokens_only():
    assert swapped("x=15") == ["x=50"]
    assert swapped("x=150") == []              # part of a longer number
    assert swapped("x=1.5") == [] and swapped("x=15.5") == [] and swapped("x=0.15") == []
    assert swapped("x=1,500") == []
    assert swapped("x_{15}=3") == [] and swapped("x_15=3") == [] and swapped("a_{1}+b_{2 15}=0") == []
    assert swapped(r"\frac15") == [] and swapped("x^15") == []    # unbraced TeX arguments are single digits
    assert swapped(r"\frac{15}{2}") == [r"\frac{50}{2}"]
    assert swapped("x^{13}=1") == ["x^{30}=1"]                   # braced exponents are said as numbers
    assert swapped("15x - 30 = 15") == ["15x - 13 = 15", "15x - 30 = 50", "50x - 30 = 15"]


def test_one_swap_at_a_time():
    assert len(latex_swaps("x+15=15")) == 2
    assert all(new.count("50") == 1 for new, *_ in latex_swaps("x+15=15"))


def test_spoken_swaps_digits_and_words():
    assert swap_spoken("x equals fifteen", 15, 50, 0) == "x equals fifty"
    assert swap_spoken("x equals 15", 15, 50, 0) == "x equals 50"
    assert swap_spoken("Fifteen x equals two", 15, 50, 0) == "Fifty x equals two"
    assert swap_spoken("x to the fifteenth", 15, 50, 0) == "x to the fiftieth"
    assert swap_spoken("forty x plus forty", 40, 14, 1) == "forty x plus fourteen"
    assert swap_spoken("15x equals 4", 15, 50, 0) == "50x equals 4"


def test_spoken_compounds_untouched():
    for text in ("fifty five", "fifty-five", "one hundred fifty", "fifteen hundred", "one point fifteen",
                 "fifteen point five", "150", "1.5"):
        assert [x for x in spoken_numbers(text) if x[2] in CONFUSION] == [], text


def test_variant_forms_align_and_drop_ambiguous():
    forms = variant_forms("x+15=15", ["x plus 15 equals 15", "x plus fifteen equals fifteen", "x plus fifteen"],
                          "x+15=50", 15, 50, 1)
    assert forms[0] == "x plus 15 equals 50"                          # flat reading of the new LaTeX
    assert "x plus fifteen equals fifty" in forms
    assert all(f != "x plus fifteen" and "fifty" not in f.split()[:3] for f in forms)   # count mismatch dropped


def test_dedup_and_rank_inheritance():
    pool = [cand("x=15", ["x equals 15", "x equals fifteen"], src_rank=0, lp=-0.1, cand_rank=0),
            cand("x=50", ["x equals 50"], src_rank=2, lp=-0.9, cand_rank=0),          # already pooled
            cand("y=14", ["y equals fourteen"], src_rank=1, lp=-0.5, cand_rank=2)]
    new = augment_pool(pool)
    assert [n["latex"] for n in new] == ["y=40"]         # x=50 exists, not duplicated
    v = new[0]
    assert (v["src_rank"], v["src_logprob"], v["cand_rank"]) == (1, -0.5, 3)
    assert v["augmented"] and v["aug_parent"] == "y=14" and v["aug_swap"] == "14->40"
    assert "y equals forty" in v["spoken"]
    assert normalise(v["latex"]) not in {normalise(p["latex"]) for p in pool}


def test_variants_not_augmented_again_and_duplicates_keep_best_parent():
    pool = [cand("x=15", ["x equals fifteen"], src_rank=0, lp=-0.1, cand_rank=1),
            cand("x = 15", ["x equals fifteen"], src_rank=3, lp=-2.0, cand_rank=0)]
    new = augment_pool(pool)
    assert len(new) == 0 or (len(new) == 1 and new[0]["src_rank"] == 0)
    pool2 = [cand("x=15", ["x equals fifteen"]), cand("x=16", ["x equals sixteen"], cand_rank=1)]
    add_variants(pool2)
    assert augment_pool(pool2) == []                     # idempotent: variants are never re-augmented
    assert {p["latex"] for p in pool2} == {"x=15", "x=16", "x=50", "x=60"}
    assert all(p["level"] == "clean" for p in pool2)     # main thread: levels filled


def test_no_number_no_variants():
    assert augment_pool([cand("x=2y+150", ["x equals 2 y plus 150"]), cand("x_{17}=1", ["x sub 17 equals 1"])]) == []


def test_end_to_end_pool_for_fifteen_contains_fifty(monkeypatch):
    """Whisper and the LLM agree on 'fifteen': the pool now also holds 50, as an ordinary candidate."""
    from src import rerank
    monkeypatch.setattr(rerank, "to_latex_timed", lambda t, ctx=None, model=None: (["x=15"], 0.1))
    monkeypatch.setattr(rerank, "to_spoken_timed", lambda l, model=None: (["x equals 15", "x equals fifteen"], 0.1))
    nbest = [("x equals fifteen", -0.1), ("X equals fifteen.", -0.4)]
    pool, _ = build_pool(nbest, workers=1)
    assert config.AUGMENT_CONFUSABLE
    assert [p["latex"] for p in pool] == ["x=15", "x=50"]
    assert pool[1]["augmented"] and pool[1]["spoken"] == ["x equals 50", "x equals fifty"]
    assert pool[1]["cand_rank"] == pool[0]["cand_rank"] + 1
    off, _ = build_pool(nbest, workers=1, augment=False)
    assert [p["latex"] for p in off] == ["x=15"]


def test_score_pool_only_missing(monkeypatch):
    from src import rerank
    calls = []
    monkeypatch.setattr(rerank, "whisper_scores", lambda a, t: calls.append(list(t)) or [-1.0] * len(t))
    monkeypatch.setattr(rerank, "phoneme_scores", lambda a, t: [-2.0] * len(t))
    pool = [cand("x=15", ["x equals fifteen"], whisper=[-0.5], phoneme=[-0.7])]
    add_variants(pool)
    rerank.score_pool(None, pool, only_missing=True)
    assert calls == [["x equals 50", "x equals fifty"]]
    assert pool[0]["whisper"] == [-0.5] and pool[1]["phoneme"] == [-2.0, -2.0]
