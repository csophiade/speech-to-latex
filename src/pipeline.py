"""Live per-line pipeline for the demo (Phase 5): the shipped derivation-mode system, models loaded once.

Per line, exactly the logic of scripts/run_derivations.py (stage 1 + run_derivation):
Whisper 5-best -> build_pool (gpt-4.1-mini, context variables of the problem line, confusable-number
augmentation) -> score_pool -> frozen System D (results/tuned.json) -> checker of every candidate
against the previous accepted output -> decide() with class-conditional thresholds: candidates that
are a single confusable-number swap of ĉ (src.decide.tag_homophone) use delta_hom, all others
delta (prereg_homophone.md, runtime class rule).

The CPU half (graded parse levels, checker, decide) uses SIGALRM timeouts that only work on a main
thread. Gradio calls handlers from worker threads, so with cpu_worker=True that half runs in a
one-process pool (spawned; each task runs on the child's main thread). Called from the main
thread, it runs inline.

Beyond the decision rule (display only, never changes status or output): for a math_error whose
pool holds a VALID homophone-class reading outside delta_hom, `suggestion` names it, so the
red flag can say "you said fifteen; fifty would make this step valid - did you mean fifty?".
"""

import re
import sys
import threading
import time
from concurrent.futures import ProcessPoolExecutor
from multiprocessing import get_context

from src import config
from src.decide import FLAGS, is_valid

sys.path.insert(0, str(config.ROOT))   # scripts/ as a namespace package (run_derivations helpers)

DELTA_POOLED = 0.5518526991984581    # pooled calibration, n = 327, k = 16
DELTA_HOM = 0.5395088972175274       # homophone-class calibration, n = 200, k = 10 (pre-registered)
N_SHOWN = 5


# ---------- CPU half (picklable; runs inline or in the worker process) ----------

def decide_line(pool, prev_out, line_index, delta, delta_hom):
    """Levels, checker statuses, deltas, homophone tags and the decision for one line's pool.
    Returns (decision, cands ĉ-first, seconds per part: levels / checker / classes / decide)."""
    from scripts.run_derivations import frozen_d, line_candidates
    from src.decide import classes, decide, tag_homophone
    from src.rerank import fill_levels

    t0 = time.perf_counter()
    fill_levels(pool)
    t1 = time.perf_counter()
    cands, check_s = line_candidates(pool, prev_out, line_index, frozen_d())
    t2 = time.perf_counter()
    parts = {"levels_s": t1 - t0, "checker_s": check_s, "classes_s": 0.0, "decide_s": 0.0}
    if not cands:
        return None, [], parts
    if line_index > 0 and prev_out is None:
        for c in cands:
            c.update(status="unchecked", explanation="no earlier line to compare with")
    for c, k in zip(cands, classes([c["latex"] for c in cands])):
        c["cls"] = k
    cands = tag_homophone(cands)
    t3 = time.perf_counter()
    d = decide(cands, delta, line_index, delta_hom=delta_hom)
    parts.update(classes_s=t3 - t2, decide_s=time.perf_counter() - t3)
    return d, cands, parts


def _warm():
    """One tiny line through decide_line in the worker (imports and first-call costs off the live path)."""
    pool = [{"latex": x, "level": None, "src_rank": 0, "src_logprob": -0.1, "cand_rank": j, "spoken": [x],
             "whisper": [0.0], "phoneme": [-float(j)]} for j, x in enumerate(["x = 15", "x = 50"])]
    return decide_line(pool, "x + 1 = 16", 1, 0.5, 0.5)[0]["status"]


def typed_decide(latex, prev_out, line_index):
    """Decision for a TYPED line (no ASR, so no alternatives): the checker against prev_out, then
    decide() on the single candidate. Only verified / math_error / unchecked are possible (given on
    line 0); a typed line is never corrected or asked back. Main-thread only (checker SIGALRM)."""
    from src.checker import check_step
    from src.decide import decide
    if line_index == 0:
        c = {"latex": latex, "delta": 0.0, "status": "given", "explanation": None}
    elif prev_out is None:
        c = {"latex": latex, "delta": 0.0, "status": "unchecked", "explanation": "no earlier line to compare with"}
    else:
        r = check_step(prev_out, latex)
        c = {"latex": latex, "delta": 0.0, "status": r["status"], "explanation": r["explanation"]}
    return decide([c], 0.0, line_index)


def readable(latex):
    """True if the checker can read latex as a line (main-thread only)."""
    from src.checker import read_line
    try:
        return read_line(latex) is not None
    except Exception:
        return False


# ---------- spoken explanations ----------

def swap_words(hat_latex, c_latex):
    """(said, meant) number words if c is one confusable swap of ĉ, else None."""
    from src.augment import latex_swaps
    from src.eval import normalise
    from src.spoken_norm import number_words
    target = normalise(c_latex)
    for new, v, partner, _ in latex_swaps(hat_latex):
        if normalise(new) == target:
            return number_words(v), number_words(partner)
    return None


def suggestion(cands):
    """Smallest-delta VALID homophone-class candidate (any delta), or None."""
    hom = [c for c in cands[1:] if c.get("hom_class") and is_valid(c["status"])]
    return min(hom, key=lambda c: c["delta"]) if hom else None


def speakable(text):
    """Checker wording made speakable: symbols as words; solution sets with radicals summarised."""
    if "sqrt" in text or len(text) > 90:
        return "this step changes the solutions, so it does not follow from the line before"
    text = re.sub(r"(?<![\w)])-(?=\s*[\dx])", "minus ", text)
    for sym, word in ((" = ", " equals "), (" < ", " is less than "), (" > ", " is greater than "),
                      (" - ", " minus "), (" + ", " plus "), ("/", " over ")):
        text = text.replace(sym, word)
    return text


def spoken_explanation(res):
    """One or two sentences for TTS (plain words; no LaTeX), or None for unflagged lines."""
    status, sug = res["status"], res.get("suggestion")
    if status == "math_error":
        if sug and sug.get("words"):   # the question alone: short enough to land right after the clip
            said, meant = sug["words"]
            return f"You said {said}; {meant} would make this step valid. Did you mean {meant}?"
        text = speakable(res.get("explanation") or "this step does not follow from the previous line").rstrip(".")
        return text[:1].upper() + text[1:] + "."
    if status == "ask_back":
        ws = [swap_words(res["heard"], o) for o in res["options"][:2]]
        if all(ws):
            (s1, m1), (s2, m2) = ws
            return ("Two sound-alike readings make this step valid. "
                    f"Did you mean {m1} instead of {s1}, or {m2} instead of {s2}?")
        return "Two readings of this line both make the step valid. Which did you mean?"
    if status == "corrected":
        w = swap_words(res["heard"], res["output"])
        return f"I heard {w[0]} and wrote {w[1]}, which makes the step valid." if w else None
    return None


# ---------- pipeline ----------

class LinePipeline:
    """process_line(audio, sr, history) -> result dict; models are loaded on construction.

    history: earlier accepted lines, dicts with "output" (LaTeX), line 0 first. The problem line's
    variables are the context variables for lines >= 1; the checker compares with history[-1].
    asr / build_pool / score_pool: injectable for offline tests (defaults: the shipped modules).
    """

    def __init__(self, delta=DELTA_POOLED, delta_hom=DELTA_HOM, llm=config.LLM_MODEL,
                 asr=None, build_pool=None, score_pool=None, cpu_worker=False, warm=True):
        self.delta, self.delta_hom, self.llm = delta, delta_hom, llm
        if asr is None:
            from src.asr import transcribe_nbest as asr
        if build_pool is None or score_pool is None:
            from src import rerank
            build_pool = build_pool or rerank.build_pool
            score_pool = score_pool or rerank.score_pool
        self.asr, self.build_pool, self.score_pool = asr, build_pool, score_pool
        self._vars = {}
        self._lock = threading.Lock()   # one line at a time on the GPU
        self._worker = ProcessPoolExecutor(1, mp_context=get_context("spawn")) if cpu_worker else None
        if self._worker:
            self._worker.submit(_warm).result()
        if warm:
            self.warm_up()

    def warm_up(self):
        """Load Whisper and the phoneme model (untimed)."""
        import numpy as np
        zero = np.zeros(config.SAMPLE_RATE, dtype=np.float32)
        self.asr(zero)
        self.score_pool(zero, [{"spoken": ["warm up"], "latex": "x", "level": "clean"}])

    def close(self):
        if self._worker:
            self._worker.shutdown(cancel_futures=True)

    def context_vars(self, history):
        """Variables of the problem line (history[0]), cached per problem."""
        if not history:
            return None
        line0 = history[0]["output"]
        if line0 not in self._vars:
            from scripts.run_derivations import problem_vars
            self._vars[line0] = problem_vars(line0)
        return self._vars[line0]

    def _decide(self, pool, prev_out, line_index):
        args = (pool, prev_out, line_index, self.delta, self.delta_hom)
        if self._worker is None and threading.current_thread() is not threading.main_thread():
            raise RuntimeError("off the main thread: construct LinePipeline(cpu_worker=True)")
        return self._worker.submit(decide_line, *args).result() if self._worker else decide_line(*args)

    def process_line(self, audio, sr, history):
        """Run one spoken line against the accepted history; see module docstring."""
        from src.audio import to_16k_mono
        t0 = time.perf_counter()
        audio, _ = to_16k_mono(audio, sr)
        if audio.size < config.SAMPLE_RATE // 4:
            raise ValueError("the recording is shorter than a quarter of a second")
        line_index = len(history)
        prev_out = history[-1]["output"] if history else None
        with self._lock:
            t1 = time.perf_counter()
            nbest = self.asr(audio)
            t2 = time.perf_counter()
            ctx = self.context_vars(history) if line_index > 0 else None
            pool, timing = self.build_pool(nbest, self.llm, 8, ctx)
            t3 = time.perf_counter()
            gpu = self.score_pool(audio, pool)
        t4 = time.perf_counter()
        d, cands, cpu = self._decide(pool, prev_out, line_index)
        t5 = time.perf_counter()
        parts = {"audio_s": t1 - t0, "asr_s": t2 - t1, "llm_s": t3 - t2,
                 "whisper_score_s": gpu["whisper_s"], "phoneme_s": gpu["phoneme_s"], **cpu,
                 "ipc_s": max(0.0, (t5 - t4) - sum(cpu.values()))}
        return self._result(d, cands, nbest, line_index, t5 - t0, parts, timing)

    def _cpu(self, fn, *args):
        """Run a picklable CPU function where SIGALRM works (the worker, or inline on the main thread)."""
        if self._worker is None and threading.current_thread() is not threading.main_thread():
            raise RuntimeError("off the main thread: construct LinePipeline(cpu_worker=True)")
        return self._worker.submit(fn, *args).result() if self._worker else fn(*args)

    def readable(self, latex):
        """True if the checker can read a typed line (e.g. the problem statement)."""
        return self._cpu(readable, latex.strip())

    def typed_line(self, latex, history):
        """A typed LaTeX line checked against history[-1] (checker + decide, no ASR); line 0 is given."""
        latex = latex.strip()
        if not history:
            return self.given_line(latex)
        d = self._cpu(typed_decide, latex, history[-1]["output"], len(history))
        res = dict(d, latency_s=0.0, latency_parts={}, line_index=len(history), transcript=None,
                   typed=True, suggestion=None, candidates=[], flagged=d["status"] in FLAGS)
        res["spoken_explanation"] = spoken_explanation(res)
        return res

    def given_line(self, latex):
        """A typed problem statement (line 0)."""
        return {"status": "given", "output": latex.strip(), "heard": latex.strip(), "options": [],
                "explanation": None, "note": None, "suggestion": None, "latency_s": 0.0,
                "latency_parts": {}, "candidates": [], "transcript": None, "line_index": 0,
                "typed": True, "spoken_explanation": None}

    def _result(self, d, cands, nbest, line_index, total_s, parts, timing):
        if d is None:
            d = {"status": "unchecked", "output": "", "heard": "", "options": [],
                 "explanation": "nothing was recognised", "note": None}
        res = dict(d, latency_s=total_s, latency_parts=parts, llm_timing=timing, line_index=line_index,
                   transcript=nbest[0][0] if nbest else "", typed=False, suggestion=None,
                   candidates=[{k: c.get(k) for k in ("latex", "delta", "status", "hom_class", "augmented")}
                               for c in sorted(cands, key=lambda c: c["delta"])[:N_SHOWN]])
        sug = suggestion(cands) if d["status"] == "math_error" else None
        if sug:
            res["suggestion"] = {"latex": sug["latex"], "delta": sug["delta"],
                                 "words": swap_words(cands[0]["latex"], sug["latex"])}
        res["flagged"] = d["status"] in FLAGS
        res["spoken_explanation"] = spoken_explanation(res)
        return res
