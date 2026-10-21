"""LLM stage: transcript -> LaTeX candidates, with a disk cache keyed by prompt hash."""

import hashlib
import json
import random
import re
import time
from functools import lru_cache

from dotenv import dotenv_values
from openai import OpenAI

from src import config
from src.eval import normalise

SYSTEM = (
    "You convert spoken mathematics into LaTeX. The input is a speech-recognition transcript "
    "and may contain recognition errors (homophones, wrong numbers, dropped words); use "
    "mathematical context to recover what the speaker most likely said.\n"
    "Return ONLY a JSON object {\"candidates\": [...]} with up to {m} LaTeX strings, most likely "
    "first. Alternatives must be genuinely different readings (another word the recogniser may have "
    "misheard, another grouping), not the same expression re-spaced; give fewer if there is no "
    "real ambiguity. Each is a bare LaTeX expression: no $ delimiters, no prose, no "
    "\\text{{}} unless the speaker clearly said words that are not maths."
)
# Few-shot pool filter: only standard commands, so the examples don't teach paper-specific macros
STANDARD_CMDS = set("""
frac sqrt sum prod int lim infty partial nabla cdot times div pm mp leq geq neq le ge ne approx
equiv sim propto in notin subset subseteq cup cap emptyset forall exists to rightarrow mapsto
log ln exp sin cos tan sec csc cot arcsin arccos arctan sinh cosh tanh max min sup inf det
alpha beta gamma delta epsilon varepsilon zeta eta theta vartheta iota kappa lambda mu nu xi pi
rho sigma tau upsilon phi varphi chi psi omega Gamma Delta Theta Lambda Xi Pi Sigma Phi Psi Omega
mathbb mathcal hat bar tilde vec dot ddot prime ldots cdots dots left right langle rangle mid
""".split())

RETRY = "That was not valid. Reply with ONLY the JSON object {\"candidates\": [\"...\", ...]}."


@lru_cache(maxsize=1)
def _client():
    return OpenAI(api_key=dotenv_values(config.ROOT / ".env")["OPENAI_API_KEY"], max_retries=8)


@lru_cache(maxsize=1)
def fewshot_pairs(n=config.N_FEWSHOT, seed=config.SEED):
    """n short (spoken, LaTeX) pairs from the MathBridge train pool."""
    pool = []
    with open(config.MATHBRIDGE_POOL) as f:
        for line in f:
            r = json.loads(line)
            eq = r["equation"].strip().strip("$").strip()
            cmds = set(re.findall(r"\\([A-Za-z]+)", eq))
            if (8 <= len(eq) <= 60 and 4 <= len(r["spoken_English"].split()) <= 25
                    and cmds <= STANDARD_CMDS):
                pool.append((r["spoken_English"], eq))
    return tuple(random.Random(seed).sample(pool, n))


def llm_call(messages, model=config.LLM_MODEL, json_mode=True):
    """Cached chat completion. Returns (text, seconds_of_original_call)."""
    payload = {"model": model, "messages": messages, "temperature": 0, "json": json_mode}
    key = hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()
    path = config.LLM_CACHE_DIR / key[:2] / f"{key}.json"
    if path.exists():
        hit = json.loads(path.read_text())
        return hit["text"], hit["seconds"]
    start = time.perf_counter()
    resp = _client().chat.completions.create(
        model=model, messages=messages, temperature=0,
        **({"response_format": {"type": "json_object"}} if json_mode else {}))
    seconds = time.perf_counter() - start
    text = resp.choices[0].message.content or ""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"payload": payload, "text": text, "seconds": seconds}, ensure_ascii=False))
    return text, seconds


def _parse_candidates(text, m):
    try:
        obj = json.loads(text)
    except json.JSONDecodeError:
        return None
    cands = obj.get("candidates") if isinstance(obj, dict) else obj
    if not isinstance(cands, list) or not all(isinstance(c, str) for c in cands):
        return None
    out, seen = [], set()
    for c in cands:
        c = re.sub(r"^\$+|\$+$", "", c.strip()).strip()
        if c and normalise(c) not in seen:
            seen.add(normalise(c))
            out.append(c)
    return out[:m] or None


def build_messages(transcript, context_vars=None, m=config.M_CANDIDATES):
    examples = "\n".join(f"- {spoken}  ->  {latex}" for spoken, latex in fewshot_pairs())
    system = SYSTEM.replace("{m}", str(m)).replace("{{}}", "{}")
    msgs = [{"role": "system", "content": f"{system}\n\nExamples of spoken maths and its LaTeX:\n{examples}"}]
    user = transcript
    if context_vars:
        user = f"(Variables already in use: {', '.join(sorted(context_vars))})\n{transcript}"
    msgs.append({"role": "user", "content": user})
    return msgs


def to_latex_timed(transcript, context_vars=None, m=config.M_CANDIDATES, model=config.LLM_MODEL):
    """(candidates, seconds). One call; one retry on malformed JSON; [] on second failure."""
    msgs = build_messages(transcript, context_vars, m)
    text, seconds = llm_call(msgs, model)
    cands = _parse_candidates(text, m)
    if cands is None:
        retry = msgs + [{"role": "assistant", "content": text}, {"role": "user", "content": RETRY}]
        text, s2 = llm_call(retry, model)
        seconds += s2
        cands = _parse_candidates(text, m) or []
    return cands, seconds


def to_latex(transcript, context_vars=None, m=config.M_CANDIDATES, model=config.LLM_MODEL):
    return to_latex_timed(transcript, context_vars, m, model)[0]


SPOKEN_SYSTEM = (
    "You say LaTeX maths out loud. Given one LaTeX expression, return ONLY a JSON object "
    "{\"spoken\": [...]} with {k} different natural ways a person would read it aloud in English "
    "(e.g. \"x squared plus one\", \"x to the power of two plus one\"). Plain words; digits are fine. "
    "No LaTeX, no symbols like ^ or \\."
)


def to_spoken_timed(latex, k=3, model=config.LLM_MODEL):
    """(2-3 natural English verbalisations of a LaTeX string, seconds); [] on failure. Cached."""
    examples = "\n".join(f"- {latex_}  ->  {spoken}" for spoken, latex_ in fewshot_pairs())
    msgs = [{"role": "system", "content": SPOKEN_SYSTEM.replace("{k}", str(k))
             + f"\n\nExamples of LaTeX and one way to say it:\n{examples}"},
            {"role": "user", "content": latex}]
    seconds = 0.0
    for attempt in (msgs, msgs + [{"role": "user", "content": RETRY.replace("candidates", "spoken")}]):
        text, s = llm_call(attempt, model)
        seconds += s
        try:
            spoken = json.loads(text).get("spoken")
        except (json.JSONDecodeError, AttributeError):
            continue
        if isinstance(spoken, list):
            out = [x.strip() for x in spoken if isinstance(x, str) and x.strip()]
            if out:
                return list(dict.fromkeys(out))[:k], seconds
    return [], seconds


def to_spoken(latex, k=3, model=config.LLM_MODEL):
    return to_spoken_timed(latex, k, model)[0]
