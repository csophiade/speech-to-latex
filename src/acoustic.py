"""Acoustic evidence: how well does a candidate text match the audio?

whisper_scores: teacher-forced mean token log-prob under Whisper.
phoneme_scores: -CTC loss per phoneme under wav2vec2-espeak (text phonemized by espeak-ng).
Both encode the audio once and score many texts in one batch.
"""

from functools import lru_cache

import numpy as np
import torch
import torch.nn.functional as F
from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor

from src import config
from src.asr import _load as _load_whisper
from src.candidates import to_spoken


@lru_cache(maxsize=1)
def _load_phoneme():
    processor = Wav2Vec2Processor.from_pretrained(config.PHONEME_MODEL)
    model = Wav2Vec2ForCTC.from_pretrained(config.PHONEME_MODEL).to(config.DEVICE).eval()
    return processor, model


@torch.inference_mode()
def whisper_scores(audio, texts):
    """Mean log-prob per token (text tokens + <|endoftext|>) of each text, teacher-forced."""
    processor, model, _, prompt = _load_whisper()
    tok = processor.tokenizer
    feats = processor(audio, sampling_rate=config.SAMPLE_RATE, return_tensors="pt").input_features
    enc = model.get_encoder()(feats.to(config.DEVICE, config.DTYPE)).last_hidden_state

    seqs = [prompt + tok.encode(" " + t.strip(), add_special_tokens=False) + [tok.eos_token_id]
            for t in texts]
    width = max(map(len, seqs))
    ids = torch.full((len(seqs), width), tok.eos_token_id, dtype=torch.long)
    mask = torch.zeros((len(seqs), width), dtype=torch.bool)
    for i, s in enumerate(seqs):
        ids[i, :len(s)] = torch.tensor(s)
        mask[i, len(prompt):len(s)] = True     # score text tokens and the final EOT
    ids, mask = ids.to(config.DEVICE), mask.to(config.DEVICE)

    logits = model(encoder_outputs=(enc.expand(len(seqs), -1, -1),),
                   decoder_input_ids=ids[:, :-1]).logits.float()
    logp = logits.log_softmax(-1).gather(-1, ids[:, 1:, None]).squeeze(-1)
    m = mask[:, 1:]
    return ((logp * m).sum(1) / m.sum(1)).tolist()


@torch.inference_mode()
def phoneme_scores(audio, texts):
    """-CTC loss / number of phonemes for each text (higher = better match)."""
    processor, model = _load_phoneme()
    inp = processor(audio, sampling_rate=config.SAMPLE_RATE, return_tensors="pt").input_values
    log_probs = model(inp.to(config.DEVICE)).logits.float().log_softmax(-1)   # (1, T, V)
    T = log_probs.shape[1]

    targets = [processor.tokenizer(t).input_ids or [processor.tokenizer.pad_token_id] for t in texts]
    lens = torch.tensor([len(t) for t in targets])
    flat = torch.tensor([i for t in targets for i in t])
    loss = F.ctc_loss(log_probs.transpose(0, 1).expand(-1, len(texts), -1), flat, torch.full((len(texts),), T),
                      lens, blank=model.config.pad_token_id, reduction="none", zero_infinity=False)
    return (-loss.cpu() / lens).tolist()


def _z(x):
    x = np.asarray(x, dtype=np.float64)
    finite = np.isfinite(x)
    if finite.sum() < 2:
        return np.where(finite, 0.0, -10.0)
    sd = x[finite].std()
    return np.where(finite, (x - x[finite].mean()) / (sd if sd > 1e-9 else 1.0), -10.0)


def mix_scores(whisper, phoneme, lengths, beta=config.BETA):
    """Per-candidate acoustic score from raw per-form scores.

    whisper/phoneme: flat raw scores for every spoken form in the utterance's pool;
    lengths: number of forms per candidate. Each score type is z-normalised across the
    pool, mixed as beta*whisper + (1-beta)*phoneme, and each candidate keeps its best form
    (-inf if it has none).
    """
    mixed = beta * _z(whisper) + (1 - beta) * _z(phoneme)
    out, i = [], 0
    for n in lengths:
        out.append(float(mixed[i:i + n].max()) if n else float("-inf"))
        i += n
    return out


def acoustic_from_spoken(audio, spoken_forms, beta=config.BETA):
    """spoken_forms: one list of verbalisations per candidate. Returns one score per candidate."""
    flat = [s for forms in spoken_forms for s in forms]
    if not flat:
        return [float("-inf")] * len(spoken_forms)
    return mix_scores(whisper_scores(audio, flat), phoneme_scores(audio, flat),
                      [len(f) for f in spoken_forms], beta)


def acoustic(audio, candidates, beta=config.BETA):
    """Acoustic score for each LaTeX candidate (max over its LLM verbalisations)."""
    return acoustic_from_spoken(audio, [to_spoken(c) for c in candidates], beta)


def margins(scores):
    """Δ(c) = a(ĉ) − a(c), where ĉ is the best-sounding candidate."""
    best = max(scores)
    return [best - s for s in scores]
