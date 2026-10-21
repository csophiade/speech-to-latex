"""Whisper n-best transcription."""

import copy
from functools import lru_cache

import torch
from transformers import GenerationMixin, WhisperForConditionalGeneration, WhisperProcessor

from src import config

_PROMPT = ["<|startoftranscript|>", "<|en|>", "<|transcribe|>", "<|notimestamps|>"]


@lru_cache(maxsize=1)
def _load():
    processor = WhisperProcessor.from_pretrained(config.WHISPER_MODEL)
    model = WhisperForConditionalGeneration.from_pretrained(
        config.WHISPER_MODEL, dtype=config.DTYPE).to(config.DEVICE).eval()
    gen_config = copy.deepcopy(model.generation_config)
    gen_config.forced_decoder_ids = None
    prompt = processor.tokenizer.convert_tokens_to_ids(_PROMPT)
    return processor, model, gen_config, prompt


@torch.inference_mode()
def transcribe_nbest(audio, k=config.K_BEAMS):
    """Return up to k (text, logprob) pairs, best first.

    logprob is the length-normalised beam log-probability (sum of token
    log-probs / length). Calls the generic beam search directly: Whisper's own
    generate() in transformers 4.57 returns the top beam k times.
    """
    processor, model, gen_config, prompt = _load()
    feats = processor(audio, sampling_rate=config.SAMPLE_RATE, return_tensors="pt").input_features
    out = GenerationMixin.generate(
        model, feats.to(config.DEVICE, config.DTYPE), generation_config=gen_config,
        decoder_input_ids=torch.tensor([prompt], device=config.DEVICE),
        num_beams=k, num_return_sequences=k, do_sample=False, max_new_tokens=200,
        return_dict_in_generate=True, output_scores=True)
    texts = processor.batch_decode(out.sequences, skip_special_tokens=True)
    return [(t.strip(), float(s)) for t, s in zip(texts, out.sequences_scores.tolist())]


@lru_cache(maxsize=1)
def _word_pipeline():
    from transformers import pipeline
    return pipeline("automatic-speech-recognition", model=config.WHISPER_MODEL,
                    device=config.DEVICE, dtype=config.DTYPE)


def word_timestamps(audio):
    """[(word, start_s, end_s)] for the greedy transcript (~2.5 s/clip; for the pause feature only)."""
    out = _word_pipeline()({"array": audio, "sampling_rate": config.SAMPLE_RATE},
                           return_timestamps="word",
                           generate_kwargs={"language": "en", "task": "transcribe"})
    return [(c["text"].strip(), *c["timestamp"]) for c in out["chunks"]]
