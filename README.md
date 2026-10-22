# Speech-to-LaTeX

## Credits and licences

Models (used as released, nothing fine-tuned):

- **CosyVoice2-0.5B** (`FunAudioLLM/CosyVoice2-0.5B`), Apache-2.0: synthesises the derivation audio.
- **Whisper large-v3-turbo** (`openai/whisper-large-v3-turbo`), MIT: speech recognition.
- **wav2vec2-lv-60-espeak-cv-ft** (`facebook/wav2vec2-lv-60-espeak-cv-ft`), Apache-2.0: phoneme scores for reranking.

Voices:

- The synthetic voices v1 and v2 are cloned from LibriSpeech speakers 1272 and 1993.
  LibriSpeech ASR corpus, CC BY 4.0: V. Panayotov, G. Chen, D. Povey and S. Khudanpur, "LibriSpeech: an ASR
  corpus based on public domain audio books", ICASSP 2015. It is derived from LibriVox public-domain recordings.
  The reference clips are not redistributed.
- The accented voice v3 is cloned from L2-ARCTIC speaker TXHC (Zhao et al., Interspeech 2018), CC BY-NC 4.0
  (non-commercial). The reference clip is not redistributed.

Datasets:

- **Speech2Latex** (`marsianin500/Speech2Latex`), CC BY 4.0: main audio-to-LaTeX test set.
- **MathSpeech** (`AAAI2025/MathSpeech`): its Hugging Face card states no licence. We use it for
  evaluation only and do not redistribute it.
- **MathBridge** (`Kyudan/MathBridge`), MIT: few-shot prompt examples.
- **ProcessBench** (`Qwen/ProcessBench`), Apache-2.0: side benchmark for the step checker.

Licences were checked against the Hugging Face model and dataset cards on 2026-10-22.
