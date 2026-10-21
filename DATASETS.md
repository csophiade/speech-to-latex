# Datasets

All data lives in `data/` (gitignored). To recreate it from scratch:

```bash
scripts/download/download_all.sh        # ~2.2 GB on disk, a few minutes
# or one at a time:
.venv/bin/python scripts/download/download_speech2latex.py
.venv/bin/python scripts/download/download_mathspeech.py
.venv/bin/python scripts/download/download_processbench.py
.venv/bin/python scripts/download/download_mathbridge.py
```

Every script is deterministic (seed 42) and safe to re-run; it overwrites its own output folder.

## At a glance

| Dataset | Folder | Type | Size kept | Role |
|---|---|---|---|---|
| Speech2Latex | `data/speech2latex/en_test_500/` | audio → LaTeX | 500 clips | **Main test set** (headline numbers) |
| Speech2Latex | `data/speech2latex/en_dev/` | audio → LaTeX | 1,086 clips | **Validation**: tune prompts, beams, thresholds |
| Speech2Latex | `data/speech2latex/en_test/` | audio → LaTeX | 2,745 clips | Full English test (final or extended runs) |
| MathSpeech | `data/mathspeech/test/` | audio → LaTeX | 1,101 clips | **Test only**: real lecture audio, out-of-domain check |
| MathBridge | `data/mathbridge/train.jsonl` | spoken text → LaTeX | 48k (val/test unused) | **Few-shot pool** for LLM prompts only |
| ProcessBench | `data/processbench/*.jsonl` | text solutions | 3,400 | **Test only**: finding the first wrong step in a student's solution |

**Rule of thumb:** this project fine-tunes nothing. Draw few-shot prompt examples from MathBridge `train`; tune (β, λ, prompts, normalisation) on `en_dev` only; report on `en_test_500` and MathSpeech. Never tune on, or take prompt examples from, the test sets. Each data folder has a `README.md` stating its role.

## Speech2Latex (`marsianin500/Speech2Latex`)

People and TTS voices reading equations from research papers aloud, labelled with the LaTeX.

- **What we kept:** only the English part of the official `equations_test` split (2,745 clips covering 615 distinct equations).
- **What we left out (intentionally — nothing is fine-tuned, so no training audio is needed):**
  - the 66 GB `s2l_equations` folder, which is one dump with only a `train` split and no test split; the script streams it just to print that
  - the ~60 GB training audio
  - the sentence splits
  - the Russian clips
- **The same equation is read several times** (about 2 TTS voices plus 2–3 humans each). So:
  - `en_test_500` is a random 500 clips from `en_test` (270 human, 230 TTS; 362 equations).
  - `en_dev` contains every `en_test` clip whose equation is *not* in `en_test_500`. You can tune on it without leaking test answers.
  - `en_test` is a superset of both. Don't use it alongside them.

Columns in `metadata.jsonl`:

| Column | Meaning |
|---|---|
| `file_name` | path to the audio file, relative to the split folder |
| `sentence` | reference LaTeX (tokens separated by spaces) |
| `sentence_normalized` | LaTeX with whitespace removed; compare against this |
| `pronunciation` | what the speaker was asked to say |
| `whisper_text` | the dataset authors' Whisper transcript (a baseline) |
| `is_tts` | 1 = synthetic voice, 0 = human. Report the two separately; human speech matters most for accessibility |
| `spk` | speaker ID |
| `count_formula_symbols_strates` | difficulty bucket by formula length (`3-10` … `50+`) |
| `duration_s`, `sample_rate` | clip length and native sample rate |

## MathSpeech (`AAAI2025/MathSpeech`)

1,101 short clips cut from real YouTube math lectures, each with a human transcription and LaTeX. On the Hub its only split is called `train`, but it is a benchmark, so it's saved as `test/`. Use it to check that results hold on natural classroom speech, not only read-aloud speech.

Columns: `transcription` (spoken words), `LaTeX` (reference, wrapped in `$…$`), `Source` (lecture URL), `id`. The audio is MP3.

## MathBridge (`Kyudan/MathBridge`)

Text only: pairs of spoken-English descriptions and LaTeX from arXiv papers, about 23M rows in total. It was **streamed**, never downloaded: the first rows went through a 20k-row shuffle buffer, empty and duplicate pairs were dropped, and 50,000 were kept and split 48,000 / 1,000 / 1,000.

Columns: `spoken_English`, `equation` (LaTeX wrapped in `$…$`), `context_before`, `context_after` (surrounding sentence text, useful as extra prompt context).

Role: only `train.jsonl` is used, as a pool of few-shot (spoken → LaTeX) examples for the LLM prompts. Nothing is trained on it; `val.jsonl`/`test.jsonl` are unused.

To get more or different rows, change `N_ROWS` or the seed in the script.

## ProcessBench (`Qwen/ProcessBench`)

3,400 step-by-step solutions to math problems, written by language models and labelled by humans. It supports the teaching side of the project: given a student's spoken solution, find where it first goes wrong.

| File | Rows | Difficulty |
|---|---|---|
| `gsm8k.jsonl` | 400 | grade school |
| `math.jsonl` | 1,000 | high school competition |
| `olympiadbench.jsonl` | 1,000 | olympiad |
| `omnimath.jsonl` | 1,000 | olympiad |

Columns: `problem`, `steps` (list of strings), `label` (index of the first wrong step, or `-1` if every step is correct), `final_answer_correct`, `generator`, `id`. The standard metric is the F1 score of accuracy on wrong solutions and accuracy on correct ones; see github.com/QwenLM/ProcessBench.

## Loading

```python
from datasets import load_dataset, Audio

# Audio sets: the split name is always "train" for audiofolder, whatever the folder holds
test = load_dataset("audiofolder", data_dir="data/speech2latex/en_test_500", split="train")
test = test.cast_column("audio", Audio(sampling_rate=16_000))   # Whisper expects 16 kHz
ex = test[0]; ex["audio"]["array"], ex["sentence_normalized"]

human_only = test.filter(lambda r: r["is_tts"] == 0)

lectures = load_dataset("audiofolder", data_dir="data/mathspeech/test", split="train")

# Text sets
mb = load_dataset("json", data_files={s: f"data/mathbridge/{s}.jsonl" for s in ["train", "val", "test"]})
pb = load_dataset("json", data_files={s: f"data/processbench/{s}.jsonl"
                                      for s in ["gsm8k", "math", "olympiadbench", "omnimath"]})
```

You can also skip `datasets` and read `metadata.jsonl` yourself, then load `<folder>/<file_name>` with `librosa.load(path, sr=16000)` as in `scripts/transcribe.py`.

## Things to keep in mind

- **Mixed sample rates.** Speech2Latex has 24 kHz and 48 kHz files; MathSpeech is 48 kHz. Always resample to 16 kHz for Whisper.
- **LaTeX formatting differs** between datasets: spaced tokens, `$…$` wrappers, `\ldots` vs `\dots`. Normalize before computing exact match or CER. Strip `$` and whitespace at minimum; for a fairer score, compare with SymPy (`parse_latex`).
- **Train/test overlap is small but not zero.** 17 of 615 Speech2Latex test equations and 3 of 1,101 MathSpeech clips have LaTeX identical (ignoring whitespace) to something in MathBridge `train`. Also, 44 of MathBridge's 1,000 test equations appear in its own train set, because short formulas like `$x$` recur with different spoken text. Filter these out if you need a strictly clean split.
- **Spoken-MQA** (github.com/amao0o0/Spoken-MQA) is optional and was not downloaded.
- The two streaming scripts end with `os._exit(0)`, because `datasets` 3.6 with `pyarrow` 26 crashes during Python shutdown after streaming. The data is already written by that point.
