"""Speech2Latex (marsianin500/Speech2Latex): English equations test set + 500-clip subset.

The s2l_equations folder is one 66 GB save_to_disk dump with no split column, so
load_dataset(..., data_dir="s2l_equations") only ever yields a single "train" split.
We stream it to print the splits (no download), then pull the real held-out test
split from data/equations_test-*.parquet (1.3 GB, 2,880 clips) and keep English only.

Outputs (data/speech2latex/):
  en_test/      all 2,745 English test clips (audiofolder)
  en_test_500/  random 500-clip subset of en_test for the main runs
  en_dev/       en_test clips whose equation is NOT in en_test_500 (leak-free tuning set)
"""

import os
import random
import shutil
import sys
import tempfile

import pyarrow.parquet as pq
from datasets import load_dataset
from huggingface_hub import hf_hub_download

from common import DATA_DIR, SEED, dir_size_mb, hf_token, write_audiofolder

REPO = "marsianin500/Speech2Latex"
TEST_FILE = "data/equations_test-00000-of-00001.parquet"
OUT = DATA_DIR / "speech2latex"
N_MAIN = 500
KEEP_COLS = ["sentence_id", "spk", "is_tts", "sentence", "sentence_normalized",
             "pronunciation", "whisper_text", "count_formula_symbols",
             "count_formula_symbols_strates", "language"]


def print_s2l_equations_splits():
    ds = load_dataset(REPO, data_dir="s2l_equations", streaming=True, token=hf_token())
    print("s2l_equations splits (streamed, not downloaded):")
    print(ds)
    print("-> only a single 'train' split; using data/equations_test for the test set.\n")


def main():
    print_s2l_equations_splits()

    with tempfile.TemporaryDirectory(dir=DATA_DIR) as tmp:
        path = hf_hub_download(REPO, TEST_FILE, repo_type="dataset",
                               local_dir=tmp, token=hf_token())
        table = pq.read_table(path, columns=KEEP_COLS + ["audio_path"])
        rows = [r for r in table.to_pylist() if r["language"] == "eng"]

    print(f"equations_test: {table.num_rows} clips, English: {len(rows)}")
    rows.sort(key=lambda r: (r["sentence_id"], r["spk"]))

    def name_fn(i, row, audio):
        return f"{row['sentence_id']}_{row['spk']}.wav"

    if OUT.exists():
        shutil.rmtree(OUT)

    main_idx = set(random.Random(SEED).sample(range(len(rows)), N_MAIN))
    main_ids = {rows[i]["sentence_id"] for i in main_idx}
    splits = {
        "en_test": rows,
        "en_test_500": [r for i, r in enumerate(rows) if i in main_idx],
        "en_dev": [r for r in rows if r["sentence_id"] not in main_ids],
    }
    for split, split_rows in splits.items():
        out = write_audiofolder([dict(r) for r in split_rows], OUT / split, "audio_path", name_fn)
        n_tts = sum(r["is_tts"] for r in split_rows)
        print(f"{split:12s} {len(split_rows):5d} clips "
              f"({len(split_rows) - n_tts} human / {n_tts} TTS, "
              f"{len({r['sentence_id'] for r in split_rows})} equations) "
              f"{dir_size_mb(out):7.1f} MB -> {out}")


if __name__ == "__main__":
    main()
    # datasets 3.6 streaming + pyarrow 26 aborts during interpreter teardown
    # ("PyGILState_Release ... finalizing"); all files are written by now, so skip it.
    sys.stdout.flush()
    os._exit(0)
