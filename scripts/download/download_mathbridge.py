"""MathBridge (Kyudan/MathBridge): spoken-English -> LaTeX text pairs, ~23M rows.

Never downloaded in full: we stream the first rows through a shuffle buffer and
keep 50,000 usable pairs, split 48k train / 1k val / 1k test.

Output: data/mathbridge/{train,val,test}.jsonl
"""

import os
import random
import sys

from datasets import load_dataset

from common import DATA_DIR, SEED, hf_token, write_jsonl

OUT = DATA_DIR / "mathbridge"
N_ROWS = 50_000
N_VAL = N_TEST = 1_000


def main():
    stream = load_dataset("Kyudan/MathBridge", split="train", streaming=True, token=hf_token())
    stream = stream.shuffle(seed=SEED, buffer_size=20_000)

    rows, seen = [], set()
    for r in stream:
        eq, spoken = (r["equation"] or "").strip(), (r["spoken_English"] or "").strip()
        if not eq or not spoken or (eq, spoken) in seen:
            continue
        seen.add((eq, spoken))
        rows.append({"spoken_English": spoken, "equation": eq,
                     "context_before": r["context_before"], "context_after": r["context_after"]})
        if len(rows) == N_ROWS:
            break

    random.Random(SEED).shuffle(rows)
    splits = {"val": rows[:N_VAL], "test": rows[N_VAL:N_VAL + N_TEST],
              "train": rows[N_VAL + N_TEST:]}
    for split, split_rows in splits.items():
        path = write_jsonl(split_rows, OUT / f"{split}.jsonl")
        print(f"{split:5s} {len(split_rows):6d} rows -> {path}")


if __name__ == "__main__":
    main()
    # datasets 3.6 streaming + pyarrow 26 aborts during interpreter teardown
    # ("PyGILState_Release ... finalizing"); all files are written by now, so skip it.
    sys.stdout.flush()
    os._exit(0)
