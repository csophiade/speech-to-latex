"""ProcessBench (Qwen/ProcessBench): 3,400 human-labelled step-by-step solutions (text).

Each row has a problem, a list of solution steps, and `label` = index of the first
wrong step (-1 if the solution is fully correct). Evaluation only.

Output: data/processbench/{gsm8k,math,olympiadbench,omnimath}.jsonl
"""

from datasets import load_dataset

from common import DATA_DIR, hf_token, write_jsonl

OUT = DATA_DIR / "processbench"


def main():
    ds = load_dataset("Qwen/ProcessBench", token=hf_token())
    print(ds)
    for split, d in ds.items():
        path = write_jsonl(d, OUT / f"{split}.jsonl")
        n_correct = sum(r["label"] == -1 for r in d)
        print(f"{split:14s} {len(d):5d} rows ({n_correct} fully correct) -> {path}")


if __name__ == "__main__":
    main()
