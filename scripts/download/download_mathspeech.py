"""MathSpeech (AAAI2025/MathSpeech): 1,101 real lecture clips with LaTeX labels.

The hub calls its only split "train", but it is an evaluation benchmark; we save
it as test.

Output: data/mathspeech/test/ (audiofolder)
"""

import shutil

from datasets import Audio, load_dataset

from common import DATA_DIR, dir_size_mb, hf_token, write_audiofolder

OUT = DATA_DIR / "mathspeech" / "test"


def main():
    ds = load_dataset("AAAI2025/MathSpeech", split="train", token=hf_token())
    ds = ds.cast_column("audio", Audio(decode=False))
    print(ds)

    def name_fn(i, row, audio):
        ext = (audio.get("path") or ".wav").rsplit(".", 1)[-1]
        return f"{i:04d}.{ext}"

    if OUT.exists():
        shutil.rmtree(OUT)
    rows = [{"id": i, **r} for i, r in enumerate(ds)]
    write_audiofolder(rows, OUT, "audio", name_fn)
    print(f"mathspeech test: {len(rows)} clips, {dir_size_mb(OUT):.1f} MB -> {OUT}")


if __name__ == "__main__":
    main()
