"""Phase 3.6: ingest the human-recorded derivations.

Usage: .venv/bin/python -W ignore scripts/ingest_recordings.py [--src DIR] [--out DIR] [--check-asr]
Reads <src>/recordings_labels.csv (one row per line: file, deriv_id, split_suggestion, deriv_type,
error_type, error_step, n_lines, line_idx (1-based), latex, spoken_text) and <src>/dNNN.wav;
loads each wav as 16 kHz mono, splits it into lines on silences, and writes
<out>/lines/<file>/line_<k>.wav (k 0-based, as in derivations.json) + <out>/manifest.jsonl.
A file whose segment count differs from its labelled line count is reported (with thresholds that
would give the right count) and skipped, never guessed. --check-asr transcribes every segment with
Whisper and accepts a segmentation only if each line's WER vs spoken_text is <= 0.5, trying other
thresholds when the default fails; files that still fail are reported and skipped.
"""

import argparse
import csv
import json
import shutil
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import soundfile as sf

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.audio import load_audio, split_on_silence  # noqa: E402
from src.config import DATA_DIR, SAMPLE_RATE  # noqa: E402

REC_DIR = DATA_DIR / "recordings"
SPLITS = {"calibration": "calib", "calib": "calib", "test": "test"}
GRID_GAP = (1.2, 1.0, 1.5, 0.8, 1.8, 0.6)   # min_gap_s values tried when the default fails
GRID_THRESH = (0.3, 0.2, 0.4, 0.15, 0.5)    # rel_thresh values
MAX_WER = 0.5   # a segment whose transcript is further from its label than this is misaligned


def read_labels(path):
    """{file: {"split", "deriv_type", "error", "lines": [(latex, spoken)]}} from the labels CSV."""
    by_file = defaultdict(list)
    with open(path, newline="") as f:
        for row in csv.DictReader(f):
            by_file[row["file"]].append(row)
    labels = {}
    for file, rows in by_file.items():
        rows.sort(key=lambda r: int(r["line_idx"]))
        r0 = rows[0]
        if [int(r["line_idx"]) for r in rows] != list(range(1, int(r0["n_lines"]) + 1)):
            raise ValueError(f"{file}: line_idx is not 1..n_lines")
        stem = Path(file).stem
        split = SPLITS.get(r0["split_suggestion"].strip().lower())
        if split is None:   # no suggestion: d001-d020 calib, rest test (data/script.md)
            split = "calib" if int(stem[1:]) <= 20 else "test"
        error = ({"type": r0["error_type"], "line": int(r0["error_step"]) - 1}
                 if r0["error_type"].strip() else None)
        labels[file] = {"split": split, "deriv_type": r0["deriv_type"], "error": error,
                        "lines": [(r["latex"], r["spoken_text"]) for r in rows]}
    return labels


def settings(audio, n, min_gap_s, rel_thresh):
    """[(params, segments)] giving exactly n segments: the requested setting first, then the grid.
    Settings that produce identical segmentations are tried once."""
    out, seen = [], set()
    for g, t in [(min_gap_s, rel_thresh)] + [(g, t) for g in GRID_GAP for t in GRID_THRESH]:
        segs = split_on_silence(audio, min_gap_s=g, rel_thresh=t)
        if len(segs) == n and tuple(segs) not in seen:
            seen.add(tuple(segs))
            out.append(({"min_gap_s": g, "rel_thresh": t}, segs))
    return out


def asr_verifier(max_wer=MAX_WER):
    """verify(clips, spoken) -> (ok, [(asr, wer)]): Whisper best-1 per clip vs the labelled text."""
    import jiwer

    from src.asr import transcribe_nbest
    from src.spoken_norm import normalise_spoken

    def verify(clips, spoken):
        res = []
        for clip, ref in zip(clips, spoken):
            hyp = transcribe_nbest(clip)[0][0]
            res.append((hyp, round(jiwer.wer(normalise_spoken(ref), normalise_spoken(hyp) or "<empty>"), 3)))
        return all(w <= max_wer for _, w in res), res
    return verify


def ingest(src, out, min_gap_s=1.2, rel_thresh=0.3, verify=None):
    """Split every labelled wav in src; returns (manifest rows, problems).

    Without verify, only the requested thresholds are used and a count mismatch is a problem.
    With verify (see asr_verifier), each candidate segmentation with the right count is transcribed
    and the first whose every line matches its label is kept; this catches segmentations whose count
    is right by coincidence (one line split in two, two lines merged)."""
    labels = read_labels(src / "recordings_labels.csv")
    rows, problems = [], []
    for file, lab in sorted(labels.items()):
        stem, n = Path(file).stem, len(lab["lines"])
        shutil.rmtree(out / "lines" / stem, ignore_errors=True)   # no stale lines from an earlier run
        if not (src / file).exists():
            problems.append({"file": file, "problem": "missing wav"})
            continue
        audio, sr = load_audio(src / file)
        found = settings(audio, n, min_gap_s, rel_thresh)
        default = split_on_silence(audio, sr, min_gap_s=min_gap_s, rel_thresh=rel_thresh)
        if verify is None:
            found = found[:1] if len(default) == n else []
        chosen, tried = None, []
        for params, segs in found:
            if verify is None:
                chosen = (params, segs, [(None, None)] * n)
                break
            ok, res = verify([audio[s:e] for s, e in segs], [sp for _, sp in lab["lines"]])
            tried.append({**params, "wer": [w for _, w in res]})
            if ok:
                chosen = (params, segs, res)
                break
        if chosen is None:
            problems.append({"file": file, "problem": f"{len(default)} segments for {n} labelled lines"
                             if len(default) != n else "segments do not match the labelled lines (ASR)",
                             "segments_s": [round((e - s) / sr, 2) for s, e in default],
                             "would_give_count": [p for p, _ in found][:3], "asr_tried": tried[:3]})
            continue
        params, segs, res = chosen
        if params != {"min_gap_s": min_gap_s, "rel_thresh": rel_thresh}:
            problems.append({"file": file, "problem": "fixed with other thresholds (check by ear)", **params})
        for k, ((s, e), (latex, spoken), (hyp, wer)) in enumerate(zip(segs, lab["lines"], res)):
            wav = out / "lines" / stem / f"line_{k}.wav"
            wav.parent.mkdir(parents=True, exist_ok=True)
            sf.write(wav, audio[s:e], SAMPLE_RATE, subtype="PCM_16")
            row = {"id": stem, "file": file, "split": lab["split"], "deriv_type": lab["deriv_type"],
                   "error": lab["error"], "line": k, "n_lines": n, "latex": latex, "spoken": spoken,
                   "wav": str(wav), "start_s": round(s / sr, 3), "end_s": round(e / sr, 3),
                   "duration_s": round((e - s) / sr, 3), "split_params": params}
            rows.append(row if verify is None else {**row, "asr": hyp, "wer": wer})
    out.mkdir(parents=True, exist_ok=True)
    (out / "manifest.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
    return rows, problems


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", type=Path, default=REC_DIR)
    ap.add_argument("--out", type=Path, default=REC_DIR)
    ap.add_argument("--min-gap", type=float, default=1.2)
    ap.add_argument("--rel-thresh", type=float, default=0.3)
    ap.add_argument("--check-asr", action="store_true")
    args = ap.parse_args()
    verify = asr_verifier() if args.check_asr else None
    rows, problems = ingest(args.src, args.out, args.min_gap, args.rel_thresh, verify)
    files = sorted({r["file"] for r in rows})
    by_split = {s: len({r["file"] for r in rows if r["split"] == s}) for s in ("calib", "test")}
    print(f"ingested {len(files)} files, {len(rows)} lines {by_split} -> {args.out / 'manifest.jsonl'}")
    for p in problems:
        print("PROBLEM", json.dumps(p))
    if args.check_asr:
        wers = np.array([r["wer"] for r in rows])
        print(f"ASR check on kept lines: median WER {np.median(wers):.2f}, mean {wers.mean():.3f}, max {wers.max():.2f}")


if __name__ == "__main__":
    main()
