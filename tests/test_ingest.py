"""Phase 3.6 ingest: silence splitting and the labels -> manifest pipeline, on generated tones."""

import csv
import importlib.util
import json
from pathlib import Path

import numpy as np
import soundfile as sf

from src.audio import split_on_silence
from src.spoken_norm import normalise_spoken

SR = 16_000
ROOT = Path(__file__).resolve().parents[1]
_spec = importlib.util.spec_from_file_location("ingest_recordings", ROOT / "scripts" / "ingest_recordings.py")
ingest_mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(ingest_mod)


def _tone(seconds, freq=220.0, amp=0.3):
    t = np.arange(int(seconds * SR)) / SR
    # amplitude-modulated so it is speech-like (syllable rate ~4 Hz), never fully silent inside a "line"
    return (amp * (0.6 + 0.4 * np.sin(2 * np.pi * 4 * t)) * np.sin(2 * np.pi * freq * t)).astype(np.float32)


def _noise(seconds, amp=0.003, seed=0):
    return (amp * np.random.default_rng(seed).standard_normal(int(seconds * SR))).astype(np.float32)


def _recording(line_lengths, gap=2.0, breath=False):
    parts = [_noise(0.5)]
    for i, n in enumerate(line_lengths):
        parts.append(_tone(n, freq=200 + 40 * i))
        if i < len(line_lengths) - 1:
            if breath:   # short loud-ish burst in the middle of the pause
                parts += [_noise(gap / 2 - 0.1, seed=i), _noise(0.1, amp=0.1, seed=99), _noise(gap / 2, seed=i + 7)]
            else:
                parts.append(_noise(gap, seed=i))
    parts.append(_noise(0.5, seed=42))
    return np.concatenate(parts)


def test_split_counts_lines():
    segs = split_on_silence(_recording([2.0, 1.5, 3.0, 1.0]))
    assert len(segs) == 4
    durs = [(e - s) / SR for s, e in segs]
    for got, want in zip(durs, [2.0, 1.5, 3.0, 1.0]):
        assert abs(got - want) < 0.5


def test_short_pause_inside_line_not_split():
    audio = np.concatenate([_noise(0.5), _tone(1.0), _noise(0.6), _tone(1.0), _noise(2.0), _tone(1.0), _noise(0.5)])
    assert len(split_on_silence(audio)) == 2


def test_breath_in_gap_ignored():
    assert len(split_on_silence(_recording([1.5, 2.0, 1.5], breath=True))) == 3


def test_level_independent():
    quiet = _recording([1.5, 1.5, 1.5]) * 0.05
    assert len(split_on_silence(quiet)) == 3


def test_normalise_spoken():
    assert normalise_spoken("x squared minus 5x plus 6 equals 0.") == "x squared minus five x plus six equals zero"
    assert normalise_spoken("-4x + 6 = 19") == normalise_spoken("negative four x plus six equals nineteen")
    assert normalise_spoken("x ≤ 1/2") == normalise_spoken("x is less than or equal to one over two")


def test_ingest_end_to_end(tmp_path):
    src, out = tmp_path / "rec", tmp_path / "out"
    src.mkdir()
    lab = [("d001.wav", "calibration", "", "", ["2x = 4", "x = 2"]),
           ("d021.wav", "test", "sign_flip", "2", ["x + 1 = 3", "x = 4", "x = 4"]),
           ("d022.wav", "test", "", "", ["x = 1", "x = 1"])]          # wav has the wrong line count
    with open(src / "recordings_labels.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["file", "deriv_id", "split_suggestion", "deriv_type", "error_type", "error_step",
                    "n_lines", "line_idx", "latex", "spoken_text"])
        for file, split, et, es, lines in lab:
            for k, latex in enumerate(lines, 1):
                w.writerow([file, file[1:4], split, "linear", et, es, len(lines), k, latex, latex])
    sf.write(src / "d001.wav", _recording([1.5, 1.0]), SR)
    sr44 = 44_100   # stereo 44.1 kHz input must be normalised to 16 kHz mono
    stereo = np.stack([_recording([1.0, 1.0, 1.0])] * 2, axis=1)
    sf.write(src / "d021.wav", np.interp(np.arange(0, len(stereo), SR / sr44), np.arange(len(stereo)),
                                         stereo[:, 0])[:, None].repeat(2, 1), sr44)
    sf.write(src / "d022.wav", _recording([1.0, 1.0, 1.0]), SR)

    rows, problems = ingest_mod.ingest(src, out)
    assert [(p["file"], p["problem"]) for p in problems] == [("d022.wav", "3 segments for 2 labelled lines")]
    assert [(r["id"], r["line"], r["split"]) for r in rows] == [
        ("d001", 0, "calib"), ("d001", 1, "calib"), ("d021", 0, "test"), ("d021", 1, "test"), ("d021", 2, "test")]
    assert rows[2]["error"] == {"type": "sign_flip", "line": 1}   # error_step is 1-based in the CSV
    assert rows[0]["error"] is None
    for r in rows:
        info = sf.info(r["wav"])
        assert info.samplerate == SR and info.channels == 1
    manifest = [json.loads(line) for line in (out / "manifest.jsonl").read_text().splitlines()]
    assert manifest == rows
    assert (out / "lines" / "d021" / "line_2.wav").exists()


def test_ingest_verifier_rejects_misaligned(tmp_path):
    src = tmp_path / "rec"
    src.mkdir()
    with open(src / "recordings_labels.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["file", "deriv_id", "split_suggestion", "deriv_type", "error_type", "error_step",
                    "n_lines", "line_idx", "latex", "spoken_text"])
        w.writerows([["d001.wav", 1, "", "linear", "", "", 2, k, "x = 1", "x equals one"] for k in (1, 2)])
    sf.write(src / "d001.wav", _recording([1.5, 1.0]), SR)

    rows, problems = ingest_mod.ingest(src, tmp_path / "ok", verify=lambda c, s: (True, [("x", 0.0)] * len(c)))
    assert len(rows) == 2 and rows[0]["split"] == "calib" and rows[0]["wer"] == 0.0 and not problems
    rows, problems = ingest_mod.ingest(src, tmp_path / "bad", verify=lambda c, s: (False, [("?", 1.0)] * len(c)))
    assert rows == [] and problems[0]["problem"].startswith("segments do not match")
    assert not (tmp_path / "bad" / "lines" / "d001").exists()
