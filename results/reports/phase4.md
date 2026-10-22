# Phase 4: decision rule, δ calibration, full derivation-mode evaluation

Pipeline per line: Whisper large-v3-turbo 5-best → `build_pool` (gpt-4.1-mini, context variables) → `score_pool` →
frozen System D (λ = 1, γ = 0.25, ε = 0.5, from `results/tuned.json`, not retuned) → `checker.check_step` of every
candidate against the system's own previous output → `decide.decide` at the calibrated δ. Code:
`scripts/run_derivations.py` (stage 1 pools, sequential stage 2), `scripts/run_phase4.py` (calibration, decision pass,
ablation, report). Results rows 33-41 in `results/results.json`. All numbers below are on the **test split** unless
stated otherwise.

Sources: v1, v2 = clean cloned synthetic voices (LibriSpeech speakers); v3 = accented cloned voice (L2-ARCTIC TXHC,
100-derivation subset, 50 of them test); rec = the owner's own recordings.

## 1. Calibration log (calib split only; no test outcome was read before δ was fixed)

| item | value |
|---|---|
| calibration set | spoken-error line items of the calib split: v1 60 + v2 60 + rec 7 = **127** |
| excluded for qc_wer > 0 | **0** (none of the 8 imperfect v1/v2 clips is a calib error line; v3 is held out of calibration) |
| n | 127 |
| α | 0.05 |
| k = ⌊α(n+1)⌋ | ⌊0.05 × 128⌋ = **6** |
| margins m_i | **all 127 are +∞** (0 finite, 0 zero) |
| δ = m_(6) | **+∞** |
| fixed-point check | pass at δ = 0 → δ = +∞; pass re-run at δ = +∞ → δ = +∞ (converged after 2 passes) |

**Why δ = +∞.** m_i is the score gap to the nearest *valid* candidate that differs from what was said. No calibration
error line had any valid alternative in its candidate pool. Most error-line pools are tiny: 191 of 253 error items at
δ = 0 had a single candidate, because the 5 beams agree on clean speech. The remaining alternatives are variants of the
same wrong step, so they are also errors. With every margin at +∞, the 6th smallest is +∞. The rule may then correct a
line whenever exactly one valid reading exists in the pool. **What the bound says:** for an exchangeable test error,
P(m_test < δ) ≤ k/(n+1) = 6/128 = 0.047. In words: at most 4.7% of spoken errors would have a valid alternative in the
pool. The test split matches: 0 of 141 test error items (all sources, v3 included) had a finite margin.

## 2. Headline: D+decide at δ = +∞, per source group

| group | derivations | lines | CER ↓ | SymEq ↑ | err items | P (adj.) | R | F1 | P (strict) | SCR ↓ | misrec. into valid | mishearings | fix rate | detectable | fix rate (det.) | ask-back | caution | flag rate | latency median |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| overall (v1+v2+v3+rec) | 470 | 2351 | 0.1229 | 0.8856 | 141 | 0.486 | 0.979 | 0.649 | 0.421 | **0/141 = 0.000** | 0.000 | 127 | 0.110 | 107 | 0.065 | 0.000 | 0.047 | 0.174 | 2.02 s |
| synthetic v1+v2 | 400 | 2026 | 0.1264 | 0.8825 | 120 | 0.482 | 0.992 | 0.649 | 0.418 | **0/120 = 0.000** | 0.000 | 112 | 0.098 | 93 | 0.054 | 0.000 | 0.049 | 0.175 | 2.03 s |
| accented v3 | 50 | 257 | 0.1097 | 0.8911 | 15 | 0.455 | 1.000 | 0.625 | 0.385 | 0/15 = 0.000 | 0.000 | 14 | 0.143 | 13 | 0.077 | 0.000 | 0.048 | 0.188 | 1.97 s |
| owner's recordings | 20 | 68 | 0.0468 | 0.9559 | **6** | 1.000 | 0.667 | 0.800 | 1.000 | **0/6 = 0.000** | 0.000 | 1 | 1/1 | 1 | 1/1 | 0.000 | 0.000 | 0.083 | 2.01 s |

How to read the table:

- CER and SymEq compare the final output line with what was actually said; SymEq coverage is 1.0. Lines include line 0.
- P/R/F1 count only `math_error` as a flag:
  - TP is a flag on an "error" line, FP a flag on a "valid" line.
  - Adjusted P ignores flags on follows_error lines. Strict P counts them as FP.
- Detection counts:

  | group | TP | FN | FP | FP on follows_error |
  |---|---|---|---|---|
  | overall | 138 | 3 | 146 | 44 |
  | synthetic | 119 | 1 | 128 | 38 |
  | v3 | 15 | 0 | 18 | 6 |
  | recordings | 4 | 2 | 0 | 0 |

- **ask_back never fired** (0 lines), so P/R with ask_back counted as a flag are identical to the table.
- Corrected lines (not a flag): 21 overall (18 synthetic, 2 v3, 1 recordings).
- Unchecked lines: 25 overall (18 synthetic, 1 v3, 6 recordings).
- Caution:
  - The caution rate counts lines 1 and later whose output carries a caution note. Caution is never a flag.
  - **All 89 test cautions were multiplications by a nonzero denominator** (for example, "multiplying by x − 3 is only allowed because x − 3 ≠ 0 here").
- Flag rate is the share of lines 1 and later flagged `math_error` or `ask_back`.
- Recordings: n = 20 derivations, 6 error items, 1 mishearing. These are reported for completeness, with **no significance claim**.

### SCR three ways (project-specific metric, lower is better)

| | silent / spoken-error items | SCR |
|---|---|---|
| overall test (v1+v2+v3+rec) | 0 / 141 | 0.000 |
| synthetic voices (v1+v2) | 0 / 120 | 0.000 (v3 alone: 0 / 15) |
| owner's recordings | 0 / 6 (n = 6 error derivations; no significance claim) | 0.000 |

**Exchangeability.** The guarantee assumes calibration and test errors are exchangeable. Calibration is dominated by
synthetic speech (120 of 127 items), while the claim is about real speakers. That is why SCR is broken down by
source above.

**Decomposition of SCR.**

- Silent corrections split into two kinds:
  - Rule-2 corrections of an error line. These can only happen when m_i < δ, and that is the event the conformal bound covers.
  - Errors misrecognised into valid maths: ĉ is valid but differs from what was said. These are silent at every δ, so δ cannot control them.
- Test counts, all sources:
  - bound event {m_i < δ}: 0/141;
  - rule-2 silent corrections: 0;
  - misrecognised into valid maths: 0/141 (0.000).

  The uncontrollable part is therefore zero on this data.
- **Caveat:** the injected and deliberate errors here are algebra slips: a dropped term, a sign, a bad expansion, an
  extraneous root. Their correct versions do not sound like what was said. An error the speaker made that is
  acoustically close to the correct value, such as saying "fifty" where "fifteen" was right, is exactly the case where
  a silent correction could happen. No such error is in the calibration set, so SCR = 0 says nothing about that kind.

## 3. Ablation on derivation test lines (dataset `deriv_test`: v1+v2+v3+rec, 470 derivations, 2351 lines)

A–D use plain selection with frozen params and no decision rule. Their error-detection column comes from running the checker on each system's own output against its own previous output, flagging `error` as `math_error` (no correction).

| system | CER ↓ | SymEq ↑ | P (adj.) | R | F1 | P (strict) | TP / FN / FP / FP-follows | median latency |
|---|---|---|---|---|---|---|---|---|
| A (top beam → LLM top-1) | 0.1281 | 0.8732 | 0.445 | 0.979 | 0.612 | 0.376 | 138 / 3 / 172 / 57 | 1.02 s |
| B (A + parse filter) | 0.1280 | 0.8745 | 0.441 | 0.979 | 0.608 | 0.373 | 138 / 3 / 175 / 57 | 1.02 s |
| C (beam + validity bonus) | 0.1258 | 0.8796 | 0.440 | 0.986 | 0.608 | 0.373 | 139 / 2 / 177 / 57 | 1.11 s |
| D (C + phoneme + rank prior) | 0.1245 | 0.8796 | 0.459 | 0.979 | 0.625 | 0.390 | 138 / 3 / 163 / 53 | 1.96 s |
| **D+decide (δ = +∞)** | **0.1229** | **0.8856** | **0.486** | 0.979 | **0.649** | **0.421** | 138 / 3 / 146 / 44 | 2.02 s |

- A–D latencies follow the `run_ablation.py` convention and exclude the checker.
- D+decide adds the checker and decide time (median 0.04 s) and is median over all lines.
- The decision rule adds 21 corrections. These lower CER and raise SymEq and precision, because a corrected line no longer flags the next line.

## 4. Native vs accented (`results/figures/native_vs_accented.png`)

This compares the same 50 test derivations, 257 lines per voice, under D+decide.

| voice | CER | SymEq | error items | P (adj.) | R | SCR | mishearings | fix rate |
|---|---|---|---|---|---|---|---|---|
| v1 | 0.1178 | 0.8833 | 15 | 0.500 | 1.000 | 0/15 | 16 | 0.250 |
| v2 | 0.1253 | 0.8794 | 15 | 0.484 | 1.000 | 0/15 | 15 | 0.133 |
| v3 (accented) | 0.1097 | 0.8911 | 15 | 0.455 | 1.000 | 0/15 | 14 | 0.143 |

The accented cloned voice is not worse than the native voices on this subset. The differences are within a few lines
(n = 15 error items and about 15 mishearings per voice), so no significance claim is made. v3 is a cloned voice, so its
accent is milder than a real L2 speaker's.

## 5. Fix rate and where flags on correct lines come from

- **Mishearings.** There are 127 test mishearings: valid or follows_error lines where ĉ ≠ what was said.
  - Fix rate is 0.110 overall. On the 107 detectable ones (ĉ flagged error) it is 0.065.
  - Fix rate is low because what was said is rarely in the pool when ĉ is wrong. It is there for only 16 of 127, and is a valid candidate for only 7 of the 107 detectable ones.
  - δ cannot fix what the pool does not contain.
- **Two-solution lines.** 87 of the 127 mishearings are "x = a or x = b" lines, out of 282 such lines in test. The LLM reads "or" as a choice between readings and drops one solution.
- **SymEq artefact.**
  - 7 corrected outputs of the form `x = \frac{a \pm b}{c}` are counted as *not* fixed, because `symeq` does not equate ± with "x = … or x = …".
  - Counting them as fixed would raise the overall fix rate from 0.110 to 0.165.
  - The table keeps the conservative number.
- **False-positive flags on valid lines (146).**
  - 75: the line was heard right, but the previous output was a mishearing.
  - 70: the line itself was misheard and the previous output was right.
  - 1: both were wrong.
  - None: both the line and the previous output were right.

  So every math_error flag on a correct step traces back to ASR or LLM, never to the checker. This agrees with the checker-only result (adjusted P = 1.000 on ground-truth text, `checker_samples.md`).
- **Misses (FN 3).**
  - v1 qform-0072 line 1 was heard as `...18xK` and came out unchecked. It is one of the qc_wer > 0 clips, kept in test as specified.
  - rec d028 line 2 was heard as `1X=94`: the capital X reads as a second variable, so the line is unchecked.
  - rec d036 line 1: the speaker said "Dividing both sides by x" aloud, the narration was transcribed into the line, and the line could not be read.

## 6. Latency (median per line, lines ≥ 1, test)

| part | ASR | to_latex (slowest call) | to_spoken (slowest call) | phoneme GPU | checker | decide | total |
|---|---|---|---|---|---|---|---|
| median s | 0.13 | 0.94 | 0.79 | 0.016 | 0.039 | <0.001 | **1.96** (synthetic 1.97, v3 1.90, recordings 2.13) |

These are the original, uncached call times from the LLM cache, the same convention as `run_ablation.py`, measured on an RTX 4090.

## 7. Example traces (δ = +∞)

Each trace lists one line per row as line, ground-truth label → status, then said / heard / output.

**Clean, v1 lin-0000**

| line | label → status | said / output |
|---|---|---|
| 0 | given → given | `6(x+4)+1=4x+21` |
| 1 | valid → verified | `6x + 24 + 1 = 4x + 21` |
| 2 | valid → verified | `2x + 25 = 21` |
| 3 | valid → verified | `2x = -4` |
| 4 | valid → verified | `x=-2` |

**Corrected mishearing, v1 lin-0018**

| line | label → status | detail |
|---|---|---|
| 0 | given → given | `2(x-6)-2=4x-25` |
| 1 | valid → **corrected** | heard `2x - 12 - 2 = 4(x - 25)`. The checker said "the solutions change from x = 11/2 to x = 43", and exactly one valid reading was in the pool, so it wrote `2x - 12 - 2 = 4x - 25`, which matches what was said. |
| 2–3 | valid → verified | `-2x - 14 = -25`, `-2x = -11` |
| 4 | valid → **unchecked** | heard `X=\frac{11}{2}`. The capital X reads as a new variable: not flagged, but not verified either. |

**Math error, v2 lin-0004 (injected dropped term)**

| line | label → status | detail |
|---|---|---|
| 0 | given → given | `6(x-2)+8=-4x-17` |
| 1 | error → **math_error** | `6x - 12 = -4x - 17`. Explanation: "the term 8 seems to have been dropped; the solutions change from x = -13/10 to x = -1/2". The user's line is kept as said. |
| 2–4 | follows_error → verified | `10x - 12 = -17`, `10x = -5`, `x = -\frac{1}{2}`: each follows correctly from the mistake. |

**Ask-back.** There is no example: ask_back fired on 0 of 2351 test lines. No pool ever held two non-equivalent valid
readings of an invalid ĉ.

**Owner's recordings**

| derivation | line | label → status | detail |
|---|---|---|---|
| d030 | 0 | given → given | `-6x + 4 < 3` |
| d030 | 1 | valid → verified | `-6x < -1` |
| d030 | 2 | valid → **corrected** | heard `x < -\frac{1}{6}` ("solutions change from x > 1/6 to x < -1/6"), output `x > \frac{1}{6}`, which matches what was said |
| d025 | 1 | error → **math_error** | "the term 90 seems to have been dropped; the solutions change from x = 60 to x = -30" |
| d025 | 2 | follows_error → verified | `x = -30` |

## 8. Files

- `results/margins.json` has the calibrated δ, n and k, and the calibration log, with a "schema" key at the top level. It contains:
  - 283 spoken-error items (calib and test, all sources): m_i, c_hat_valid_and_different, excluded_qc and in_calibration;
  - 258 mishearing items, with a compact record that `src.calibrate.simulate(record, δ)` replays at any δ (for the notebook slider).
- `results/figures/scr_tradeoff.png`:
  - left: fix rate against SCR per source group, with the calibrated point and the 5% line;
  - right: both rates against δ.
- `results/figures/native_vs_accented.png`.
- Per-line records are in `data/cache/derivations/gpt-4.1-mini/phase4/records.jsonl` (calibrated run), with `calib_records.jsonl`, `calibration.json`, `ablation.json` and `report.json` alongside.
- `tests/test_phase4_e2e.py` runs 3 real derivations end to end from the cached pools: clean v1 lin-0000, injected error v2 lin-0004 and recording d025. It also checks one corrected mishearing.

## 9. Limitations

- **δ = +∞ is degenerate.**
  - The calibration is valid, but it is not informative about the size of δ, only about whether valid alternatives occur at all.
  - SCR = 0 at every δ on this data, so the trade-off curve only shows fix rate rising with δ.
  - The guarantee does not cover errors that are acoustically close to the correct value (see the caveat in section 2).
- **Pools are the bottleneck.**
  - Most mishearings cannot be fixed because what was said never reaches the pool.
  - The main cause is two-solution lines.
- **context_vars deviation.**
  - Stage 1 passes the variables of the ground-truth line 0, not those of the system's own previous output. This keeps stage 1 independent of δ.
  - With own outputs, 3281 of 3362 lines at δ = 0 would have received the same set {x}. For 76 the previous output did not parse, so it has no variables, and 5 had a different variable set.
- **SymEq artefacts.** ± against "or", and ", \text{ or }", are not judged equal. This understates the fix rate (see section 5) and inflates the mishearing count.
- **Recordings.**
  - Spoken narration ("dividing both sides by x") and clipped line starts reach the transcript.
  - n = 20 test derivations with 6 error items: no significance claims.
  - For recordings, `qc_wer` is the manifest's Whisper WER, not a TTS QC score.
- **Imperfect clips in test.** 11 test clips have qc_wer > 0 and were kept as specified. Two of them are error items: v1 qform-0072:1 and one recording.
- **Approximate sweep.** The sweep re-scores stored pools whose candidate statuses were checked against the previous output of the calibrated run. Away from δ = +∞ it is therefore approximate. The headline numbers come from the real sequential run.
- **Synthetic voices.** v1, v2 and v3 are synthetic clones. v3's accent comes from a single reference clip of one L2 speaker.
