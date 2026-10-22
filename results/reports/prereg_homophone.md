# Pre-registration: class-conditional δ, fresh homophone test set

Written and committed **before** the fresh test set was generated, synthesised or scored. The git timestamp of this file precedes every artefact of the fresh set. Nothing below may be changed after the fresh set's outputs have been looked at; any bug fix made afterwards is reported with its before/after numbers.

## Why

The pooled calibration (δ = 0.5518527, n = 327, k = 16) controls the silent-correction rate of the *mixture* of error types. The risky class, homophone errors (the speaker says a number whose sound-alike would make the step valid), already falls below the pooled δ 15/200 = 7.5% of the time in calibration, and exceeded the bound on the existing homophone test set (11/100 v1, 10/100 v2). Calibrating δ per error class is class-conditional (Mondrian) conformal calibration: one threshold per class, each calibrated on that class's calibration errors only.

## Fixed quantities

| quantity | value | source |
|---|---|---|
| α | 0.05 | TASKS.md |
| δ_hom | **0.5395088972175274** | k = ⌊0.05 × 201⌋ = 10-th smallest margin of the 200 homophone calibration errors (calib split, v1 + v2, 0 excluded) |
| δ_pooled | **0.5518526991984581** | pooled calibration, n = 327, k = 16 (unchanged) |
| System D params | frozen (results/tuned.json) | Phase 2 |
| augmentation | on (confusable-number swaps, src/augment.py) | shipped system |

## Runtime class rule (used by the decision rule)

A pool candidate c is in the **homophone class** iff c is a single confusable-number swap (13↔30 … 19↔90, src/augment.py rules) of the best-sounding candidate ĉ, i.e. normalise(swap(ĉ)) == normalise(c) for some one-number swap. Homophone-class candidates use δ_hom; all other candidates use δ_pooled. Everything else in src/decide.py is unchanged (verified / corrected / ask_back / math_error / unchecked).

## Fresh test set

- 100 new homophone derivations from the same generator (scripts/gen_homophone.py, src/homophone.py) with a **new seed, 4242**, split label "test", stratified exactly as the existing set, and no problem statement (line 0) shared with any existing derivation (400 main + 200 homophone).
- Synthesised in voices v1 and v2 with the same verbaliser, QC (Whisper) and repair procedure (scripts/tts_redo.py, up to 3 rounds).
- **No test item is excluded.** QC results are reported; any error line where Whisper does not hear the said number after repair is listed.

## Primary endpoint

SCR on the fresh set's 100 spoken homophone errors at the class-conditional δ, reported **per voice** (v1 and v2 are the same derivations re-read: a sensitivity check, not replicates). Comparison to α = 5% with a beta-binomial p-value using Beta(k, n + 1 − k) = Beta(10, 191) for the calibration draw. A per-voice SCR whose upper tail p ≥ 0.05 is "consistent with the bound"; p < 0.05 is "bound exceeded". No significance claim beyond this.

## Secondary (descriptive only)

- SCR with no gate (δ = +∞) and with δ_pooled alone, same items, per voice.
- Status distribution on the error lines (math_error, ask_back, corrected, verified, unchecked).
- Flags on the fresh set's valid lines (line ≥ 1 before the error line).

## Run procedure

1. Generate (seed 4242) → synthesise → QC/repair → stage 1 (augmented pools) → stage 2 sequential decisions with the class rule above.
2. Compute the endpoints with src/calibrate.py metrics. Report all numbers whatever they are.
3. Hard cutoff: if the fresh set is not synthesised, QC'd and scored by 02:00 SGT Fri, the confirmatory run is cut and only the pooled results are reported.
