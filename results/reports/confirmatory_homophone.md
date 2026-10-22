# Confirmatory run: class-conditional δ on a fresh homophone test set

This run follows the committed pre-registration `results/reports/prereg_homophone.md` (commit **8dc764e**). The prereg was written and committed before the fresh set existed. Its thresholds, endpoints and run procedure were applied unchanged. No item was excluded, and there were no post-hoc bug fixes. All numbers are reported as they came out. One difference from the prereg wording is disclosed: the prereg says "stratified exactly as the existing set", but the generator fixes only type and position. Type (20 per type) and position (21 final / 79 intermediate) match; direction does not quite (ty-said-for-teen 64 / teen-said-for-ty 36 against 59/41 on the existing set), because how often each direction occurs depends on what the generator can produce per problem (e.g. quadratic_formula yields only 22 teen-said-for-ty candidates). This is a generator-driven difference, not a choice made after seeing outputs; the item file was hashed before synthesis.

## Result

**Primary endpoint (SCR at the class-conditional δ, per voice):**

| voice | silent corrections | SCR | bound k/(n+1) = 10/201 | beta-binomial p, Beta(10, 191) | verdict (prereg rule) |
|---|---|---|---|---|---|
| v1 | **2 / 100** | 0.02 | 0.0498 | 0.93 | consistent with the bound |
| v2 | **3 / 100** | 0.03 | 0.0498 | 0.82 | consistent with the bound |

v1 and v2 are the same 100 derivations re-read by two voices. They are a sensitivity check, not replicates. No significance claim is made beyond the prereg rule (p ≥ 0.05 means "consistent with the bound").

**Secondary endpoints (descriptive only; same 100 items per voice):**

| | v1 | v2 |
|---|---|---|
| SCR, no gate (δ = +∞, augmented) | 91/100 | 93/100 |
| SCR, pooled δ = 0.5519 alone | 7/100 | 9/100 |
| SCR, class-conditional (δ_hom = 0.5395 for homophone-class candidates, δ_pooled otherwise) | **2/100** | **3/100** |
| error-line statuses, class-conditional | math_error 97, corrected 2, unchecked 1, ask_back 0, verified 0 | math_error 95, corrected 3, unchecked 2, ask_back 0, verified 0 |
| error-line statuses, pooled δ | math_error 92, corrected 7, unchecked 1 | math_error 89, corrected 9, unchecked 2 |
| error-line statuses, no gate | corrected 91, math_error 9 | corrected 93, math_error 6, unchecked 1 |
| flags on valid lines (lines 1 … t−1), every pass | 5/128 (all math_error: 3 previous output misheard, 2 line misheard) | 3/128 (all math_error: previous output misheard) |

- Every silent correction at the class-conditional δ went to a homophone-class candidate: 2 for v1, 3 for v2. Their Δ values are between 0.509 and 0.538.
- Compared with the pooled-δ pass, the class rule changed 11 error lines (5 v1, 6 v2) from `corrected` to `math_error`, and no other line. The flags on valid lines are the same in all three passes.
- "Misrecognised into valid maths" (ĉ itself valid and different, which δ cannot control) was 0/100 for both voices.
- For reference only, the pooled-δ pass against its own pooled bound (k/(n+1) = 16/328 = 0.049; Beta(16, 312)) gives beta-binomial p = 0.24 (v1) and 0.08 (v2). The prereg does not test this.

Other descriptive numbers (class-conditional pass):

- CER / SymEq: 0.106 / 0.854 (v1), 0.097 / 0.863 (v2).
- Mishearings: 3 (v1) and 1 (v2), too few for a fix rate to mean anything.
- Median latency per line ≥ 1: 2.34 s (v1), 2.15 s (v2). These timings were taken while the demo shared the GPU.

## What it means

- **The pooled bound controls the mixture of error types, not each type.** Pooled δ calibrates over all error types together, so a risky class can exceed α. On the existing homophone test it did: 11/100, p = 0.02.
- **Class-conditional (Mondrian) conformal calibration fixes this.** It uses one threshold per error class, each calibrated on that class's calibration errors only. On a fresh, prespecified homophone set it gives 2/100 and 3/100, consistent with the 5% bound for both readings.
- **The no-gate baseline shows how much δ does.** Without the gate, augmentation silently "corrects" 91 and 93 of 100 homophone mistakes. The class-conditional δ removes 89 and 90 of those.
- **What it does not show:**
  - The errors are researcher-constructed: generated -teen/-ty swaps read by cloned TTS voices. Real dictation is not a sample from this mix.
  - The bound assumes the fresh items are exchangeable with the 200 homophone calibration errors. That holds only approximately, by design: both sets come from the same generator, but the fresh set uses a different seed with the 600 existing problems excluded and a slightly different direction mix (see above), and the 200 calibration errors are 100 derivations each read by v1 and v2, i.e. paired readings, not 200 independent draws (see the pairing section of `phase4.md`). It does not hold in the wild.
  - **Scope of the bound under the class rule.** δ_hom is calibrated on the margins of homophone-class errors, and the 10/201 bound covers silent corrections to homophone-class candidates. On a homophone error line, a candidate *outside* the homophone class is still gated at δ_pooled = 0.5519, so one with Δ between 0.5395 and 0.5519 could silently correct the line without being covered by the 10/201 bound. This did not happen here: every silent correction in the class-conditional pass went to a homophone-class candidate.
  - v1/v2 are not independent.

## Runtime class rule

A candidate c is in the homophone class iff c is a single confusable-number swap (13↔30 … 19↔90, `src/augment.py` boundary rules) of the best-sounding candidate ĉ, compared with `normalise()`. This is `src/decide.is_homophone_of` and `tag_homophone`. Homophone-class candidates are compared with δ_hom = 0.5395088972175274. Every other candidate is compared with δ_pooled = 0.5518526991984581. With `delta_hom=None`, `decide()` behaves exactly as before; the full test suite passes.

## Procedure as run (all times UTC, Thu 22 Oct 2026; cutoff 18:00 UTC = 02:00 SGT)

| step | what | time |
|---|---|---|
| generate | `scripts/gen_homophone.py --fresh`: same generator (`src/homophone.py`), seed 4242. The generator was run with the 600 existing problem statements (400 first-pass + 200 homophone) excluded. Its stratified "test" half was kept: 100 items, ids `homf-0000…0099`, set `homophone_fresh`. 0 line-0 overlaps. Stratification: 20 per type; position final 21 / intermediate 79 (the existing test set also has 21/79); direction ty-said-for-teen 64 / teen-said-for-ty 36 (existing 59/41). | 12:51–13:13 |
| lock | sha256 of the item file, `scripts/run_homfresh.py`, `src/decide.py`, `scripts/run_derivations.py` and the prereg recorded before synthesis. One later edit to `run_homfresh.py` (figure label text and axis limit only) was made and logged while synthesis was still running, before any output existed. | 13:13 |
| synthesise | `synthesize_derivations.py --items-file …homophone_fresh.json`, v1 + v2, 656 clips, 0 failures. LLM verbalisation said the said number on 100/100 error lines per voice. | 13:13–13:40 |
| QC + repair | `tts_redo.py --rounds 3 --id-prefix homf-`. 30/656 clips imperfect before repair, 2 after. Neither is an error line: v1 homf-0029 line 3 ("four x" heard as "for X"), v2 homf-0065 line 0 ("four x" → "for x"). Both are kept: the prereg excludes no item. `check_homophone_audio.py`: Whisper hears the said number on **100/100** error-line clips per voice; none fail. | 13:40–13:47 |
| stage 1 | `run_derivations.py stage1 --items-file …`: augmented pools for 656 clips, 0 failures | 13:48–13:52 |
| stage 2 | `run_homfresh.py run`: three sequential passes (class, pooled, +∞), about 100 s each | 13:57–14:02 |
| endpoints | `run_homfresh.py report --log` (run twice: 14:03 and again at 14:04:49, the `report.json` mtime; same script hash, output identical, no duplicate rows in `results.json`) | 14:03–14:05 |

The confirmatory run was complete at 14:02 UTC, about 4 h before the cutoff.

## Files

- `data/derivations/homophone_fresh.json`: the fresh set. `homophone.json` was not modified.
- `data/cache/derivations/gpt-4.1-mini/phase4/homfresh/records_{class,pooled,inf}.jsonl`, `report.json`, `run_timing.json`.
- `results/results.json` rows 60–65: `deriv_test_homfresh_v1` / `_v2`, systems `D+decide(class)` (primary, with `bound_test`), `D+decide (pooled δ only)`, and `D+decide (no gate, δ=+inf)`.
- `results/figures/scr_tradeoff.png`: the headline figure (see the caption). The earlier pooled fix-rate/SCR figure is now `results/figures/scr_tradeoff_pooled.png`.
- Code:
  - `src/decide.py` (`delta_hom`, `is_homophone_of`, `tag_homophone`);
  - `scripts/run_homfresh.py`;
  - `--fresh` in `scripts/gen_homophone.py`;
  - `--items-file` in `run_derivations.py stage1` and `check_homophone_audio.py`;
  - `tts_qc.all_errors` reads the fresh file.
- Tests: `tests/test_decide.py` (class rule).
