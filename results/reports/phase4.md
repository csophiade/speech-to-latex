# Phase 4: decision rule, δ calibration, full derivation-mode evaluation (final, second pass)

## Headline

The shipped system now adds **confusable-number variants** to every candidate pool (13↔30 … 19↔90, `src/augment.py`).
δ is calibrated on a pooled calibration set that includes a **homophone stress set**: the speaker says a number whose
sound-alike is the correct one, e.g. "fifty" where fifteen is right.

**On this test data, augmentation buys no measured benefit.** The non-augmented system weakly dominates the shipped
one on every measured axis. On the pooled test set it has the same or a higher fix rate (v1 8/58 = 0.138 vs 7/58 = 0.121; v2
7/59 = 0.119 vs 5/59 = 0.085) and SCR 0/160 instead of 11/160 and 10/160. The benefit augmentation is meant to bring
(recovering real -teen/-ty mishearings) is hypothetical here: there are none in the test set. The three operating points
on the 100 homophone test errors per voice are:

| system | homophone test SCR, v1 | homophone test SCR, v2 | what it means |
|---|---|---|---|
| no augmentation, δ = +∞ (first-pass system) | 0/100 | 0/100 | **cannot correct sound-alikes**: the right-sounding valid reading is never in the pool. No measured fix-rate cost on this data. |
| augmentation, no gate (δ = +∞) | **97/100** | **95/100** | **corrects sound-alikes indiscriminately**: almost every homophone mistake is silently "fixed". Its only extra "power" is these wrong corrections. |
| augmentation + calibrated δ = 0.552 (shipped) | **11/100** | **10/100** | **the gate cuts silent corrections to 11 and 10 per 100**, with no fix-rate gain over no augmentation |

The earlier "safety without power / power without safety / both" framing is retracted: it is not supported by this data.
What the experiment does show is narrower. *If* the pool contains a sound-alike reading, an ungated system silently
corrects nearly every homophone mistake, and the calibrated δ removes about 9 in 10 of those silent corrections. For the
slide: "δ is what makes adding sound-alike candidates safe enough to ship; on this data adding them has not yet paid off."

There are three things this does not show:

- **The homophone stress number is above the 5% target.** The bound is calibrated for the *pooled* error mix: k/(n+1) = 16/328 = 4.9%.
  - On the pooled test set, SCR is 11/160 = 6.9% (v1) and 10/160 = 6.3% (v2). Both point estimates are above 4.9%, but the bound is **not rejected** (one-sided beta-binomial p = 0.20 and 0.28).
  - On the homophone subset alone, SCR is 11% and 10%. The bound is rejected there (beta-binomial p = 0.022 and 0.043).
  - The p-values use the beta-binomial because split-conformal miscoverage is itself random over the calibration draw, Beta(k, n+1−k) = Beta(16, 312). A plain Binomial(n, 0.049) treats 0.049 as exact and is anti-conservative: it gives 0.16/0.26 pooled and 0.010/0.024 homophone.
  - v1 and v2 are not two independent tests: they read the same derivations.
  - The pooled number is largely mechanical. All 60 first-pass errors per voice have margin +∞, so pooled SCR = homophone SCR × 100/160: the homophone number diluted.
  - The bound is marginal over the calibration mix, not a per-subset guarantee. 127 of the 327 calibration errors have margin +∞, and they dilute the homophone items: 15 of 200 homophone calibration items (7.5%) lie below δ.
  - So the sentence "even when the right answer sounds identical, silent corrections stay under the bound" is **not supported** and is not used anywhere in this report.
- **Augmentation did not raise the fix rate.**
  - No mishearing in any test group was fixed by an augmented candidate (0 everywhere).
  - No first-pass mishearing was a -teen/-ty swap.
  - On the first-pass test set (v1+v2+rec), the fix rate *falls* from 0.106 to 0.080. The cause is the finite δ: it refuses 3 corrections whose margin is ≥ 0.552, which δ = +∞ used to make. Augmentation itself is not the cause.
  - Refused corrections leave a misheard line in place, so the next correct line gets flagged. Cascade flags rise from 66 to 72.
- **The error mix is researcher-constructed.** It has two parts: injected algebra slips (first-pass set) and number-for-homophone swaps (stress set).
  - Neither is a sample of real students' mistakes.
  - The headline figures are rates on this mix only.

Figure: `results/figures/scr_tradeoff.png`.

- Left panel: fix rate against SCR on the pooled test set, per voice, with and without augmentation. The three real operating points are marked.
- Right panel: homophone-test SCR against δ, with the empirical CDF of the calibration margins (pooled, and homophone-only) that δ was read from.

Pipeline per line:

1. Whisper large-v3-turbo 5-best.
2. `build_pool` (gpt-4.1-mini, context variables) plus confusable-number variants.
3. `score_pool`.
4. Frozen System D (λ = 1, γ = 0.25, ε = 0.5, `results/tuned.json`, not retuned).
5. `checker.check_step` of every candidate against the system's own previous output.
6. `decide.decide` at δ.

Code: `scripts/run_derivations.py` (stage 1, augmentation, sequential stage 2) and `scripts/run_phase4.py` (calibrate, run, ablation, report). Results rows 42–59 in `results/results.json`. Rows 33–41 are first-pass rows, kept and marked `superseded: true`, except v3 (row 35, marked `pre_augmentation: true`).

**Sources:**

- v1, v2: clean cloned synthetic voices (LibriSpeech speakers). They read the *same* derivations, so they are not independent samples and are reported separately everywhere.
- rec: the owner's recordings.
- v3: accented voice. It is not re-run and keeps its first-pass, pre-augmentation numbers (section 8).

**Sets:**

- first-pass: the 400 derivations from before.
- homophone: `data/derivations/homophone.json`, 100 calib + 100 test, v1 and v2 only. The error is on line t ≥ 1, and the derivation ends at line t.

"Overall" never includes v3. All numbers are on the **test split** unless stated otherwise.

## 1. Calibration log (calib split only; δ was fixed before any test record was produced)

| item | value |
|---|---|
| calibration set | spoken-error items of the calib split, pooled: first-pass v1 60 + v2 60 + rec 7, homophone v1 100 + v2 100 = **327** |
| exclusions | **0** |
| synthetic exclusion rule | qc_wer > 0. The 2 imperfect homophone clips (v2 hom-0072 and hom-0074, line 0, "for" heard for "four") are not error lines, and no first-pass calib error clip has qc_wer > 0. |
| recordings | No label-quality signal exists. The manifest `wer` is raw Whisper WER of the owner's speech, which measures the ASR under test, so it is not used. All 7 recording calib errors are kept (all have wer = 0 anyway). |
| v3 | held out of calibration |
| n | 327 |
| α | 0.05 |
| k = ⌊α(n+1)⌋ | ⌊0.05 × 328⌋ = **16** |
| finite margins (final pass) | 185 of 327: homophone 185 of 200, first-pass 0 of 127 (all +∞, augmentation included) |
| δ = m_(16) | **0.5519** (item v2/hom-0073 line 1) |
| smallest margins | 0.504, 0.508, 0.509, 0.510, 0.515, 0.518, 0.519, 0.526, 0.536, 0.540, 0.540, 0.543, 0.545, 0.546, 0.551, **0.552**, 0.553, 0.556 … The nearest valid alternative is an augmentation variant in every one of them. |
| fixed-point check | Pass at δ = 0 gives δ = 0.5519. Re-run at 0.5519 gives 0.5519 (converged after 2 passes). |
| items below δ in calibration | 15 / 327 = 4.6% pooled; 15 / 200 = 7.5% of the homophone items |
| pairing | Calibration items are paired: v1 and v2 read the same derivations (160 v1/v2 pairs; 88 with both margins finite, Spearman ρ = 0.69, p = 7e-14). Several of the 16 smallest margins are both readings of one derivation (hom-0064, hom-0049, hom-0006, hom-0073, hom-0015). The effective sample is about **167 derivation clusters**, not 327. |
| cluster sensitivity (not adopted) | One margin per derivation, n = 167, k = 8: δ = 0.5395 using the smaller of the two readings, δ = 0.5596 using the larger. The shipped δ = 0.5519 lies between them. |

**Bound wording.** This is a calibrated bound under exchangeability. For a test error drawn like the pooled calibration errors,
P(m_test < δ) ≤ k/(n+1) = 0.049. It is not rejected on the pooled test set (6.9% for v1 and 6.3% for v2, each n = 160; both point estimates are above the bound; beta-binomial p = 0.20 and 0.28). It is exceeded on the homophone subset alone (11% and 10%; p = 0.022 and 0.043).

Pairing does not break the marginal bound when the test derivations are exchangeable with the calibration derivations. The bound is over derivations, with each reading being one draw. But δ is less precise than n = 327 suggests, because about 167 independent clusters stand behind it.

**Where the margins come from.** A variant inherits its parent's beam log-prob and pays the rank prior ε × 1 = 0.5. Its margin is therefore 0.5 plus γ = 0.25 times the difference in *z-normalised* phoneme scores. Those z-scores are relative to the pool: they are normalised over all the spoken forms in that pool. So the margin measures pool structure as much as acoustics.

- **Margin exactly 1.0 = a two-candidate pool.** All 49 homophone calibration items at margin 1.0 (and all 43 test items) come from pools of two candidates, the heard line and its one variant. In these pools each candidate's spoken forms phonemise identically or almost so: 40 of 49 calibration items and 33 of 43 test items have exactly two distinct phoneme scores. Two values z-normalise to +1 and −1 *however close the raw scores are*, so the margin is 0.5 + 0.25 × 2 = 1.0. The other 31 calibration and 32 test two-candidate pools have forms that differ in phonemes, and they get other margins. This is the jump at δ = 1 in the right panel of the figure. Example: v1/hom-0165/1 has raw phoneme scores −0.23 vs −0.62.
- **Margins near 0.5 come from larger pools.** Spurious candidates (e.g. `\text{quantity}`) widen the z-scale, which squeezes the heard-vs-variant gap towards 0. Near-0.5 margins come from this scaling, not from measured acoustic similarity.
- **The raw acoustics do separate the pair.** On raw (un-normalised) best-form phoneme scores, the scorer prefers what was heard over the nearest valid variant in 183/185 (98.9%) finite homophone calibration items and 192/192 test items. The median raw gap is 0.42 (10th–90th percentile 0.28–0.79).

So δ = 0.552 is a threshold on a pool-relative quantity: "correct to a sound-alike only when, after z-scaling across this pool, the heard form's phoneme advantage is under about 0.2 pool-z". It is not a statement about how well "fifteen" and "fifty" can be told apart. Nothing was retuned. A pool-independent acoustic term would be a Phase 2 change.

**Not adopted (for the owner to decide).** A homophone-only (class-conditional) calibration on the 200 homophone calibration items alone gives δ = 0.5395 (k = 10). It is computed from calibration data only. It targets the per-subset guarantee that the stress number asks for. It was **not** evaluated on test: choosing it after seeing the test result would make that evaluation post hoc, so adopting it needs a fresh homophone test set.

## 2. SCR three ways (calibrated δ = 0.552, real sequential run; SCR is project-specific, lower is better)

| | v1 | v2 | k/(n+1) |
|---|---|---|---|
| (a) homophone test subset (stress) | **11/100 = 0.110** | **10/100 = 0.100** | 0.049 |
| (b) full pooled test (first-pass + homophone) | **11/160 = 0.069** | **10/160 = 0.063** | 0.049 |
| (c) owner's recordings (n = 20 derivations, 6 error items; no significance claim) | 0/6 (rec) | | |

- **No-gate reference (δ = +∞, augmented):**
  - homophone subset: 97/100 for v1, 95/100 for v2;
  - pooled: 97/160 = 0.606 for v1, 95/160 = 0.594 for v2;
  - recordings: 0/6.
- **Not augmented (δ = +∞):** 0 everywhere.
- Every silent correction at the calibrated δ is a rule-2 correction to an augmentation variant: 11 for v1, 10 for v2.
- "Misrecognised into valid maths" (ĉ itself valid and different, which δ cannot control) is 0/100 and 0/160 for both voices.
- The first-pass algebra-slip errors still have no valid alternative in any pool: SCR is 0/60 for each voice at every δ.

## 3. Per-voice results at the calibrated δ (D+decide, shipped system)

| group | derivations | lines | CER ↓ | SymEq ↑ | err items | TP / FN / FP / FP-follows | P (adj.) | R | P (strict) | SCR | mishearings | fix rate | flag rate | median latency |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| homophone test v1 | 100 | 317 | 0.0856 | 0.8328 | 100 | 87 / 13 / 3 / 0 | 0.967 | 0.870 | 0.967 | 0.110 | 4 | 3/4 | 0.415 | 2.12 s |
| homophone test v2 | 100 | 317 | 0.0911 | 0.8233 | 100 | 88 / 12 / 3 / 0 | 0.967 | 0.880 | 0.967 | 0.100 | 1 | 1/1 | 0.419 | 2.03 s |
| pooled test v1 | 300 | 1330 | 0.1167 | 0.8744 | 160 | 146 / 14 / 72 / 24 | 0.670 | 0.913 | 0.603 | 0.069 | 58 | 0.121 | 0.235 | 2.05 s |
| pooled test v2 | 300 | 1330 | 0.1175 | 0.8624 | 160 | 148 / 12 / 74 / 19 | 0.667 | 0.925 | 0.614 | 0.063 | 59 | 0.085 | 0.234 | 1.96 s |
| first-pass test v1 | 200 | 1013 | 0.1269 | 0.8875 | 60 | 59 / 1 / 69 / 24 | 0.461 | 0.983 | 0.388 | 0/60 | 54 | 0.074 | 0.187 | 2.04 s |
| first-pass test v2 | 200 | 1013 | 0.1261 | 0.8746 | 60 | 60 / 0 / 71 / 19 | 0.458 | 1.000 | 0.400 | 0/60 | 58 | 0.069 | 0.185 | 1.95 s |
| owner's recordings | 20 | 68 | 0.0468 | 0.9559 | 6 | 4 / 2 / 0 / 0 | 1.000 | 0.667 | 1.000 | 0/6 | 1 | 1/1 | 0.083 | 2.32 s |
| first-pass overall v1+v2+rec (v3 excluded; v1/v2 not independent) | 420 | 2094 | 0.1246 | 0.8835 | 126 | 123 / 3 / 140 / 43 | 0.468 | 0.976 | 0.402 | 0/126 | 113 | 0.080 | 0.183 | 2.00 s |

How to read the table:

- CER and SymEq compare the output line with what was said. Lines include line 0.
- P/R count only `math_error` as a flag:
  - Adjusted P ignores flags on follows_error lines.
  - Strict P counts them as FP.
- **ask_back fired on 0 lines.** No pool held two non-equivalent valid readings below δ.
- The 13 and 12 homophone FNs are 11 and 10 silent corrections, plus 2 lines that came out unchecked.
- The homophone mishearing counts are tiny (4 and 1), so their fix rates mean nothing.
- **Fixed by an augmented candidate: 0 in every group.**

## 4. Before / after augmentation (first-pass test, same sources; superseded first-pass rows kept)

Columns: **pre** = no augmentation, δ = +∞ (the first-pass system); **aug ∞** = augmented, no gate; **aug δ** = augmented, calibrated δ = 0.552 (shipped).

The "pre" column was recomputed by stripping the variants from the cached pools (`strip_augmentation`). This reproduces the first-pass records exactly: 0 differences in status, output or candidate list over 4,206 line records.

| metric | v1 pre | v1 aug ∞ | v1 aug δ | v2 pre | v2 aug ∞ | v2 aug δ | rec (all three) |
|---|---|---|---|---|---|---|---|
| fix rate | 5/54 = 0.093 | 0.093 | **4/54 = 0.074** | 6/58 = 0.103 | 0.103 | **4/58 = 0.069** | 1/1 |
| SCR (first-pass errors) | 0/60 | 0/60 | 0/60 | 0/60 | 0/60 | 0/60 | 0/6 |
| flags on valid lines (FP) | 63 | 63 | 69 | 65 | 65 | 71 | 0 |
| … previous output misheard (cascade) | 31 | 31 | 34 | 35 | 35 | 38 | 0 |
| … line itself misheard | 31 | 31 | 34 | 30 | 30 | 33 | 0 |
| … both | 1 | 1 | 1 | 0 | 0 | 0 | 0 |
| CER | 0.1270 | 0.1270 | 0.1269 | 0.1258 | 0.1258 | 0.1261 | 0.0468 |
| SymEq | 0.8885 | 0.8885 | 0.8875 | 0.8766 | 0.8766 | 0.8746 | 0.9559 |

- **Totals for v1+v2+rec:**
  - fix rate: 0.106 → 0.106 → 0.080;
  - corrected lines: 19 → 19 → 10;
  - FP on valid lines: 128 → 128 → 140;
  - cascade flags: 66 → 66 → 72.
- The first-pass report's 146 FP and 75 cascade flags included v3 (18 FP). Without v3 the comparable figures are 128 and 66.
- Augmentation changes nothing on the first-pass test set at δ = +∞. The selected candidate ĉ is identical, and no first-pass error or mishearing gets a valid variant.
- Every change in the "aug δ" column comes from δ moving from +∞ to 0.552:
  - 3 correct fixes are refused: v1 lin-0018, v2 lin-0018 and v2 ineq-0022, with valid-reading margins of 1.0, 1.0 and 0.669, all above δ.
  - 6 other ungated corrections are also refused. They were corrections to a valid reading that was not what was said, so they were not fixes in the first place.
- **The owner predicted augmentation would recover mishearings and reduce cascade flags. That did not happen on this data, and both moved slightly the wrong way.**

The homophone set under the same three columns:

| metric | v1 pre | v1 aug ∞ | v1 aug δ | v2 pre | v2 aug ∞ | v2 aug δ |
|---|---|---|---|---|---|---|
| SCR | 0/100 | 97/100 | 11/100 | 0/100 | 95/100 | 10/100 |
| errors flagged (TP) | 98 | 3 | 87 | 98 | 4 | 88 |
| CER / SymEq | 0.0806 / 0.8675 | 0.1247 / 0.5678 | 0.0856 / 0.8328 | 0.0868 / 0.8517 | 0.1303 / 0.5615 | 0.0911 / 0.8233 |

## 5. Ablation on the main derivation test set (dataset `deriv_test`: first-pass test, v1+v2+rec, 420 derivations, 2094 lines; augmented pools)

A–D use plain selection with frozen params and no decision rule. Their detection column comes from running the checker on each system's own output against its own previous output.

| system | CER ↓ | SymEq ↑ | P (adj.) | R | F1 | P (strict) | TP / FN / FP / FP-follows |
|---|---|---|---|---|---|---|---|
| A (top beam → LLM top-1) | 0.1291 | 0.8725 | 0.447 | 0.976 | 0.614 | 0.380 | 123 / 3 / 152 / 49 |
| B (A + parse filter) | 0.1290 | 0.8739 | 0.442 | 0.976 | 0.609 | 0.376 | 123 / 3 / 155 / 49 |
| C (beam + validity bonus) | 0.1270 | 0.8792 | 0.444 | 0.984 | 0.612 | 0.378 | 124 / 2 / 155 / 49 |
| D (C + phoneme + rank prior) | 0.1262 | 0.8792 | 0.462 | 0.976 | 0.628 | 0.393 | 123 / 3 / 143 / 47 |
| **D+decide (δ = 0.552, shipped)** | **0.1246** | **0.8835** | **0.468** | 0.976 | **0.632** | **0.402** | 123 / 3 / 140 / 43 |
| *superseded:* D+decide, no augmentation, δ = +∞ | 0.1245 | 0.8849 | 0.490 | 0.976 | 0.653 | 0.426 | 123 / 3 / 128 / 38 |

- A–D are identical before and after augmentation on every metric. A and B never look at variants. C and D never select one on this set, because a variant inherits its parent's beam score and pays the rank prior.
- Per voice, CER / SymEq / P (adj.) for D → D+decide:
  - v1: 0.1273 / 0.8835 / 0.454 → 0.1269 / 0.8875 / 0.461;
  - v2: 0.1288 / 0.8707 / 0.458 → 0.1261 / 0.8746 / 0.458;
  - rec: 0.0499 / 0.9412 / 0.800 → 0.0468 / 0.9559 / 1.000.
- On the homophone test set, plain D flags 98 of 100 (v1) and 97 of 100 (v2) homophone errors. It never selects the variant, so it is not the selection that silently corrects. Only the δ rule can.

## 6. Latency (one definition)

**Definition:** median per-line wall time over test lines ≥ 1 (line 0 is never checked). It is the sum of these components:

- Whisper 5-best;
- the to_latex LLM call (slowest call, original uncached time);
- the to_spoken LLM call (same convention);
- GPU acoustic scoring of the augmented pool;
- the checker;
- decide.

It is measured on an RTX 4090 over the shipped run (v1, v2, rec, both sets).

| part | ASR | to_latex | to_spoken | phoneme GPU | checker | decide | total |
|---|---|---|---|---|---|---|---|
| median s | 0.13 | 0.94 | 0.80 | 0.017 | 0.082 | <0.001 | **2.01** |

- Per group: first-pass v1 2.04, v2 1.95, recordings 2.32, homophone v1 2.12, v2 2.03.
- **The first-pass report gave two medians, 2.02 and 1.96.** 2.02 was over all lines including line 0, and 1.96 was over lines ≥ 1. Only the lines ≥ 1 definition is used now.
- The checker median doubled (0.039 → 0.082 s) because each line has more candidates to check.
- The A–D ablation latencies (selection only, no checker) are left out of this report so that there is a single latency definition.

## 7. Example traces (calibrated δ = 0.552 unless stated)

**1. Homophone error refused, v1 hom-0138 (test).** The speaker says "fifty" where 15 is right.

| line | label → status | detail |
|---|---|---|
| 0 | given → given | `5(x+3)+8=x+20` |
| 1 | error → **math_error** | said and heard `5x + 50 + 8 = x + 20`. Augmentation put the valid `5x + 15 + 8 = x + 20` in the pool at Δ = 1.0 ≥ δ, so it is not used. Explanation: "the solutions change from x = -3/4 to x = -19/2". The user's line is kept. **With no gate (δ = +∞) the same line is silently "corrected" to `5x + 15 + 8`.** |

**2. Homophone error silently corrected (one of the 11), v1 hom-0141 (test).** The speaker says "sixty x" where 16x is right.

| line | label → status | detail |
|---|---|---|
| 0 | given → given | `x(x+4)=16x-27` |
| 1 | error → **corrected** (silent) | said and heard `x^2 + 4x = 60x - 27`. The variant `x^2 + 4x = 16x - 27` is valid at Δ = 0.512 < δ, so the output is the corrected line and the speaker's mistake is hidden. This is the residual risk that SCR measures. |

**3. Cost in power: a real mishearing no longer fixed, v1 lin-0018 (first-pass test).**

| line | label → status | detail |
|---|---|---|
| 1 | valid → **math_error** | said `2x - 12 - 2 = 4x - 25`, heard `2x - 12 - 2 = 4(x - 25)`. The valid reading (what was said) is in the pool at Δ = 1.0 ≥ δ, so it is flagged. At δ = +∞ (first pass) it was **corrected**. |
| 2 | valid → **math_error** (cascade) | `-2x - 14 = -25` is correct, but it is checked against the misheard line 1 ("check the expansion of 4(x - 25)"). |

**4. Injected algebra slip, v2 lin-0004.** This is unchanged by augmentation.

| line | label → status | detail |
|---|---|---|
| 1 | error → **math_error** | `6x - 12 = -4x - 17`: "the term 8 seems to have been dropped; the solutions change from x = -13/10 to x = -1/2". The variants (`-4x - 70` and others) are all errors too, so the margin is +∞. |
| 2–4 | follows_error → verified | each follows correctly from the mistake |

**5. Owner's recording, rec d025.**

| line | label → status | detail |
|---|---|---|
| 0 | given → given | `4x + 90 = 5x + 30`. Variants `4x + 19`, `5x + 13` are in the pool at Δ ≥ 0.95 and are not chosen. |
| 1 | error → **math_error** | `-1x=30`: "the term 90 seems to have been dropped; the solutions change from x = 60 to x = -30" |
| 2 | follows_error → verified | `x = -30` |

**Mishearing fixed thanks to augmentation:** none. There is no test line, in any group, where the output came from an augmentation variant and matched what was said.

## 8. Accented voice v3 (pre-augmentation; not re-run)

The owner asked for no further accented-voice work. v3 keeps its first-pass numbers: unaugmented pools, δ = +∞ from the first-pass calibration. They are not comparable with the shipped system's δ. `results/figures/native_vs_accented.png` is the first-pass figure.

| voice (50 test derivations, 257 lines) | CER | SymEq | error items | P (adj.) | R | SCR | mishearings | fix rate | median latency |
|---|---|---|---|---|---|---|---|---|---|
| v3, **pre-augmentation, δ = +∞** | 0.1097 | 0.8911 | 15 | 0.455 | 1.000 | 0/15 | 14 | 0.143 | 1.90 s |

## 9. Files

- **`results/margins.json`**: the pooled calibration (α, k, n, δ, exclusions, log) with a top-level "schema" key.
  - 653 spoken-error items (calib and test; v1, v2, rec; `set` first-pass|homophone), each with:
    - `m_i` (augmented pools) and `m_i_no_augmentation`;
    - `nearest_alt_augmented`;
    - `excluded_qc` and `in_calibration`;
    - `status_at_calibrated_delta`;
    - a compact record that `src.calibrate.simulate(record, δ)` replays. Its candidates carry an `augmented` flag.
  - 243 mishearing items.
- **`results/figures/scr_tradeoff.png`**: the headline figure. A copy of the first-pass figure is at `/tmp/claude-0/-workspace/14c433da-a855-4d21-a32b-b06750b0ea68/scratchpad/p4b/scr_tradeoff.pre_aug.png`.
- **`data/cache/derivations/gpt-4.1-mini/phase4/`**:
  - `calibration.json`, `calib_records.jsonl`;
  - `records.jsonl` (calibrated), `records_aug_inf.jsonl`, `records_noaug_inf.jsonl`;
  - `ablation.json`, `report.json`;
  - the first-pass copies `*_pre_aug.*`.
- **`tests/test_phase4_e2e.py`**: 5 real derivations at the calibrated δ:
  - clean;
  - injected error;
  - recording;
  - lin-0018, corrected only without a gate;
  - **homophone v1 hom-0138**, flagged at δ and silently corrected at +∞.
- **`tests/test_phase4_report.py`** (review fixes): no power/safety framing, bound described as "not rejected" with beta-binomial p-values, dominance of no-augmentation on the pooled test, the calibration-pairing log, and margin 1.0 = two-candidate pools. `tests/test_calibrate.py` adds `bound_pvalues` and `cluster_calibration`.

## 10. Limitations

- **The error mix is researcher-constructed.** Injected algebra slips plus homophone swaps, synthesised by cloned TTS voices. SCR on this mix says nothing about how often real speakers make sound-alike mistakes.
- **Exchangeability.**
  - The bound is about the pooled calibration mix: 61% homophone, 39% first-pass. It is not rejected on the pooled test set (62.5% homophone; beta-binomial p = 0.20 / 0.28), though both point estimates are above it.
  - It is exceeded on the homophone subset, a sub-population the marginal bound does not cover.
  - Calibration is almost entirely synthetic: 320 of 327 items.
- **v1 and v2 are not independent.** They read the same derivations, so per-voice numbers are two readings of one sample, not n = 320. The same holds in calibration: n = 327 items, but about 167 derivation clusters (v1/v2 margin correlation ρ = 0.69).
- **Augmentation has no measured benefit here.** The non-augmented system weakly dominates the shipped one on the pooled test (fix rate and SCR). The case for shipping augmentation rests on real -teen/-ty mishearings that this test set does not contain.
- **Small recording set.** 20 test derivations, 6 error items, no finite margins: no significance claim.
- **The margin is the rank prior plus a pool-relative z-score.** Variant margins are ≥ 0.5 by construction (ε × 1), and the acoustic part depends on pool size and contents (two-candidate pools always give 1.0). This depends on the frozen ε = 0.5 and γ = 0.25. A variant whose parent was not ĉ pays more.
- **Approximate sweep.** The curves in the figure re-score stored pools whose statuses were checked against the calibrated run's previous outputs. The marked points come from real sequential runs.
- **Latency of the hom rows.** These rows were built with augmentation on, so their GPU timing covers the whole pool. Migrated rows add the variants' scoring time separately. Both slightly overstate a live single-batch system.
- **Carried over from the first pass.**
  - context_vars come from ground-truth line 0.
  - SymEq does not equate ± with "or", which understates the fix rate.
  - Spoken narration in recordings reaches the transcript.
  - 11 imperfect-QC test clips are kept in test.
