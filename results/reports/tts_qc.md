# TTS QC (Phase 3.5)

20 random clips (seed 42) from 4054 synthesised lines (400 derivations, voices ['v1', 'v2']).
Whisper best-1 vs the LLM verbalisation, both normalised (digits and symbols spelled out, filler words dropped). Garbage = WER > 0.5.

**Garbage: 0 / 20** (threshold for fixing the verbalisation prompt: > 4). Mean WER 0.000.

| # | clip | voice | LaTeX | spoken (LLM) | Whisper best-1 | WER | garbage |
|---|---|---|---|---|---|---|---|
| 1 | qform-0021 line 4 | v2 | `x = -\frac{5}{4} \text{ or } x = -3` | x equals negative five over four or x equals negative three | x equals negative 5 over 4 or x equals negative 3. | 0.00 |  |
| 2 | ineq-0045 line 1 | v2 | `6x + 24 - 1 < -3x - 31` | six x plus twenty four minus one is less than negative three x minus thirty one | 6x plus 24 minus 1 is less than negative 3x minus 31. | 0.00 |  |
| 3 | ineq-0010 line 2 | v1 | `4x - 9 \leq -18` | four x minus nine is less than or equal to negative eighteen | 4x minus 9 is less than or equal to negative 18. | 0.00 |  |
| 4 | qform-0063 line 2 | v2 | `4x^2 - 13x + 3 = 0` | four x squared minus thirteen x plus three equals zero | 4x squared minus 13x plus 3 equals 0 | 0.00 |  |
| 5 | lin-0032 line 2 | v1 | `5x - 11 = -33` | five x minus eleven equals negative thirty three | 5 x minus 11 equals negative 33. | 0.00 |  |
| 6 | lin-0020 line 1 | v1 | `4x + 4 - 7 = 2x - 12` | four x plus four minus seven equals two x minus twelve | 4 x plus 4 minus 7 equals 2 x minus 12. | 0.00 |  |
| 7 | lin-0011 line 2 | v1 | `10x + 12 = -7` | ten x plus twelve equals negative seven | 10x plus 12 equals negative 7. | 0.00 |  |
| 8 | ineq-0057 line 1 | v1 | `2x - 8 + 5 < 4x - 13` | two x minus eight plus five is less than four x minus thirteen | 2x minus 8 plus 5 is less than 4x minus 13. | 0.00 |  |
| 9 | qform-0061 line 1 | v2 | `4x^2 + 4x = -25x - 30` | four x squared plus four x equals negative twenty five x minus thirty | 4x squared plus 4x equals negative 25, x minus 30. | 0.00 |  |
| 10 | ineq-0041 line 4 | v2 | `x < -5` | x is less than negative five | x is less than negative 5. | 0.00 |  |
| 11 | qform-0037 line 1 | v1 | `5x^2 + x = -22x - 24` | five x squared plus x equals negative twenty two x minus twenty four | 5x squared plus x equals negative 22x minus 24. | 0.00 |  |
| 12 | qform-0063 line 3 | v1 | `x = \frac{13 + \sqrt{121}}{8} \text{ or } x = \frac{13 - \sqrt{121}}{8}` | x equals the quantity thirteen plus the square root of one hundred twenty one, over eight or x equals the quantity thirteen minus the square root of one hundred twenty one, over eight | x equals the quantity 13 plus the square root of 121 over 8, or x equals the quantity 13 minus the square root of 121 over 8. | 0.00 |  |
| 13 | rat-0041 line 2 | v2 | `x - 3 = -2` | x minus three equals negative two | x minus 3 equals negative 2. | 0.00 |  |
| 14 | qfac-0062 line 0 | v2 | `x \cdot (3x + 6) = 20x - 15` | x times the quantity three x plus six equals twenty x minus fifteen | x times the quantity 3, x plus 6, equals 20, x minus 15. | 0.00 |  |
| 15 | ineq-0035 line 1 | v2 | `5x - 15 + 6 > 8x - 10` | five x minus fifteen plus six is greater than eight x minus ten | 5x minus 15 plus 6 is greater than 8x minus 10. | 0.00 |  |
| 16 | qform-0001 line 3 | v2 | `x = \frac{-10 + \sqrt{196}}{8} \text{ or } x = \frac{-10 - \sqrt{196}}{8}` | x equals the quantity negative ten plus the square root of one hundred ninety six, over eight or x equals the quantity negative ten minus the square root of one hundred ninety six, over eight | x equals the quantity negative 10 plus the square root of 196 over 8, or x equals the quantity negative 10 minus the square root of 196 over 8. | 0.00 |  |
| 17 | qfac-0011 line 0 | v1 | `x \cdot (x - 3) = -5x + 24` | x times the quantity x minus three equals negative five x plus twenty four | x times the quantity x minus 3 equals negative 5 x plus 24. | 0.00 |  |
| 18 | ineq-0013 line 0 | v1 | `2(x + 2) + 2 > 3x + 9` | two times the quantity x plus two plus two is greater than three x plus nine | 2 times the quantity x plus 2 plus 2 is greater than 3x plus 9. | 0.00 |  |
| 19 | ineq-0012 line 2 | v1 | `-2x - 5 \leq 6` | negative two x minus five is less than or equal to six | Negative 2x minus 5 is less than or equal to 6. | 0.00 |  |
| 20 | ineq-0038 line 3 | v1 | `3x < 24` | three x is less than twenty four | 3x is less than 24 | 0.00 |  |

## All 4054 clips

| voice | clips | mean WER | garbage | WER > 0 | first word lost | last word lost |
|---|---|---|---|---|---|---|
| v1 | 2027 | 0.010 | 3 | 119 | 51 | 26 |
| v2 | 2027 | 0.020 | 6 | 83 | 18 | 27 |

Clips with WER > 0.15:

- qform-0026 line 0 v2 (WER 17.73): "x times the quantity two x plus three equals three x plus eight" -> "x times the quantity equals equals equals equals equals equals equals equals equals equals equals equals equals equals equals equals equals equals equals equals equals equals equals equals equals equals equals equals equals equals equals equals equals equals equals equals equals equals equals equals equals equals equals equals equals equals equals equals equals equals equals equals equals equals equals equals equals equals equals equals equals equals equals equals equals equals equals equals equals equals equals equals equals equals equals equals equals equals equals equals equals equals equals equals equals equals equals equals equals equals equals equals equals equals equals equals equals equals equals equals equals equals equals equals equals equals equals equals equals equals equals equals equals equals equals equals equals equals equals equals equals equals equals equals equals equals equals equals equals equals equals equals equals equals equals equals equals equals equals equals equals equals equals equals equals equals equals equals equals equals equals equals equals equals equals equals equals equals equals equals equals equals equals equals equals equals equals equals equals equals equals equals equals equals equals equals equals equals equals equals equals equals equals equals equals equals equals equals equals equals equals equals equals equals equals equals"
- ineq-0079 line 1 v2 (WER 10.38): "four x plus eight minus five is less than or equal to negative four x plus nine" -> "4x plus 8 minus 5 is less than or equal to negative 4x plus 9 is less than or equal to negative 4x plus 9 is less than or equal to negative 4x plus 9 is less than or equal to negative 4x plus 9 is less than or equal to negative 4x plus 9 is less than or equal to negative 4x plus 9 is less than or equal to negative 4x plus 9 is less than or equal to negative 4x plus 9 is less than or equal to negative 4x plus 9 is less than or equal to negative 4x plus 9 is less than or equal to negative 4x plus 9 is less than or equal to negative 4x plus 9 is less than or equal to negative 4x plus 9 is less than or equal to negative 4x plus 9 is less than or equal to negative 4x plus 9 is less than or equal to negative 4x plus 9 is less than or equal to negative 4x plus 9 is less than or equal to negative"
- ineq-0010 line 3 v2 (WER 1.00): "four x is less than or equal to negative nine" -> "Okay."
- lin-0064 line 2 v2 (WER 1.00): "six x minus seventeen equals negative twelve" -> "Oh, my God."
- lin-0014 line 3 v2 (WER 0.80): "negative three x equals eight" -> "Negative."
- qfac-0026 line 0 v1 (WER 0.73): "x times the quantity x minus four equals six x minus twenty one" -> "x times the quantity equals"
- lin-0039 line 2 v1 (WER 0.67): "the quantity x plus one squared equals nine" -> "The quantity squared equals mine."
- rat-0009 line 4 v2 (WER 0.67): "x equals four" -> "x"
- rat-0020 line 0 v1 (WER 0.55): "four over the quantity x minus five equals one over the quantity x plus four" -> "4 over the quantity equals 1 over the quantity."
- lin-0007 line 4 v2 (WER 0.50): "x equals negative two" -> "x equals no"
- rat-0041 line 0 v1 (WER 0.50): "negative six over the quantity x minus three equals three" -> "6 over the quantity equals 3."
- rat-0043 line 3 v1 (WER 0.50): "the quantity x minus six squared equals four" -> "The quantity squared equals 4."
- rat-0061 line 3 v1 (WER 0.50): "the quantity x plus three squared equals sixteen" -> "The quantity squared equals 16."
- lin-0049 line 0 v1 (WER 0.42): "six times the quantity x plus three plus one equals four x plus thirty" -> "6 times the quantity equals 4x plus 30."
- lin-0038 line 3 v2 (WER 0.40): "eight x equals negative six" -> "edX equals negative 6."
- qform-0048 line 0 v1 (WER 0.40): "x times the quantity two x plus one equals x plus eight" -> "x times the quantity equals x plus 8."
- lin-0072 line 0 v1 (WER 0.38): "six times the quantity x minus three equals thirty six" -> "6 times the quantity equals 36."
- rat-0069 line 0 v1 (WER 0.38): "negative ten over the quantity x minus three equals three" -> "negative 10 over the quantity equals 3."
- rat-0070 line 0 v1 (WER 0.38): "the quantity eighteen over the quantity x minus three equals negative two" -> "The quantity 18 over the quantity equals negative 2."
- qfac-0030 line 0 v1 (WER 0.36): "x times the quantity two x plus three equals twelve x minus four" -> "x times the quantity equals 12x minus 4"
- lin-0011 line 4 v1 (WER 0.33): "x equals negative nineteen over ten" -> "x equals negative 19"
- lin-0023 line 4 v1 (WER 0.33): "x equals seven" -> "Wix equals 7."
- qfac-0027 line 3 v2 (WER 0.33): "x equals five" -> "x equals"
- rat-0047 line 0 v1 (WER 0.33): "negative seven over the quantity x plus three equals negative two" -> "negative 7 over the quantity equals negative 2"
- rat-0010 line 0 v1 (WER 0.30): "negative twenty four over the quantity x plus three equals negative four" -> "negative 24 over the quantity equals negative 4."
- ineq-0062 line 0 v1 (WER 0.29): "five times the quantity x minus five plus eight is less than or equal to two x minus twenty four" -> "5 times the quantity is less than or equal to 2x-24."
- lin-0045 line 4 v2 (WER 0.25): "x equals negative one" -> "vex equals negative 1."
- lin-0047 line 4 v1 (WER 0.25): "x equals negative one" -> "X equals negative."
- lin-0049 line 3 v1 (WER 0.25): "two x equals eleven" -> "to x equals 11."
- lin-0071 line 3 v1 (WER 0.25): "negative x equals two" -> "and negative x equals 2."
- qfac-0005 line 3 v1 (WER 0.25): "two x equals four" -> "to x equals 4."
- qfac-0039 line 4 v1 (WER 0.25): "x equals negative three or x equals two" -> "ax equals negative 3 or ax equals 2."
- qfac-0055 line 3 v1 (WER 0.25): "two x equals fourteen" -> "2 Planck. X equals 14."
- qform-0006 line 0 v1 (WER 0.25): "x times the quantity four x plus five equals negative five x plus six" -> "Chems the quantity for x plus 5 equals negative 5x plus 6."
- qform-0071 line 1 v1 (WER 0.25): "three x squared minus six x equals negative eleven x plus two" -> "3x squared minus 6 x equals negative 11"
- rat-0011 line 2 v1 (WER 0.25): "nineteen equals six x" -> "19 equals 6x2."
- rat-0011 line 2 v2 (WER 0.25): "nineteen equals six x" -> "19 equals 6."
- rat-0068 line 2 v2 (WER 0.25): "eleven equals two x" -> "11 equals to X."
- ineq-0051 line 3 v1 (WER 0.20): "negative x is greater than six" -> "Negative x is greater than sixth."
- ineq-0051 line 4 v2 (WER 0.20): "x is less than negative six" -> "x is less than negative."
- lin-0004 line 3 v1 (WER 0.20): "ten x equals negative five" -> "Can x equals negative 5?"
- lin-0009 line 3 v1 (WER 0.20): "two x equals negative twelve" -> "to x equals negative 12."
- lin-0029 line 3 v1 (WER 0.20): "five x equals negative fifteen" -> "5x equals negative"
- lin-0063 line 3 v1 (WER 0.20): "two x equals negative three" -> "to x equals negative 3."
- lin-0063 line 3 v2 (WER 0.20): "two x equals negative three" -> "2x equals negative 3x."
- lin-0070 line 3 v1 (WER 0.20): "five x equals negative five" -> "5. x equals negative."
- qfac-0042 line 3 v2 (WER 0.20): "the quantity two x plus one times the quantity x minus four equals zero" -> "The quantity. 2x plus 1 times the quantity x. I miss 4 equals 0."
- qfac-0069 line 2 v1 (WER 0.20): "x plus two equals eight" -> "0x plus 2 equals 8."
- qfac-0071 line 3 v1 (WER 0.20): "two x equals negative four" -> "to x equals negative 4."
- qfac-0071 line 3 v2 (WER 0.20): "two x equals negative four" -> "to x equals negative 4."
- rat-0006 line 4 v2 (WER 0.20): "eight x equals negative eight" -> "8x necules negative 8."
- rat-0007 line 2 v1 (WER 0.20): "four x plus sixteen equals negative four x minus eight" -> "for x plus 16 equals negative for x minus 8"
- rat-0016 line 3 v1 (WER 0.20): "four x equals forty four" -> "for x equals 44"
- rat-0074 line 3 v1 (WER 0.20): "negative four x equals three" -> "negative four hex equals three"
- rat-0078 line 3 v2 (WER 0.20): "six x equals negative four" -> "6x equals negative."
- ineq-0007 line 3 v1 (WER 0.17): "four x is less than negative eleven" -> "for x is less than negative 11."
- lin-0018 line 3 v1 (WER 0.17): "negative two x equals negative eleven" -> "2x equals negative 11."
- lin-0061 line 4 v2 (WER 0.17): "x equals negative eleven over six" -> "X recalls negative. Eleven over six."
- lin-0063 line 2 v2 (WER 0.17): "two x plus ten equals seven" -> "to x plus 10 equals 7."
- lin-0065 line 3 v1 (WER 0.17): "four x equals negative thirty two" -> "for x equals negative 32"
- rat-0003 line 2 v1 (WER 0.17): "five equals three x plus fifteen" -> "5 equals 3 XS plus 15."
- rat-0035 line 4 v1 (WER 0.17): "four x equals negative twenty nine" -> "for x equals negative 29."
- rat-0076 line 3 v2 (WER 0.17): "negative three x equals twenty eight" -> "Negative 3x equals 20e."

## Notes

- First pass (plain spoken text sent to TTS): 0/20 garbage in the sample, but over all 200 clips 3 were garbage and a leading "x" was swallowed in 20 (v1 4, v2 16), e.g. "x times the quantity ..." -> "times the quantity ...". The verbalisation was faithful; the TTS front end was the problem.
- Fix: scripts/synthesize_derivations.tts_text spells a bare x as "ex" for CosyVoice only (manifest keeps the plain text). On the 12 worst lines this recovered the first word 12/12 (vs 5/12 re-rendered unchanged). All clips were re-synthesised with the fix; the numbers above are after it.
- Remaining dropped last words are random, not systematic: re-rendering the 10 affected lines unchanged fixed 10/10 (adding a final period fixed 9/10). For the full run, transcribe every clip (--scan-all), delete clips with WER > 0 and rerun the resumable synthesis once.

## Repair pass (scripts/tts_redo.py)

Every clip with WER > 0 in the full scan was re-synthesised (CosyVoice samples, so a fresh take usually restores a dropped word) and the take with the lower WER was kept; up to 3 rounds.

| round | re-rendered | now WER 0 |
|---|---|---|
| 1 | 202 | 164 |
| 2 | 38 | 23 |
| 3 | 15 | 7 |

Final: 4,046 / 4,054 clips at WER 0; 8 clips at WER 0.08–0.12, mostly Whisper hearing "four" as "for"; none above 0.15. Per-clip `qc_wer` and `qc_asr` are stored in `data/derivations/audio/manifest.jsonl` (and `qc.jsonl`), so Phase 4 can exclude any clip whose audio does not match its label.
