# TTS QC (Phase 3.5)

20 random clips (seed 42) from 200 synthesised lines (20 derivations, voices ['v1', 'v2']).
Whisper best-1 vs the LLM verbalisation, both normalised (digits and symbols spelled out, filler words dropped). Garbage = WER > 0.5.

**Garbage: 0 / 20** (threshold for fixing the verbalisation prompt: > 4). Mean WER 0.021.

| # | clip | voice | LaTeX | spoken (LLM) | Whisper best-1 | WER | garbage |
|---|---|---|---|---|---|---|---|
| 1 | rat-0011 line 3 | v1 | `6x = 19` | six x equals nineteen | 6x equals 19. | 0.00 |  |
| 2 | ineq-0051 line 3 | v2 | `-x > 6` | negative x is greater than six | Negative x is greater than 6. | 0.00 |  |
| 3 | ineq-0016 line 1 | v2 | `6x - 6 + 8 < 8x` | six x minus six plus eight is less than eight x | 6x minus 6 plus 8 is less than 8. | 0.10 |  |
| 4 | rat-0045 line 4 | v2 | `x = -3` | x equals negative three | x equals negative 3 | 0.00 |  |
| 5 | lin-0073 line 4 | v2 | `(x + 6)(x + 2) = 0` | the quantity x plus six times the quantity x plus two equals zero | The quantity x plus 6 times the quantity x plus 2 equals 0. | 0.00 |  |
| 6 | lin-0073 line 2 (deliberate error) | v1 | `(x + 4)^2 = 4` | the quantity x plus four squared equals four | The quantity x plus 4 squared equals 4. | 0.00 |  |
| 7 | lin-0070 line 2 | v2 | `5x - 8 = -13` | five x minus eight equals negative thirteen | 5x minus 8 equals negative 13. | 0.00 |  |
| 8 | ineq-0069 line 0 | v2 | `5(x + 6) + 3 \geq -3x + 37` | five times the quantity x plus six plus three is greater than or equal to negative three x plus thirty seven | 5 times the quantity x plus 6 plus 3 is greater than or equal to negative 3 x plus 37. | 0.00 |  |
| 9 | rat-0045 line 3 | v2 | `-4x = 12` | negative four x equals twelve | Negative 4x equals 12. | 0.00 |  |
| 10 | ineq-0051 line 1 | v2 | `4x + 8 + 4 > 5x + 18` | four x plus eight plus four is greater than five x plus eighteen | 4x plus 8 plus 4 is greater than 5. x plus 18. | 0.00 |  |
| 11 | rat-0022 line 3 (deliberate error) | v1 | `-2x = -38` | negative two x equals negative thirty eight | negative 2, x equals negative 38. | 0.00 |  |
| 12 | qform-0049 line 4 | v2 | `x = -1 \text{ or } x = -2` | x equals negative one or x equals negative two | x equals negative 1 or x equals negative 2. | 0.00 |  |
| 13 | ineq-0051 line 2 | v1 | `-x + 12 > 18` | negative x plus twelve is greater than eighteen | negative x plus 12 is greater than 18. | 0.00 |  |
| 14 | qform-0066 line 1 (deliberate error) | v1 | `2x^2 - x = -2x` | two x squared minus x equals negative two x | 2x squared minus x equals negative 2x squared | 0.11 |  |
| 15 | qfac-0053 line 3 | v2 | `(3x - 2)(x - 7) = 0` | the quantity three x minus two times the quantity x minus seven equals zero | The quantity 3x minus 2 times the quantity x minus 7 equals 0. | 0.00 |  |
| 16 | ineq-0016 line 3 | v2 | `-2x < -2` | negative two x is less than negative two | Negative 2x is less than negative 2. | 0.00 |  |
| 17 | ineq-0016 line 2 | v2 | `-2x + 2 < 0` | negative two x plus two is less than zero | Negative 2x plus 2 is less than 0. | 0.00 |  |
| 18 | ineq-0051 line 3 | v1 | `-x > 6` | negative x is greater than six | Negative x is greater than sixth. | 0.20 |  |
| 19 | lin-0070 line 0 | v2 | `2(x - 3) - 2 = -3x - 13` | two times the quantity x minus three minus two equals negative three x minus thirteen | 2 times the quantity x minus 3 minus 2 equals negative 3x minus 13. | 0.00 |  |
| 20 | lin-0070 line 4 | v2 | `x = -1` | x equals negative one | x equals negative 1. | 0.00 |  |

## All 200 clips

| voice | clips | mean WER | garbage | WER > 0 | first word lost | last word lost |
|---|---|---|---|---|---|---|
| v1 | 100 | 0.015 | 0 | 9 | 1 | 7 |
| v2 | 100 | 0.007 | 0 | 5 | 0 | 3 |

Clips with WER > 0.15:

- lin-0023 line 4 v1 (WER 0.33): "x equals seven" -> "Wix equals 7."
- rat-0011 line 2 v1 (WER 0.25): "nineteen equals six x" -> "19 equals 6x2."
- rat-0011 line 2 v2 (WER 0.25): "nineteen equals six x" -> "19 equals 6."
- ineq-0051 line 3 v1 (WER 0.20): "negative x is greater than six" -> "Negative x is greater than sixth."
- ineq-0051 line 4 v2 (WER 0.20): "x is less than negative six" -> "x is less than negative."
- lin-0070 line 3 v1 (WER 0.20): "five x equals negative five" -> "5. x equals negative."

## Notes

- First pass (plain spoken text sent to TTS): 0/20 garbage in the sample, but over all 200 clips 3 were garbage and a leading "x" was swallowed in 20 (v1 4, v2 16), e.g. "x times the quantity ..." -> "times the quantity ...". The verbalisation was faithful; the TTS front end was the problem.
- Fix: scripts/synthesize_derivations.tts_text spells a bare x as "ex" for CosyVoice only (manifest keeps the plain text). On the 12 worst lines this recovered the first word 12/12 (vs 5/12 re-rendered unchanged). All clips were re-synthesised with the fix; the numbers above are after it.
- Remaining dropped last words are random, not systematic: re-rendering the 10 affected lines unchanged fixed 10/10 (adding a final period fixed 9/10). For the full run, transcribe every clip (--scan-all), delete clips with WER > 0 and rerun the resumable synthesis once.
