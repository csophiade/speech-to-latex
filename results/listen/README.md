# Synthesised clips to listen to (Phase 3.5)

CosyVoice2-0.5B, zero-shot voices v1 (male) and v2 (female) cloned from LibriSpeech dev-clean clips. Synthetic audio, no licence restriction. Text = the LLM verbalisation of the LaTeX line (what the TTS was asked to say).

| file | voice | derivation, line | LaTeX | text spoken | note | Whisper hears |
|---|---|---|---|---|---|---|
| 1_lin-0073_line2_v1.wav | v1 | lin-0073, line 2 | `(x + 4)^2 = 4` | the quantity x plus four squared equals four | DELIBERATE ERROR (square_both_sides) | The quantity x plus 4 squared equals 4. |
| 2_qfac-0003_line2_v2.wav | v2 | qfac-0003, line 2 | `x - 3 = -2` | x minus three equals negative two | DELIBERATE ERROR (divide_by_variable) | x minus 3 equals negative 2. |
| 3_ineq-0016_line4_v2.wav | v2 | ineq-0016, line 4 | `x < 1` | x is less than one | DELIBERATE ERROR (missed_flip) | x is less than 1. |
| 4_qform-0064_line3_v1.wav | v1 | qform-0064, line 3 | `x = \frac{11 + \sqrt{49}}{4} \text{ or } x = \frac{11 - \sqrt{49}}{4}` | x equals the quantity eleven plus the square root of forty nine, over four or x equals the quantity eleven minus the square root of forty nine, over four | valid | x equals the quantity 11 plus the square root of 49 over 4 or x equals the quantity 11 minus the square root of 49 over 4. |
| 5_rat-0057_line0_v2.wav | v2 | rat-0057, line 0 | `\frac{2}{x + 1} = \frac{1}{x - 2}` | two over the quantity x plus one equals one over the quantity x minus two | valid | 2 over the quantity x plus 1 equals 1 over the quantity x minus 2. |
