# Demo script (Phase 5.3): narrated flow and dry-run outcomes

The demo runs the shipped system: Whisper 5-best, then gpt-4.1-mini LaTeX candidates plus confusable-number variants, then frozen System D, then the checker against the previous accepted line, then the decision rule. The decision rule uses **class-conditional δ** (a Mondrian conformal calibration):

- A candidate that is a single confusable-number swap of the best-sounding reading ĉ (for example 15 ↔ 50) is the **homophone class**. It uses δ_hom = 0.5395, calibrated on the 200 homophone calibration errors (k = 10).
- Every other candidate uses the pooled δ = 0.5519 (n = 327, k = 16).

Both values are fixed in `results/reports/prereg_homophone.md`. The code is in `src/pipeline.py` and `demo/app.py`.

How each planted case resolves was **measured, not assumed**. `scripts/run_demo_examples.py` writes `demo/examples/<name>/resolved.json`. The same flow was then repeated through the running app's API, clicking the buttons as scripted below. Δ is the System D score gap to ĉ. A line is corrected silently only if a valid reading has Δ below its class threshold.

**Both outcomes are correct behaviour.** For a planted sound-alike, the system either corrects it automatically (when the acoustics nearly tie) or raises a red flag that names the sound-alike. Neither is a failure. The narration for both is below; use the one that happens.

**The page (top to bottom).** Options row (Accessible view, Read comments aloud, Show history, Show the words I said); the document, one row per line with its comment card beside it (like a Google Docs comment; below about 720 px wide the card sits under its line); then the input: a step heading ("Step 1 of 2 · Say the problem", then "Step 2 of 2 · Say line N", or "Answer first · Choose A or B"), the status box (Ready / Recording / Processing / Done / Needs your answer / a refusal message, all in one place), the **▶ Play next example line** button while an example runs, the microphone, **Check this line**, the closed **Type LaTeX instead** panel (with a live preview), and **Remove last line** / **Start over**. On the right of the input: "How it works" and the **Try an example** panel (dropdown + Start example). After every update the page scrolls the newest line into view and keyboard focus goes to the next action.

Colours: green = verified, amber = corrected (shows "heard / wrote"), red = math error (with the explanation), blue = ask-back, grey = unchecked. Every status also has an icon and a word (✓ Follows, ✎ Corrected, ✗ Check this step, ? Did you mean …?, ? Your choice, ○ Not checked). Flags, corrections and questions appear as a comment card beside their line, with a ▶ Listen button and, on the latest line, the answer buttons. No timings are shown anywhere in the UI (a closed "Developer details" panel shows the latest line's pipeline output without timing fields). "Accessible view" gives larger text, higher contrast, larger buttons and focus rings, and turns on "Read comments aloud".

---

## Before going live

1. Open the demo URL (in `demo_url.txt`; local http://127.0.0.1:7860 by default) and check that the four examples appear in the dropdown. A public link needs a login: `demo/app.py --share --auth user:password` (the app refuses `--share` without `--auth`).
2. Load "Owner's recording: clean quadratic" and step through it: three lines, all green. This warms the LLM connection.
3. Press **Start over**.
4. **Stray recordings:** a mic clip with no speech (silence, noise, a click) is rejected before ASR with "No speech detected … nothing was added", and a line whose output is empty or is leaked prompt text is never accepted. If a misheard line was added anyway, **Remove last line** takes it off (in an example, Play next example line then replays that line); no Start over needed.
5. Spoken explanations need the TTS server (`scripts/tts_server.py`). The app starts it if it is not running. If the server is down, the comment card shows the text with "Spoken version unavailable", and the demo still works.

## Part 1: canned quadratic ("Quadratic: divide by x, then a planted fifteen/fifty", voice v1)

Open **Try an example** (right of the microphone), choose the example and press **Start example**, then press the blue **▶ Play next example line** button (just above the microphone; it has keyboard focus after each line, so Enter also works) once per line. Each clip plays automatically ("▶ Replay recording" plays it again). After a flag, the spoken explanation plays when the clip ends.

| line | said | resolved as (dry-run) | Δ of the alternative |
|---|---|---|---|
| 0 | x³ + 5x² = 50x | problem | n/a |
| 1 | x² + 5x = 50 (divides by x) | **red flag**: "dividing by x may lose the solution x = 0" (spoken) | the only other reading, x² + 5x = 15, is also an error |
| 2 | x² + 5x − **15** = 0 (fifty is correct) | **red flag**, with "you said fifteen; fifty would make this step valid … Did you mean fifty?" (spoken) | x² + 5x − 50 = 0: valid, homophone class, Δ = **1.000** > δ_hom = 0.5395, so no silent correction |
| 2, after clicking **"Yes, I meant fifty"** | | amber: heard −15 / wrote −50, "corrected by you" | n/a |
| 3 | (x + 10)(x − 5) = 0 | verified | n/a |
| 4 | x = −10 or x = 5 | verified | n/a |

**Narration (line 1, divide by x):** "I divide both sides by x. The system flags it straight away and says why: dividing by x may lose the solution x = 0. That's a real mistake, and the system doesn't fix it for me. It tells me."

**Narration (line 2, planted fifteen/fifty, the outcome that happened):** "Now I say fifteen where the step needs fifty. The system doesn't quietly rewrite what I said. It says: *you said fifteen; fifty would make this step valid. Did you mean fifty?* That's the faithfulness point. A sound-alike that happens to make the maths work is exactly the kind of change you don't want made behind your back. The threshold δ is what decides whether the acoustics are close enough to fix silently. Here they aren't, so it asks. I click 'Yes, I meant fifty', and the line turns amber: what it heard and what it wrote, both shown."

**Narration (line 2, if it ever auto-corrects instead; this did not happen in the dry-run):** "I said fifteen, but the acoustics are nearly a tie with fifty, and fifty makes the step valid. So it writes fifty and shows both: heard fifteen, wrote fifty. That's the silent-correction case δ is calibrated to keep rare: at most about 5% of spoken mistakes get 'fixed' like this."

**Important:** click "Yes, I meant fifty" (in the comment card) before going to line 3. If you don't, the line stays as said (−15), and line 3 is checked against it and also flagged red. That is correct behaviour (every line is checked against the line before it), but it's a distracting cascade on stage.

## Part 2: ask-back ("Ask-back: fifty times the quantity … equals fifteen times fifty", voice v1)

| line | said | resolved as (dry-run) | Δ |
|---|---|---|---|
| 0 | 15x² + 75x = 750 | problem | n/a |
| 1 | 50(x² + 5x) = 15 · 50 (a slip: the leading 15 said as fifty) | **ask-back** (blue). Options: A 50(x² + 5x) = 50 × 50, B 15(x² + 5x) = 15 × 50 | A: Δ = 0.534, B: Δ = 0.536, both valid, both homophone class, both < δ_hom = 0.5395 |
| 1, after clicking **B** | | "asked back: you chose B" | n/a |
| 2 | x² + 5x = 50 | verified | n/a |

**You must choose A or B before the next line.** While the ask-back is unanswered, the heading says "Answer first · Choose A or B", Play next example line shows "Choose A or B first", and a new recording or typed line is refused with a message in the status box, because the next line is checked against the choice.

**Narration:** "Here there are two sound-alike readings, and both make the step valid. Fifty times fifty works, and so does fifteen times fifty. Picking one would be a guess, so the system asks: *did you mean fifty instead of fifteen, or fifteen instead of fifty?* I pick B, and the next line checks against my choice."

**Caveat (honest, fragile):** this ask-back fires with a margin of only 0.004 to 0.006 below δ_hom. It is deterministic on this clip (two dry-runs and the example run gave identical results), but a re-recording could push one reading over the threshold. If the line comes out red, with "you said … would make this step valid. Did you mean …?", say so: "with a slightly clearer recording only one reading is close enough to ask about, so it flags instead". The same click resolves it.

I tried two other ask-back constructions, and **neither fired**. Both came out as red flags with a "did you mean" suggestion, because both swaps scored Δ 0.93–1.09:

- 60x² = 16 after 16x² − 16 = 0
- 14x² = 40 after 14x² = 14

Their definitions are kept in `scripts/make_demo_examples.py` (`ATTEMPTS`). With clean TTS, -teen/-ty swaps are usually not acoustically close. That matches Phase 4: only 7.5% of homophone calibration errors fall below δ.

## Part 3: the owner's real recordings (real speech is better theatre)

- **"Owner's recording: quadratic with a sign slip in the factoring" (d031).** x² + 2x − 8 = 0 is followed by (x − 4)(x − 2) = 0, which is a **red flag**: "the solutions change from x = −4 or x = 2 to x = 2 or x = 4" (spoken). Next, x = 4 or x = 2 is **verified**, because it follows from the line as said.
  - **Narration:** "My own voice, my own mistake: I got a sign wrong in the factoring. It catches it and tells me what changed. The last line is verified because it does follow from what I wrote. Every step is checked against the step before it, not against the answer."
- **"Owner's recording: clean quadratic" (d033).** x² + 3x − 10 = 0, then (x − 2)(x + 5) = 0, then x = 2 or x = −5. **All verified (green).**

I did not use d044, the owner's real divide-by-x recording. The speaker says "dividing both sides by x, x squared equals nine", and the LLM writes \frac{x^2}{x} = 9. The flag is then a generic "solutions change", not the divide-by-x explanation, and the last line ("plus or minus three") is read as an unparseable `\pm`. The synthetic quadratic in Part 1 carries the divide-by-x case instead.

## Part 4: live microphone (owner's live mic test)

1. Say the problem as the first line (Step 1 of 2), or open **Type LaTeX instead**, type it (the preview renders it as you type) and press **Add typed line**. Typed lines after the problem are checked like spoken ones (✓ Follows, ✗ Check this step or ○ Not checked; never "corrected").
2. Then, for each line: press Record, say **one** line, and press Stop. The line is checked as soon as you stop (**Check this line** is the manual fallback).
3. Suggested script: problem `x^2 + 5x = 50`, then say "x squared plus five x minus fifteen equals zero". The dry-run with the canned clip as the "mic" gave a red flag plus "did you mean fifty?". Live speech may tie closer and auto-correct, or ask back. Every outcome is correct; narrate whichever happens using the lines above.

Error messages (shown in the status box above the microphone; nothing else on the page changes):

- No recording: "No recording yet: press Record, say one line, then press Stop."
- A clip under 0.25 s is rejected with a message.
- LLM or network failures say which stage failed and suggest trying an example as the fallback.

## Measured latency (seconds per line, RTX 4090)

These are end-to-end per line in the app: resample, ASR, LLM, acoustic scoring, checker, classes and decide.

| example | first run (LLM uncached) | repeat (LLM cached) |
|---|---|---|
| quadratic, lines 0–4 | 2.00, 1.66, 2.57, 2.46, 2.40 | 0.22, 0.35, 0.54, 0.45, 0.46 |
| askback, lines 0–2 | 2.58, 6.08, 1.91 | 1.55, 4.03, 0.45 |
| rec_error, lines 0–2 | n/a (cached in Phase 4) | 0.21, 0.41, 0.34 |
| rec_clean, lines 0–2 | n/a (cached in Phase 4) | 0.23, 0.39, 0.21 |

- **Live mic lines are uncached**, so expect about 1.5–3 s per line, and up to about 6 s for a line with many candidates.
- The ask-back line is the slowest because of the SymEq class grouping of its pool: 2.85 s of the 4.03 s. This is the shipped `decide.classes`, unchanged.
- The spoken explanation is synthesised in the background once the line is flagged. It takes about 3–4 s per sentence and plays after the example clip ends.

## Things that are display-only (not part of the evaluated decision rule)

- The **"did you mean …?"** suggestion on a red flag lists the best valid homophone-class reading, whatever its Δ. It never changes the status or the output. Only the user's click does.
- The **resolution buttons** (ask-back A/B, "Yes, I meant …", "No, keep what I said") record the user's choice, and later lines are checked against it.
