"""Pure HTML/MathML rendering of the demo UI (demo/ui_render.py)."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "demo"))
import ui_render as ui  # noqa: E402


def line(status, **kw):
    return dict({"status": status, "output": "x^2 + 5x = 50", "heard": "x^2 + 5x = 50", "options": [],
                 "explanation": None, "note": None, "suggestion": None, "transcript": "x squared plus 5x equals 50"},
                **kw)


def test_math_html_is_mathml_with_spoken_label():
    h = ui.math_html("x^2 + 5x = 50")
    assert "<math" in h and "<msup>" in h
    assert 'aria-label="x squared plus 5 x equals 50"' in h


def test_math_html_text_words_and_fallback():
    assert "aria-label=\"x equals minus 10 or x equals 5\"" in ui.math_html(r"x = -10 \text{ or } x = 5")
    bad = ui.math_html(r"\left( x")
    assert "<code>" in bad and "<math" not in bad
    assert "nothing recognised" in ui.math_html("")
    assert "&lt;script&gt;" in ui.math_html("<script>")
    unary = ui.mathml("x = -10")
    assert unary.count('form="prefix"') == 1
    assert 'form="prefix"' not in ui.mathml("x - 10 = (x + 1) - 3")
    assert 'displaystyle="true"' in ui.math_html("x = 1", display=True)


def test_every_status_has_icon_and_word():
    for st, (_, icon, word) in ui.STATUS.items():
        b = ui.badge(line(st))
        assert icon in b and word in b
    sug = {"latex": "x = 50", "delta": 1.0, "words": ("fifteen", "fifty")}
    assert "Did you mean fifty?" in ui.badge(line("math_error", suggestion=sug))
    assert ui.doc_html([line("given")]).count("Problem") == 1


def test_cards_titles_and_actions():
    assert ui.card_html(1, line("verified")) == ""
    assert ui.card_html(0, line("given")) == ""
    err = ui.card_html(1, line("math_error", explanation="dividing by x may lose the solution x = 0"))
    assert "Check this step" in err and "dividing by x" in err and "data-act" not in err
    sug = {"latex": "x^2 + 5x - 50 = 0", "delta": 1.0, "words": ("fifteen", "fifty")}
    flag = ui.card_html(2, line("math_error", suggestion=sug), audio_url="gradio_api/file=/t/a.wav")
    assert "Did you mean fifty?" in flag and "Yes, I meant fifty" in flag and "No, keep what I said" in flag
    assert 'data-act="A"' in flag and 'data-act="B"' in flag and 'data-play="gradio_api/file=/t/a.wav"' in flag
    assert "data-act" not in ui.card_html(2, line("math_error", suggestion=sug), actions=False)
    ask = ui.card_html(1, line("ask_back", options=["a = 1", "a = 2"]))
    assert "Which did you mean?" in ask and ask.count("<math") == 2
    done = ui.card_html(1, line("ask_back", options=["a = 1", "a = 2"], resolved_from="you chose B"))
    assert "You chose B" in done and "data-act" not in done
    fixed = ui.card_html(1, line("corrected", heard="x - 15 = 0", output="x - 50 = 0",
                                 spoken_explanation="I heard fifteen and wrote fifty, which makes the step valid."))
    assert "Corrected: heard fifteen / wrote fifty" in fixed
    unk = ui.card_html(1, line("unchecked", explanation="this line could not be read"))
    assert "Couldn&#x27;t read this line" in unk
    assert "Spoken version unavailable" in ui.card_html(
        1, line("math_error", explanation="e", spoken_explanation="E."), tts_failed=True)


def test_doc_grid_aligns_cards_and_history_toggle():
    hist = [line("given"), line("verified"), line("math_error", explanation="wrong")]
    full = ui.doc_html(hist)
    assert full.count('class="cell line') == 3 and full.count('class="cell rail') == 3
    assert full.index('id="line-2"') < full.index("Check this step")
    latest = ui.doc_html(hist, show_history=False)
    assert 'id="line-0"' not in latest and 'id="line-2"' in latest and "2 earlier lines hidden" in latest
    assert "Heard:" in ui.doc_html(hist, show_heard=True) and "Heard:" not in full
    assert "appear here" in ui.doc_html([])


def test_only_latest_card_has_actions():
    sug = {"latex": "x = 50", "delta": 1.0, "words": ("fifteen", "fifty")}
    hist = [line("given"), line("math_error", suggestion=sug), line("verified")]
    assert "data-act" not in ui.doc_html(hist)


def test_status_and_steps():
    assert ui.step_title([]) == "Step 1 of 2 · Say the problem"
    assert ui.step_title([line("given")]) == "Step 2 of 2 · Say line 1"
    assert ui.step_title([line("given"), line("verified")], finished=True).startswith("Example finished")
    assert "Choose A or B" in ui.step_title([line("given"), line("ask_back", options=["a", "b"])])
    assert "p-flag" in ui.status_html([line("given"), line("math_error", explanation="e")])
    assert "p-error" in ui.status_html([], ("error", "No recording yet"))
    assert "p-ready" in ui.status_html([])
    assert "p-done" in ui.status_html([line("given"), line("verified")])
    assert "choose A or B" in ui.status_html([line("given"), line("ask_back", options=["a", "b"])])
    assert ui.pending([line("given"), line("ask_back", resolved_from="you chose A")]) is None


def test_preview():
    assert "<math" in ui.preview_html(r"\frac{1}{2}")
    assert "Can't render" in ui.preview_html(r"\left( x")
    assert "Preview appears" in ui.preview_html("  ")
