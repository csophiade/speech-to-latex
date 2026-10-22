"""Pure HTML rendering for the demo UI (no Gradio): maths as MathML, the document grid, comment cards.

The document is one CSS grid with two cells per line: the line itself (left) and its comment card
(right, like a Google Docs comment), so every card sits level with the line it refers to. Below
about 720 px the grid becomes one column and each card stacks under its line.

Every function here is pure: it takes line records (the dicts from src.pipeline) and returns HTML.
Interactive resolution buttons inside a card carry data-act="A" / "B"; the page script in
demo/app.py forwards those clicks to hidden Gradio buttons (HTML cannot call Python directly).
"""

import html
import re

from latex2mathml.converter import convert as _to_mathml

# status -> (css key, icon, word label); never colour alone
STATUS = {
    "given": ("given", "◆", "Problem"),
    "verified": ("ok", "✓", "Follows"),
    "corrected": ("fix", "✎", "Corrected"),
    "math_error": ("err", "✗", "Check this step"),
    "ask_back": ("ask", "?", "Your choice"),
    "unchecked": ("unk", "○", "Not checked"),
}


def cap(s):
    s = str(s or "")
    return s[:1].upper() + s[1:]


def esc(s):
    return html.escape(str(s or ""), quote=True)


# ---------- maths ----------

def spoken_form(latex):
    """Plain-words reading of a LaTeX line for aria-labels: '\\text{or}' kept as a word."""
    from src.verbalise import flat_reading
    parts = re.split(r"\\text\{([^{}]*)\}", latex or "")
    out = []
    for k, part in enumerate(parts):
        if k % 2:
            out.append(part.strip())
        elif part.strip():
            try:
                out.append(flat_reading(part))
            except Exception:
                out.append(part.strip())
    return re.sub(r"\s+", " ", " ".join(w for w in out if w)).strip()


def mathml(latex):
    """MathML string for latex, or None if it cannot be converted."""
    try:
        out = _to_mathml((latex or "").strip())
    except Exception:
        return None
    return _prefix_signs(out) if "<math" in out else None


_CLOSERS = ("&#x00029;", "&#x0005D;", "&#x0007D;", "|", "&#x0007C;", "!")


def _prefix_signs(ml):
    """Mark a sign that starts an expression (after '=', '(' or at the start) as a prefix operator,
    so '= −10' is not spaced like a subtraction."""
    def fix(m):
        before = m.group(1)
        if before.startswith("<mo") and any(c in before for c in _CLOSERS):
            return m.group(0)
        return f'{before}<mo form="prefix" lspace="0" rspace="0">{m.group(2)}</mo>'
    return re.sub(r'(<mrow>|<math[^>]*>|<mo[^>]*>[^<]*</mo>)<mo>(&#x02212;|&#x0002B;)</mo>', fix, ml)


def math_html(latex, display=False):
    """Rendered maths (MathML, read by screen readers) with the spoken form as its label;
    raw LaTeX in <code> if conversion fails; a placeholder for an empty line."""
    latex = (latex or "").strip()
    if not latex:
        return '<span class="m empty">(nothing recognised)</span>'
    ml = mathml(latex)
    label = esc(spoken_form(latex) or latex)
    cls = "m disp" if display else "m"
    if ml is None:
        return f'<span class="{cls} raw" aria-label="{label}"><code>{esc(latex)}</code></span>'
    if display:   # display-size maths, but left-aligned with its label (display="block" centres it)
        ml = ml.replace('display="inline"', 'display="inline" displaystyle="true"', 1)
    return f'<span class="{cls}" aria-label="{label}" title="{esc(latex)}">{ml}</span>'


def pretty(text):
    """Checker text for display: sqrt(85) -> √85, sqrt(a + b) -> √(a + b)."""
    text = re.sub(r"sqrt\((\w+)\)", "√\\1", text or "")
    return text.replace("sqrt(", "√(")


# ---------- one line ----------

def badge(r):
    key, icon, word = STATUS.get(r["status"], ("unk", "○", r["status"]))
    w = _words(r)
    if r["status"] == "math_error" and r.get("suggestion") and w and not r.get("resolved_from"):
        icon, word = "?", f"Did you mean {w[1]}?"
    elif r["status"] == "ask_back" and r.get("resolved_from"):
        word = cap(r["resolved_from"])
    elif r["status"] == "corrected" and r.get("resolved_from"):
        word = "Corrected by you"
    elif r["status"] == "math_error" and r.get("resolved_from"):
        word = "Kept as you said it"
    return (f'<span class="badge s-{key}"><span class="ic" aria-hidden="true">{icon}</span>'
            f'<span class="word">{esc(word)}</span></span>')


def line_label(i):
    return "Problem" if i == 0 else f"Line {i}"


def line_cell(i, r, show_heard=False):
    key = STATUS.get(r["status"], ("unk",))[0]
    heard = ""
    if show_heard:
        if r.get("typed"):
            heard = '<div class="heard">Typed</div>'
        elif r.get("transcript"):
            heard = f'<div class="heard">Heard: <q>{esc(r["transcript"].strip())}</q></div>'
    math = math_html(r["output"], display=True)
    if r["status"] == "corrected":
        math = (math_html(r["output"], display=True)
                + f'<div class="was">you said {math_html(r["heard"])}</div>')
    return (f'<div class="cell line s-{key}" id="line-{i}" data-line="{i}" tabindex="-1">'
            f'<div class="lhead"><span class="num">{line_label(i)}</span>'
            f'{"" if r["status"] == "given" else badge(r)}</div>'
            f'{math}{heard}</div>')


# ---------- comment cards ----------

def _words(r):
    sug = r.get("suggestion") or {}
    return sug.get("words")


def card_content(r):
    """(title, body html, actions [(act, label html)]) for a line's comment, or None for no card."""
    st, pending = r["status"], not r.get("resolved_from")
    if st == "math_error":
        w = _words(r)
        if r.get("suggestion") and w and pending:
            said, meant = w
            body = (f'<p>You said <b>{esc(said)}</b>. With <b>{esc(meant)}</b> this step would be valid:</p>'
                    f'<p class="opt">{math_html(r["suggestion"]["latex"])}</p>')
            if r.get("explanation"):
                body += f'<p class="why">As said: {esc(pretty(r["explanation"]))}.</p>'
            return (f"Did you mean {meant}?", body,
                    [("A", f"Yes, I meant {esc(meant)}"), ("B", "No, keep what I said")])
        body = f'<p>{esc(pretty(r.get("explanation") or "This step does not follow from the line before."))}</p>'
        if r.get("resolved_from"):
            body += '<p class="why">You kept it as said, so the next line is checked against it.</p>'
        return "Check this step", body, []
    if st == "ask_back":
        a, b = (r.get("options") or ["", ""])[:2]
        if pending:
            body = ('<p>Two readings of what you said both make this step valid.</p>'
                    f'<ol class="choices"><li><b>A</b> {math_html(a)}</li><li><b>B</b> {math_html(b)}</li></ol>')
            return "Which did you mean?", body, [("A", "A"), ("B", "B")]
        return "Which did you mean?", f'<p>{esc(cap(r["resolved_from"]))}: {math_html(r["output"])}</p>', []
    if st == "corrected":
        w = _words(r) if r.get("resolved_from") else None
        title = "Corrected"
        if w:
            title = f"Corrected: heard {w[0]} / wrote {w[1]}"
        else:
            m = re.match(r"I heard (.+?) and wrote (.+?),", r.get("spoken_explanation") or "")
            if m:
                title = f"Corrected: heard {m.group(1)} / wrote {m.group(2)}"
        body = (f'<p>Heard {math_html(r["heard"])}</p><p>Wrote {math_html(r["output"])}</p>'
                '<p class="why">' + ("You confirmed this correction." if r.get("resolved_from") else
                                     "This reading sounds almost the same and makes the step valid.")
                + '</p>')
        return title, body, []
    if st == "unchecked":
        why = r.get("explanation") or ""
        title = "Couldn't read this line" if "read" in why or not r["output"].strip() else "Couldn't check this line"
        return title, f'<p>{esc(pretty(why) or "The checker could not compare this line with the one before.")}</p>', []
    if r.get("note"):
        return "Note", f'<p>{esc(pretty(r["note"]))}</p>', []
    return None


def card_html(i, r, audio_url=None, actions=True, tts_failed=False):
    """The comment card for line i, or '' if the line needs none.
    audio_url: URL of the spoken explanation (None: no audio). actions: show resolution buttons."""
    c = card_content(r)
    if c is None:
        return ""
    title, body, acts = c
    key = STATUS.get(r["status"], ("unk",))[0]
    tools = ""
    if audio_url:
        tools = (f'<button type="button" class="play" data-play="{esc(audio_url)}" '
                 f'aria-label="Listen to this comment">▶ Listen</button>')
    elif tts_failed and r.get("spoken_explanation"):
        tools = '<span class="why">Spoken version unavailable.</span>'
    btns = ""
    if actions and acts:
        btns = '<div class="acts">' + "".join(
            f'<button type="button" class="act{" primary" if k == "A" else ""}" data-act="{k}" '
            f'data-line="{i}">{lab}</button>' for k, lab in acts) + "</div>"
    heading = f'{line_label(i)}: {esc(title)}'
    return (f'<aside class="card s-{key}" aria-label="{esc(heading)}">'
            f'<div class="ctitle"><span class="ic" aria-hidden="true">{"?" if pending([r]) else STATUS.get(r["status"], ("", "○"))[1]}</span>'
            f'{esc(title)}</div><div class="cbody">{body}</div>'
            f'<div class="ctools">{tools}{btns}</div></aside>')


# ---------- the document ----------

def doc_html(history, show_history=True, show_heard=False, audio_url=lambda r: None, tts_failed=lambda r: False):
    """The whole document grid. With show_history off only the latest line (and its card) is shown."""
    if not history:
        return ('<div class="doc empty-doc"><div class="cell line placeholder">'
                '<span class="num">Problem</span><p>Your lines will appear here, each checked against '
                'the one before it.</p></div><div class="cell rail"></div></div>')
    n = len(history)
    first = 0 if show_history else n - 1
    rows = []
    if first > 0:
        rows.append(f'<div class="cell line folded">{first} earlier line{"s" if first > 1 else ""} hidden '
                    '(turn on "Show history" to see them)</div><div class="cell rail"></div>')
    for i in range(first, n):
        r = history[i]
        latest = i == n - 1
        card = card_html(i, r, audio_url(r), actions=latest, tts_failed=tts_failed(r))
        rows.append(line_cell(i, r, show_heard)
                    + f'<div class="cell rail{" has-card" if card else ""}">{card}</div>')
    return (f'<div class="doc" role="list" aria-label="Your working, {n} line{"s" if n > 1 else ""}">'
            + "".join(rows) + "</div>")


# ---------- step and status ----------

def pending(history):
    """'ask_back' if the latest line is an unanswered ask-back, 'suggestion' for an open "did you
    mean", else None."""
    r = history[-1] if history else None
    if not r or r.get("resolved_from"):
        return None
    if r["status"] == "ask_back":
        return "ask_back"
    if r["status"] == "math_error" and r.get("suggestion"):
        return "suggestion"
    return None


def step_title(history, finished=False):
    """The heading above the input: what to do next. Step 1 is the problem; step 2 repeats, one
    working line at a time (numbered like the lines in the document)."""
    if not history:
        return "Step 1 of 2 · Say the problem"
    if finished:
        return "Example finished · Start over, or say another line"
    if pending(history) == "ask_back":
        return f"Answer first · Choose A or B for line {len(history) - 1}"
    return f"Step 2 of 2 · Say line {len(history)}"


def status_html(history, phase=None):
    """Status pill: ready / done / needs your answer (recording and processing are set in the page)."""
    p = pending(history)
    if p == "ask_back":
        kind, text = "need", "Needs your answer: choose A or B in the comment beside the line, then say the next line."
    elif p == "suggestion":
        kind, text = "need", ("Needs your answer: did you mean the sound-alike? Answer in the comment beside "
                              "the line, or just say the next line.")
    elif not history:
        kind, text = "ready", "Ready. Press Record, say the problem, then press Stop."
    else:
        r = history[-1]
        word = STATUS.get(r["status"], ("", "", r["status"]))[2].lower()
        n = line_label(len(history) - 1)
        if r["status"] == "given":
            kind, text = "done", "Done: problem set. Press Record and say line 1."
        elif r["status"] == "ask_back":   # answered
            kind, text = "done", f"Done: {n}, {r['resolved_from']}. Press Record and say the next line."
        elif r["status"] in ("verified", "corrected"):
            kind, text = "done", f"Done: {n} {word}. Press Record and say the next line."
        else:
            kind, text = "flag", (f"{n} added, but needs a look: {word}. Read the comment beside it, then "
                                  "press Record for the next line (or Remove last line).")
    if phase:
        kind, text = phase
    icons = {"ready": "●", "done": "✓", "need": "?", "flag": "!", "recording": "●", "processing": "…",
             "error": "✗"}
    return (f'<div class="pill p-{kind}" role="status" aria-live="polite">'
            f'<span class="ic" aria-hidden="true">{icons.get(kind, "●")}</span>{esc(text)}</div>')


def preview_html(latex):
    """Live preview of typed LaTeX."""
    latex = (latex or "").strip()
    if not latex:
        return '<div class="preview muted">Preview appears here as you type.</div>'
    ml = mathml(latex)
    if ml is None:
        return f'<div class="preview bad">Can\'t render this yet: <code>{esc(latex)}</code></div>'
    return f'<div class="preview">{math_html(latex, display=True)}</div>'
