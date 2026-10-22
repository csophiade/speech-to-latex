"""Gradio demo (Phase 5): dictate a derivation line by line; every line is checked against the last one.

Run:  .venv/bin/python -W ignore demo/app.py [--port 7860] [--share --auth user:password]
(local-only on 127.0.0.1 by default; a public share link requires a login.)
Starts the TTS server (scripts/tts_server.py) for spoken explanations if it is not running; without
it, explanations are shown as text only. The decision logic is src/pipeline.py (shipped system:
augmented pools, frozen System D, checker, class-conditional δ).
"""

import argparse
import html
import json
import os
import re
import sys
import tempfile
import time
import uuid
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import gradio as gr

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src import tts_live  # noqa: E402
from src.audio import load_audio, speech_seconds, to_16k_mono  # noqa: E402

EXAMPLES = Path(__file__).resolve().parent / "examples"
CACHE = Path(os.environ.get("DEMO_CACHE", Path(tempfile.gettempdir()) / "speech-to-latex-demo"))
ORDER = ["quadratic", "askback", "rec_error", "rec_clean"]   # the narrated order (demo/script.md)
LATEX = [{"left": "$$", "right": "$$", "display": True}, {"left": "$", "right": "$", "display": False}]
COLOURS = {"given": "#6b7280", "verified": "#15803d", "corrected": "#b45309", "math_error": "#b91c1c",
           "unchecked": "#9ca3af", "ask_back": "#1d4ed8"}
LABELS = {"given": "problem", "verified": "verified", "corrected": "corrected", "math_error": "math error",
          "unchecked": "unchecked", "ask_back": "which did you mean?"}
CSS = """
.line {border-left: 6px solid var(--c); background: color-mix(in srgb, var(--c) 9%, transparent);
       padding: 8px 12px; margin: 6px 0; border-radius: 6px; position: relative;}
.line .badge {color: var(--c); font-weight: 600; font-size: 0.8em; text-transform: uppercase; letter-spacing: .04em;}
.line .lat {position: absolute; right: 10px; top: 6px; font-size: 0.75em; opacity: 0.7;}
.line .why {font-size: 0.92em; margin-top: 4px;}
.line .note {font-size: 0.8em; opacity: 0.75; margin-top: 2px;}
.line .math {font-size: 1.25em;}
#corner {text-align: right; font-size: 0.85em; opacity: 0.8;}
"""

PIPE = None
TTS_POOL = ThreadPoolExecutor(1)
TTS_JOBS = {}
CANCELLED = set()      # TTS keys whose line was reset, undone, resolved or superseded
MIN_SPEECH_S = 0.25    # mic clips with less speech than this are rejected before ASR
LEAKS = ("Variables already in use",)   # LLM prompt text that must never be accepted as a line


def pipe():
    global PIPE
    if PIPE is None:
        from src.pipeline import LinePipeline
        PIPE = LinePipeline(cpu_worker=True)
    return PIPE


def examples():
    out = {}
    order = {n: i for i, n in enumerate(ORDER)}
    dirs = sorted(EXAMPLES.iterdir(), key=lambda d: (order.get(d.name, len(order)), d.name)) if EXAMPLES.exists() else []
    for d in dirs:
        if (d / "example.json").exists():
            out[json.loads((d / "example.json").read_text())["title"]] = d
    return out


# ---------- rendering ----------

def m(latex):
    if not (latex or "").strip():
        return "<i>(nothing recognised)</i>"
    return f"${html.escape(latex, quote=False)}$"


def pretty(text):
    """Checker text for display: sqrt(85) -> √85, sqrt(a + b) -> √(a + b)."""
    text = re.sub(r"sqrt\((\w+)\)", "√\\1", text or "")
    return text.replace("sqrt(", "√(")


def plain(latex):
    """LaTeX as readable button text (buttons cannot render maths)."""
    s = latex or ""
    s = re.sub(r"\\frac\{([^{}]*)\}\{([^{}]*)\}", r"(\1)/(\2)", s)
    s = re.sub(r"\\sqrt\{([^{}]*)\}", r"√(\1)", s)
    for a, b in (("\\times", "×"), ("\\cdot", "·"), ("\\pm", "±"), ("\\left", ""), ("\\right", ""),
                 ("^2", "²"), ("^3", "³"), ("\\,", " "), ("\\", "")):
        s = s.replace(a, b)
    s = re.sub(r"\^\{([^{}]*)\}", r"^(\1)", s).replace("{", "").replace("}", "")
    return re.sub(r"\s+", " ", s).strip()


def render_line(i, r):
    st = r["status"]
    body = [f'<div class="math">{m(r["output"])}</div>']
    if st == "corrected":
        body = [f'<div class="math">heard: {m(r["heard"])} &nbsp;/&nbsp; wrote: {m(r["output"])}</div>']
    if r.get("resolved_from") and st != "ask_back":
        body.append(f'<div class="note">{r["resolved_from"]}</div>')
    if st == "ask_back" and not r.get("resolved_from"):
        a, b = r["options"][:2]
        body.append(f'<div class="why">heard {m(r["heard"])}, which does not follow. '
                    f'Two readings do: <b>A</b> {m(a)} &nbsp; <b>B</b> {m(b)}</div>')
    if st in ("math_error", "unchecked") and r.get("explanation"):
        body.append(f'<div class="why">{html.escape(pretty(r["explanation"]))}</div>')
    if st == "math_error" and r.get("suggestion") and r["suggestion"].get("words"):
        said, meant = r["suggestion"]["words"]
        body.append(f'<div class="why">you said <i>{said}</i>; <i>{meant}</i> would make this step valid: '
                    f'{m(r["suggestion"]["latex"])}. Did you mean {meant}?</div>')
    if r.get("note"):
        body.append(f'<div class="note">note: {html.escape(r["note"])}</div>')
    lat = "typed" if r.get("typed") else f'{r["latency_s"]:.1f} s'
    return (f'<div class="line" style="--c:{COLOURS.get(st, "#6b7280")}">'
            f'<span class="badge">{i}. {label(r)}</span><span class="lat">{lat}</span>'
            + "".join(body) + "</div>")


def label(r):
    if r["status"] == "ask_back" and r.get("resolved_from"):
        return f'asked back: {r["resolved_from"]}'
    return LABELS.get(r["status"], r["status"])


def render(state):
    if not state["history"]:
        return "*Type or say the problem (line 0) to start.*"
    return "\n".join(render_line(i, r) for i, r in enumerate(state["history"]))


def corner(state):
    spoken = [r for r in state["history"] if not r.get("typed")]
    if not spoken:
        return ""
    r = spoken[-1]
    p = r["latency_parts"]
    parts = ", ".join(f"{k[:-2]} {v:.2f}" for k, v in p.items() if v >= 0.01)
    mean = sum(x["latency_s"] for x in spoken) / len(spoken)
    return f"last line **{r['latency_s']:.2f} s** ({parts}) · mean {mean:.2f} s over {len(spoken)}"


def pending_choice(state):
    """True while the latest line is an unanswered ask-back: no new line until A or B is chosen."""
    r = state["history"][-1] if state["history"] else None
    return bool(r and r["status"] == "ask_back" and not r.get("resolved_from"))


def resolvable(r):
    return bool(r and not r.get("resolved_from")
                and (r["status"] == "ask_back" or (r["status"] == "math_error" and r.get("suggestion"))))


def buttons(state):
    """Resolution buttons for the latest line (ask_back options, or a red flag's sound-alike)."""
    r = state["history"][-1] if state["history"] else None
    hide = gr.update(visible=False)
    if r is None or r.get("resolved_from"):
        return hide, hide
    if r["status"] == "ask_back":
        a, b = r["options"][:2]
        return gr.update(visible=True, value=f"A: {plain(a)}"), gr.update(visible=True, value=f"B: {plain(b)}")
    if r["status"] == "math_error" and r.get("suggestion"):
        meant = (r["suggestion"].get("words") or ("", "the sound-alike"))[1]
        return (gr.update(visible=True, value=f"Yes, I meant {meant}: {plain(r['suggestion']['latex'])}"),
                gr.update(visible=True, value="No, keep what I said"))
    return hide, hide


def example_status(state):
    ex = state.get("example")
    if not ex:
        return gr.update(interactive=False, value="Next example line")
    left = len(ex["lines"]) - state["pos"]
    if left and pending_choice(state):
        return gr.update(interactive=False, value="Choose A or B first")
    return gr.update(interactive=left > 0, value=f"Next example line ({left} left)" if left else "Example finished")


def views(state):
    a, b = buttons(state)
    return (render(state), corner(state), a, b, example_status(state),
            gr.update(interactive=bool(state["history"])))


def fresh():
    return {"history": [], "example": None, "pos": 0, "tts": None, "clip_end": 0.0}


def cancel_tts(state):
    """Stop a pending spoken explanation from playing (its line was reset, undone, resolved or superseded)."""
    if state and state.get("tts"):
        CANCELLED.add(state["tts"])
        job = TTS_JOBS.pop(state["tts"], None)
        if job:
            job.cancel()


# ---------- actions ----------

def _tts_job(state, r):
    """Pre-generate the spoken explanation of a flagged / corrected line in the background."""
    text = r.get("spoken_explanation")
    if not text or r["status"] not in ("math_error", "ask_back", "corrected"):
        return None
    key = uuid.uuid4().hex
    CACHE.mkdir(parents=True, exist_ok=True)
    TTS_JOBS[key] = TTS_POOL.submit(tts_live.synth, text, CACHE / f"{key}.wav")
    return key


def _process(state, audio, sr):
    if pending_choice(state):
        raise gr.Error("Choose A or B for the blue line first; a new line is checked against your choice.")
    try:
        r = pipe().process_line(audio, sr, state["history"])
    except ValueError as e:
        raise gr.Error(f"Could not use that recording: {e}")
    except Exception as e:   # model / network failures: say which stage, keep the session alive
        raise gr.Error(f"Processing failed ({type(e).__name__}: {e}). Check the LLM API key / network and "
                       f"try again, or use Load example.")
    if not r["output"].strip() or any(x in r["output"] for x in LEAKS):
        raise gr.Error("Nothing usable was recognised, so the line was not added. Say the line again.")
    cancel_tts(state)
    state["history"].append(r)
    state["tts"] = _tts_job(state, r)
    return state


def add_line(state, mic):
    if mic is None:
        raise gr.Error("No recording: press the microphone, say one line, stop, then Add line.")
    sr, audio = mic
    try:
        speech = speech_seconds(to_16k_mono(audio, sr)[0])
    except ValueError:
        speech = 0.0
    if speech < MIN_SPEECH_S:
        raise gr.Error("No speech detected in that recording (silence or noise), so nothing was added. "
                       "Record the line again.")
    state = dict(state, example=None, clip_end=0.0)
    return (_process(state, audio, sr), *views(state), None)


def set_problem(state, text):
    text = (text or "").strip()
    if not text:
        raise gr.Error("Type the problem as LaTeX, e.g. x^2 + 5x = 50.")
    from src.checker import read_line
    try:
        ok = read_line(text) is not None
    except Exception:
        ok = False
    if not ok:
        gr.Warning("That problem could not be read as maths; later lines will be unchecked.")
    cancel_tts(state)
    state = fresh()
    state["history"].append(pipe().given_line(text))
    return (state, *views(state))


def resolve(state, which):
    """Apply a resolution button to the latest line (the user's choice; never automatic)."""
    if not state["history"] or not resolvable(state["history"][-1]):
        gr.Warning("Nothing to choose on the latest line.")
        return (state, *views(state))
    cancel_tts(state)
    r = dict(state["history"][-1])
    if r["status"] == "ask_back":
        pick = r["options"][0 if which == "A" else 1]
        r.update(output=pick, resolved_from=f"you chose {which}")
    elif r["status"] == "math_error" and r.get("suggestion"):
        if which == "A":
            r.update(output=r["suggestion"]["latex"], status="corrected", heard=r["heard"],
                     resolved_from="corrected by you (the sound-alike reading)")
        else:
            r.update(resolved_from="kept as said")
    state = dict(state, history=state["history"][:-1] + [r], tts=None)
    return (state, *views(state))


def load_example(state, title):
    exs = examples()
    if title not in exs:
        raise gr.Error("Pick an example first.")
    d = exs[title]
    meta = json.loads((d / "example.json").read_text())
    cancel_tts(state)
    state = fresh()
    state["example"] = {"dir": str(d), "lines": meta["lines"]}
    return next_line(state)


def next_line(state):
    ex = state.get("example")
    if not ex or state["pos"] >= len(ex["lines"]):
        raise gr.Error("No example line left: Load example again or Reset.")
    if pending_choice(state):
        raise gr.Error("Choose A or B for the blue line first; the next line is checked against your choice.")
    ln = ex["lines"][state["pos"]]
    wav = Path(ex["dir"]) / ln["wav"]
    audio, sr = load_audio(wav)
    state = dict(state, pos=state["pos"] + 1, clip_end=time.time() + len(audio) / sr)
    state = _process(state, audio, sr)
    return (state, *views(state), str(wav))


def fetch_tts(state):
    """Wait for the pre-generated explanation; play it after the example clip finishes."""
    key = state.get("tts")
    job = TTS_JOBS.get(key) if key else None
    if job is None:
        return gr.update(), gr.update()
    deadline = time.time() + 60
    while not job.done() and key not in CANCELLED and time.time() < deadline:
        time.sleep(0.1)
    if key in CANCELLED:
        return gr.update(), gr.update()
    TTS_JOBS.pop(key, None)
    try:
        path = job.result(timeout=0) if job.done() else None
    except Exception:
        path = None
    text = state["history"][-1].get("spoken_explanation") or ""
    if path is None:
        return None, f"(spoken explanation unavailable, TTS server not running) {text}"
    while time.time() < state.get("clip_end", 0.0) + 0.3 and key not in CANCELLED:
        time.sleep(0.1)
    if key in CANCELLED:
        return gr.update(), gr.update()
    return path, text


def undo(state):
    """Remove the latest line (a stray or misheard recording); an example steps back one line."""
    if not state["history"]:
        raise gr.Error("Nothing to remove.")
    cancel_tts(state)
    pos = state["pos"]
    if state.get("example") and pos > 0 and len(state["history"]) == pos:
        pos -= 1
    state = dict(state, history=state["history"][:-1], tts=None, pos=pos)
    return (state, *views(state), None, "")


def reset(state):
    cancel_tts(state)
    state = fresh()
    return (state, *views(state), None, None, "")


# ---------- layout ----------

def build():
    titles = list(examples())
    with gr.Blocks(css=CSS, title="Speech to LaTeX: dictate a derivation") as demo:
        state = gr.State(fresh())
        with gr.Row():
            gr.Markdown("## Speech → LaTeX, checked line by line\nSay one line at a time. Each line is "
                        "transcribed, written as LaTeX, and checked against the line before it.")
            lat = gr.Markdown(elem_id="corner")
        with gr.Row():
            with gr.Column(scale=3):
                lines = gr.Markdown(render(fresh()), latex_delimiters=LATEX, sanitize_html=False)
                with gr.Row():
                    btn_a = gr.Button(visible=False, variant="primary")
                    btn_b = gr.Button(visible=False)
                say = gr.Textbox(label="Spoken explanation", interactive=False)
                speech = gr.Audio(label="Spoken explanation (audio)", autoplay=True, interactive=False)
            with gr.Column(scale=2):
                with gr.Row():
                    problem = gr.Textbox(label="Problem (line 0, LaTeX)", placeholder="x^2 + 5x = 50", scale=3)
                    set_btn = gr.Button("Set problem", scale=1)
                mic = gr.Audio(sources=["microphone"], type="numpy", label="Say one line")
                with gr.Row():
                    add_btn = gr.Button("Add line", variant="primary", scale=2)
                    undo_btn = gr.Button("Remove last line", interactive=False, scale=1)
                with gr.Accordion("Canned examples", open=True):
                    pick = gr.Dropdown(titles, value=titles[0] if titles else None, label="Example")
                    with gr.Row():
                        load_btn = gr.Button("Load example")
                        next_btn = gr.Button("Next example line", interactive=False)
                    clip = gr.Audio(label="Example line audio", autoplay=True, interactive=False)
                reset_btn = gr.Button("Reset", variant="stop")
                gr.Markdown("green verified · amber corrected · red math error · blue ask-back · grey unchecked")
        out = [state, lines, lat, btn_a, btn_b, next_btn, undo_btn]
        say_out = [speech, say]
        add_btn.click(add_line, [state, mic], out + [mic]).then(fetch_tts, state, say_out)
        set_btn.click(set_problem, [state, problem], out)
        load_btn.click(load_example, [state, pick], out + [clip]).then(fetch_tts, state, say_out)
        next_btn.click(next_line, state, out + [clip]).then(fetch_tts, state, say_out)
        btn_a.click(lambda s: resolve(s, "A"), state, out)
        btn_b.click(lambda s: resolve(s, "B"), state, out)
        undo_btn.click(undo, state, out + [speech, say])
        reset_btn.click(reset, state, out + [clip, speech, say])
    return demo


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--share", action="store_true")
    ap.add_argument("--port", type=int, default=7860)
    ap.add_argument("--auth", default=os.environ.get("DEMO_AUTH"),
                    help="user:password login (required with --share; or set DEMO_AUTH)")
    ap.add_argument("--host", default="127.0.0.1", help="bind address (0.0.0.0 to expose on the LAN)")
    a = ap.parse_args()
    auth = tuple(a.auth.split(":", 1)) if a.auth else None
    if a.share and not auth:
        ap.error("--share opens a public link to the GPU and the paid LLM API; pass --auth user:password")
    CACHE.mkdir(parents=True, exist_ok=True)
    tts_live.start_server(log=str(CACHE / "tts_server.log"))
    pipe()   # load models before the first visitor
    demo = build()
    demo.queue(default_concurrency_limit=1)
    demo.launch(share=a.share, auth=auth, server_name=a.host, server_port=a.port,
                allowed_paths=[str(EXAMPLES), str(CACHE)])


if __name__ == "__main__":
    main()
