"""Gradio demo (Phase 5): dictate a derivation line by line; every line is checked against the last one.

Run:  .venv/bin/python -W ignore demo/app.py [--port 7860] [--share --auth user:password]
(local-only on 127.0.0.1 by default; a public share link requires a login.)
Starts the TTS server (scripts/tts_server.py) for spoken explanations if it is not running; without
it, explanations are shown as text only. The decision logic is src/pipeline.py (shipped system:
augmented pools, frozen System D, checker, class-conditional δ).

Layout: a document (left) of the lines so far, each with its comment card on the right (like a
Google Docs comment; demo/ui_render.py), then one sequential input: the microphone first, typed
LaTeX as the secondary option. "Accessible view" enlarges text and targets and raises contrast.
"""

import argparse
import json
import os
import sys
import tempfile
import time
import uuid
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import gradio as gr

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from src import tts_live  # noqa: E402
from src.audio import load_audio, speech_seconds, to_16k_mono  # noqa: E402
import ui_render as ui  # noqa: E402

EXAMPLES = Path(__file__).resolve().parent / "examples"
CACHE = Path(os.environ.get("DEMO_CACHE", Path(tempfile.gettempdir()) / "speech-to-latex-demo"))
ORDER = ["quadratic", "askback", "rec_error", "rec_clean"]   # the narrated order (demo/script.md)

PIPE = None
TTS_POOL = ThreadPoolExecutor(1)
TTS_JOBS = {}
TTS_FAILED = set()     # TTS keys whose synthesis failed (card shows text only)
CANCELLED = set()      # TTS keys that must not autoplay (line reset, undone, resolved or superseded)
MIN_SPEECH_S = 0.25    # mic clips with less speech than this are rejected before ASR
LEAKS = ("Variables already in use",)   # LLM prompt text that must never be accepted as a line
HIDDEN = ("latency_s", "latency_parts", "llm_timing")   # never shown (cached runs made them misleading)
pretty = ui.pretty


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


def file_url(path):
    """Relative Gradio file URL (works behind a path-prefix proxy such as VS Code port forwarding)."""
    return f"gradio_api/file={path}"


# ---------- state and views ----------

def fresh():
    return {"history": [], "example": None, "pos": 0, "tts": None, "clip_end": 0.0}


def pending_choice(state):
    """True while the latest line is an unanswered ask-back: no new line until A or B is chosen."""
    return ui.pending(state["history"]) == "ask_back"


def resolvable(r):
    return bool(r and not r.get("resolved_from")
                and (r["status"] == "ask_back" or (r["status"] == "math_error" and r.get("suggestion"))))


def audio_url(r):
    key = r.get("tts_key")
    if not key or key in TTS_FAILED:
        return None
    return file_url(CACHE / f"{key}.wav")


def render_doc(state, show_history=True, show_heard=False):
    return ui.doc_html(state["history"], show_history, show_heard, audio_url,
                       lambda r: r.get("tts_key") in TTS_FAILED)


def finished(state):
    ex = state.get("example")
    return bool(ex and state["history"] and state["pos"] >= len(ex["lines"]))


def example_button(state):
    """The primary "Play next line" button next to the input; shown only while an example runs."""
    ex = state.get("example")
    if not ex:
        return gr.update(visible=False, interactive=False, value="Play next line")
    left = len(ex["lines"]) - state["pos"]
    if left and pending_choice(state):
        return gr.update(visible=True, interactive=False, value="Choose A or B first")
    return gr.update(visible=True, interactive=left > 0,
                     value=f"▶ Play next example line ({left} left)" if left else "Example finished")


def dev_details(state):
    if not state["history"]:
        return {}
    r = state["history"][-1]
    return {k: v for k, v in r.items() if k not in HIDDEN}


def views(state, show_history=True, show_heard=False, mic=None, error=None):
    """(doc, status, mic, next, undo, dev) for a state. mic: an extra mic update (e.g. clear it);
    error: a message shown in the status box instead of the normal status (nothing else changes)."""
    phase = ("error", error) if error else None
    return (render_doc(state, show_history, show_heard), ui.status_html(state["history"], phase),
            gr.update(**(mic or {})), example_button(state),
            gr.update(interactive=bool(state["history"])), dev_details(state))


def step_html(state):
    return f'<h2 class="step">{ui.esc(ui.step_title(state["history"], finished(state)))}</h2>'


def cancel_tts(state, drop=False):
    """Stop the latest spoken explanation from autoplaying; drop=True also cancels its synthesis
    (the line itself is gone)."""
    if state and state.get("tts"):
        CANCELLED.add(state["tts"])
        if drop:
            job = TTS_JOBS.pop(state["tts"], None)
            if job:
                job.cancel()


# ---------- actions ----------

def _tts_job(r):
    """Pre-generate the spoken explanation of a commented line in the background; returns its key."""
    text = r.get("spoken_explanation")
    if not text or r["status"] not in ("math_error", "ask_back", "corrected"):
        return None
    key = uuid.uuid4().hex
    CACHE.mkdir(parents=True, exist_ok=True)
    job = TTS_POOL.submit(tts_live.synth, text, CACHE / f"{key}.wav")

    def done(f, key=key):
        if f.cancelled() or f.exception() is not None or f.result() is None:
            TTS_FAILED.add(key)
    job.add_done_callback(done)
    TTS_JOBS[key] = job
    return key


def _accept(state, r):
    cancel_tts(state)
    r = dict(r)
    key = _tts_job(r)
    r["tts_key"] = key
    state = dict(state, history=state["history"] + [r], tts=key)
    return state


def _process(state, audio, sr):
    if pending_choice(state):
        raise gr.Error("Choose A or B in the comment beside the line first; the next line is checked against your choice.")
    try:
        r = pipe().process_line(audio, sr, state["history"])
    except ValueError as e:
        raise gr.Error(f"Could not use that recording: {e}")
    except Exception as e:   # model / network failures: say which stage, keep the session alive
        raise gr.Error(f"Processing failed ({type(e).__name__}: {e}). Check the LLM API key / network and "
                       f"try again, or try an example.")
    if not r["output"].strip() or any(x in r["output"] for x in LEAKS):
        raise gr.Error("Nothing usable was recognised, so the line was not added. Say the line again.")
    return _accept(state, r)


def _out(state, sh, hd, mic=None, speaker="", clip=None, typed=None, error=None):
    """All outputs: state, doc, status, mic, next, undo, dev, step, speaker, clip, typed."""
    return (state, *views(state, sh, hd, mic, error), step_html(state), speaker,
            gr.update() if clip is None else clip, gr.update() if typed is None else typed)


def safe(fn):
    """Event handler wrapper: a refused action (gr.Error) or a failure is shown in the status box,
    with the page otherwise unchanged; nothing raises to Gradio (which would badge every output)."""
    def run(state, *args):
        sh, hd = args[-2:]
        try:
            return fn(state, *args)
        except gr.Error as e:
            msg = getattr(e, "message", None) or str(e)
        except Exception as e:   # keep the session alive whatever happens
            msg = f"Something went wrong ({type(e).__name__}: {e}). Try again, or try an example."
        return _out(state, sh, hd, speaker=gr.update(), error=msg)
    run.__name__ = fn.__name__
    return run


def add_line(state, mic, sh=True, hd=False):
    if mic is None:
        raise gr.Error("No recording yet: press Record, say one line, then press Stop.")
    sr, audio = mic
    try:
        speech = speech_seconds(to_16k_mono(audio, sr)[0])
    except ValueError:
        speech = 0.0
    if speech < MIN_SPEECH_S:
        raise gr.Error("No speech detected in that recording (silence or noise), so nothing was added. "
                       "Record the line again.")
    state = dict(state, example=None, clip_end=0.0)
    return _out(_process(state, audio, sr), sh, hd, mic={"value": None}, clip="")


def add_line_auto(state, mic, sh=True, hd=False):
    """On Stop: check the recording straight away (no-op if the recording has not arrived)."""
    if mic is None:
        return _out(state, sh, hd)
    return add_line(state, mic, sh, hd)


def add_typed(state, text, sh=True, hd=False):
    """A typed LaTeX line: the problem if there is none yet, else checked against the last line."""
    text = (text or "").strip()
    if not text:
        raise gr.Error("Type a line as LaTeX first, e.g. x^2 + 5x = 50.")
    if pending_choice(state):
        raise gr.Error("Choose A or B in the comment beside the line first; the next line is checked against your choice.")
    if not state["history"] and not pipe().readable(text):
        gr.Warning("That problem could not be read as maths; later lines will be unchecked.")
    try:
        r = pipe().typed_line(text, state["history"])
    except Exception as e:
        raise gr.Error(f"Could not check that line ({type(e).__name__}: {e}).")
    state = _accept(dict(state, example=None, clip_end=0.0), r)
    return _out(state, sh, hd, clip="", typed="")


def set_problem(state, text, sh=True, hd=False):
    """Start over with a typed problem (line 0)."""
    cancel_tts(state, drop=True)
    return add_typed(fresh(), text, sh, hd)


def resolve(state, which, sh=True, hd=False):
    """Apply a resolution button to the latest line (the user's choice; never automatic)."""
    if not state["history"] or not resolvable(state["history"][-1]):
        gr.Warning("Nothing to choose on the latest line.")
        return _out(state, sh, hd)
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
    return _out(state, sh, hd)


def load_example(state, title, sh=True, hd=False):
    exs = examples()
    if title not in exs:
        raise gr.Error("Pick an example first.")
    d = exs[title]
    meta = json.loads((d / "example.json").read_text())
    cancel_tts(state, drop=True)
    state = fresh()
    state["example"] = {"dir": str(d), "lines": meta["lines"]}
    return next_line(state, sh, hd)


def next_line(state, sh=True, hd=False):
    ex = state.get("example")
    if not ex or state["pos"] >= len(ex["lines"]):
        raise gr.Error("No example line left: start the example again, or Start over.")
    if pending_choice(state):
        raise gr.Error("Choose A or B in the comment beside the line first; the next line is checked against your choice.")
    ln = ex["lines"][state["pos"]]
    wav = Path(ex["dir"]) / ln["wav"]
    audio, sr = load_audio(wav)
    state = dict(state, pos=state["pos"] + 1, clip_end=time.time() + len(audio) / sr)
    state = _process(state, audio, sr)
    src = ui.esc(f"{file_url(wav)}?n={uuid.uuid4().hex[:8]}")
    clip = (f'<div class="clip"><audio autoplay src="{src}"></audio>'
            f'<span class="cliplabel">Example recording of {ui.esc(ui.line_label(state["pos"] - 1).lower())} '
            f'is playing.</span><button type="button" class="play" data-play="{src}">▶ Replay recording</button>'
            '</div>')
    return _out(state, sh, hd, clip=clip)


def fetch_tts(state, read_aloud=True):
    """Autoplay the latest spoken explanation (after an example clip ends) if reading aloud is on."""
    key = state.get("tts")
    job = TTS_JOBS.get(key) if key else None
    if job is None or not read_aloud:
        return gr.update()
    deadline = time.time() + 60
    while not job.done() and key not in CANCELLED and time.time() < deadline:
        time.sleep(0.1)
    if key in CANCELLED or not job.done():
        return gr.update()
    path = job.result() if not job.cancelled() and job.exception() is None else None
    if path is None:
        gr.Info("Spoken explanations are unavailable right now (TTS server not running); comments are shown as text.")
        return gr.update()
    while time.time() < state.get("clip_end", 0.0) + 0.3 and key not in CANCELLED:
        time.sleep(0.1)
    if key in CANCELLED:
        return gr.update()
    return f'<audio autoplay src="{ui.esc(file_url(path))}"></audio>'


def undo(state, sh=True, hd=False):
    """Remove the latest line (a stray or misheard recording); an example steps back one line."""
    if not state["history"]:
        raise gr.Error("Nothing to remove.")
    cancel_tts(state, drop=True)
    pos = state["pos"]
    if state.get("example") and pos > 0 and len(state["history"]) == pos:
        pos -= 1
    state = dict(state, history=state["history"][:-1], tts=None, pos=pos)
    return _out(state, sh, hd, clip="", typed="")


def reset(state, sh=True, hd=False):
    cancel_tts(state, drop=True)
    return _out(fresh(), sh, hd, mic={"value": None}, clip="", typed="")


def rerender(state, sh, hd):
    return render_doc(state, sh, hd)


# ---------- page ----------

CSS = """
:root, .gradio-container {
  --ink: #1f2328; --muted: #59636e; --line: #d1d9e0; --paper: #ffffff; --wash: #f6f8fa;
  --ok: #1a7f37; --fix: #8a5a00; --err: #c3171f; --ask: #0a5bc4; --unk: #59636e; --given: #3d444d;
  --ok-bg: #effaf2; --fix-bg: #fff7e0; --err-bg: #fff0f0; --ask-bg: #eef5ff; --unk-bg: #f3f4f6; --given-bg: #f6f8fa;
  --focus: #0a5bc4; --rail: 340px; --fs: 17px;
}
body.a11y, body.a11y .gradio-container {
  --ink: #000; --muted: #2b2b2b; --line: #000; --ok: #0b5a1e; --fix: #5c3b00; --err: #8a0000;
  --ask: #00337f; --unk: #2b2b2b; --given: #000; --ok-bg: #fff; --fix-bg: #fff; --err-bg: #fff;
  --ask-bg: #fff; --unk-bg: #fff; --given-bg: #fff; --focus: #000; --rail: 420px; --fs: 23px;
  --text-sm: 18px; --text-md: 22px; --text-lg: 26px; --text-xl: 30px; --text-xxl: 34px;
  --block-label-text-size: 20px; --block-title-text-size: 22px; --checkbox-label-text-size: 21px;
  --button-large-text-size: 24px; --button-large-padding: 18px 26px; --input-text-size: 23px;
  --button-medium-text-size: 22px; --button-small-text-size: 20px; --checkbox-size: 28px;
  --body-text-color-subdued: #2b2b2b; --block-label-text-color: #000;
}
html { scrollbar-gutter: stable; }
.gradio-container { max-width: 1200px !important; margin: 0 auto !important; color: var(--ink);
  font-family: system-ui, -apple-system, "Segoe UI", Roboto, sans-serif !important; }
body { background: var(--paper); }
footer { display: none !important; }
#hdr h1 { font-size: 1.6em; margin: 0 0 4px; color: var(--ink); }
#hdr p { margin: 0; color: var(--muted); font-size: var(--fs); max-width: 62ch; }
#opts { gap: 6px 26px; align-items: center; justify-content: flex-start; flex-wrap: wrap; }
#opts > div { flex: 0 0 auto !important; width: auto !important; min-width: 0 !important; }
#opts label span { white-space: nowrap; }
.step { font-size: calc(var(--fs) * 1.25); margin: 4px 0 2px; color: var(--ink); }
:focus-visible { outline: 3px solid var(--focus) !important; outline-offset: 2px !important; }
body.a11y :focus-visible { outline: 5px solid var(--focus) !important; box-shadow: 0 0 0 8px #ffd400 !important; }
input::placeholder, textarea::placeholder { color: #6e7781 !important; font-style: italic; }

/* document grid: line cell + comment cell per row; cells keep their own height */
.doc { display: grid; grid-template-columns: minmax(0, 1fr) var(--rail); gap: 10px 22px; font-size: var(--fs);
  color: var(--ink); align-items: start; }
.cell.line { background: var(--paper); border: 1px solid var(--line); border-left: 6px solid var(--c, var(--line));
  border-radius: 8px; padding: 10px 14px; min-height: 64px; scroll-margin-top: 12px; }
.cell.line:focus { outline: none; } .cell.line:focus-visible { outline: 3px solid var(--focus) !important; }
.cell.line.placeholder, .cell.line.folded { border-style: dashed; color: var(--muted); }
.cell.line.latest { box-shadow: 0 0 0 2px var(--c, var(--line)); }
.cell.rail { position: relative; }
.lhead { display: flex; gap: 10px; align-items: baseline; flex-wrap: wrap; margin-bottom: 4px; }
.num { font-weight: 600; color: var(--muted); font-size: .85em; min-width: 4.2em; }
.badge { display: inline-flex; gap: 5px; align-items: center; font-weight: 600; font-size: .85em; color: var(--c);
  background: var(--cb); border: 1px solid var(--c); border-radius: 999px; padding: 1px 10px; }
.m { color: var(--ink); } .m math { font-size: 1.15em; }
.m.disp { display: block; text-align: left; } .m.disp math { font-size: 1.35em; margin: 2px 0; }
.m.raw code { font-size: .95em; }
.was { color: var(--muted); font-size: .9em; }
.heard { color: var(--muted); font-size: .85em; margin-top: 4px; }
.s-ok { --c: var(--ok); --cb: var(--ok-bg); } .s-fix { --c: var(--fix); --cb: var(--fix-bg); }
.s-err { --c: var(--err); --cb: var(--err-bg); } .s-ask { --c: var(--ask); --cb: var(--ask-bg); }
.s-unk { --c: var(--unk); --cb: var(--unk-bg); } .s-given { --c: var(--given); --cb: var(--given-bg); }

/* comment cards */
.card { background: var(--cb); border: 1px solid var(--c); border-left: 6px solid var(--c); border-radius: 8px;
  padding: 10px 14px; box-shadow: 0 1px 3px rgba(0,0,0,.08); position: relative; }
.card::before { content: ""; position: absolute; left: -23px; top: 26px; width: 17px; border-top: 2px solid var(--c); }
.ctitle { font-weight: 700; color: var(--c); display: flex; gap: 8px; align-items: baseline; }
.cbody p { margin: 6px 0; } .cbody .why { color: var(--muted); font-size: .9em; }
.choices { margin: 6px 0; padding-left: 0; list-style: none; } .choices li { margin: 4px 0; }
.ctools { display: flex; flex-wrap: wrap; gap: 8px; margin-top: 8px; align-items: center; }
.card button, .clip button { font: inherit; font-size: .95em; min-height: 40px; padding: 6px 14px; border-radius: 8px;
  cursor: pointer; border: 1.5px solid var(--c, var(--ask)); background: var(--paper); color: var(--c, var(--ask));
  font-weight: 600; }
.card button.primary { background: var(--c); color: #fff; }
.card button:disabled { opacity: .5; cursor: wait; }
body.a11y .card button, body.a11y .clip button { min-height: 56px; padding: 10px 20px; border-width: 3px; }
body.a11y .cell.line, body.a11y .card { border-width: 2px; border-left-width: 10px; }

/* status box */
.pill { display: flex; gap: 10px; align-items: flex-start; border-radius: 10px; padding: 12px 14px; font-size: var(--fs);
  border: 1px solid var(--c); border-left: 6px solid var(--c); background: var(--cb); color: var(--ink); }
.pill .ic { color: var(--c); font-weight: 700; }
.p-ready, .p-done { --c: var(--ok); --cb: var(--ok-bg); } .p-need { --c: var(--ask); --cb: var(--ask-bg); }
.p-flag, .p-error { --c: var(--err); --cb: var(--err-bg); }
.p-recording { --c: var(--err); --cb: var(--err-bg); } .p-processing { --c: var(--fix); --cb: var(--fix-bg); }
.p-recording .ic { animation: blink 1s steps(2) infinite; }
@keyframes blink { 50% { opacity: .2; } }
@media (prefers-reduced-motion: reduce) { .p-recording .ic { animation: none; } }
.guide { font-size: calc(var(--fs) * .9); color: var(--muted); } .guide ol { padding-left: 1.3em; margin: 6px 0; }
.preview { font-size: var(--fs); padding: 8px 12px; border: 1px dashed var(--line); border-radius: 8px; min-height: 44px; }
.preview.muted { color: var(--muted); } .preview.bad { color: var(--err); }
.clip { display: flex; flex-wrap: wrap; gap: 8px; align-items: center; font-size: calc(var(--fs) * .9); color: var(--muted); }
#mic { min-height: 150px; }
#inputrow { display: grid !important; grid-template-columns: minmax(0, 1fr) var(--rail); gap: 22px; }
#check-btn, #next-btn { font-size: calc(var(--fs) * 1.05); }
#speaker { display: none; }
/* no Gradio fade while a line is processed: the status box says what is happening */
.html-container.pending { opacity: 1 !important; }
.acts { display: flex; flex-wrap: wrap; gap: 8px; }
#next-btn:disabled { background: #e6eaef !important; color: #3d444d !important; border-color: #8c959f !important; }
.progress-text, .eta-bar { display: none !important; }   /* never show Gradio timers */
#reset-btn { border: 1.5px solid #a40e17 !important; }
button:focus-visible { transition: none !important; }
#dev span { color: #1f2328 !important; }   /* developer JSON: readable, not syntax-coloured */
#pick input { padding-right: 34px !important; text-overflow: ellipsis; }

/* accessible view: Gradio's own pieces */
body.a11y .label-wrap, body.a11y .label-wrap span { font-size: 22px !important; font-weight: 700; }
body.a11y input[type=checkbox] { width: 28px !important; height: 28px !important; }
body.a11y #opts label span { font-size: 21px; }
body.a11y #mic label[data-testid=block-label] { font-size: 20px; }
body.a11y #mic .controls button { font-size: 22px !important; min-height: 54px; padding: 10px 22px !important;
  border: 2px solid #000 !important; }
body.a11y #mic .controls select { font-size: 19px !important; max-width: 100%; min-height: 48px; }
body.a11y #mic .controls { flex-wrap: wrap; gap: 10px; }
body.a11y button { min-height: 52px; }

@media (max-width: 720px) {
  .doc, #inputrow { grid-template-columns: 1fr !important; }
  .cell.rail.has-card { margin: -4px 0 6px 18px; }
  .cell.rail:not(.has-card) { display: none; }
  .card::before { left: -12px; top: -10px; width: 0; height: 10px; border-top: 0; border-left: 2px solid var(--c); }
}
"""

JS = r"""
() => {
  const u = new URL(location.href);
  if (!u.searchParams.has('__theme')) { u.searchParams.set('__theme', 'light'); location.replace(u); return; }
  const icons = {recording: '●', processing: '…'};
  window.s2lPill = (kind, text) => {
    const el = document.querySelector('#status .pill'); if (!el) return;
    el.className = 'pill p-' + kind;
    el.innerHTML = '<span class="ic" aria-hidden="true">' + (icons[kind] || '●') + '</span>' + text;
  };
  const gbtn = (id) => { const h = document.getElementById(id); return h && (h.tagName === 'BUTTON' ? h : h.querySelector('button')); };
  // one voice at a time: Listen buttons, example clips and read-aloud comments never overlap
  let player = null, playing = null;
  const stopListen = () => { if (player) player.pause(); if (playing) playing.textContent = playing.dataset.label; };
  document.addEventListener('play', (e) => {
    if (e.target !== player) stopListen();
    document.querySelectorAll('audio').forEach(a => { if (a !== e.target) a.pause(); });
  }, true);
  let acted = false;
  document.addEventListener('click', (e) => {
    const act = e.target.closest('[data-act]');
    if (act) {
      const b = gbtn('bridge-' + act.dataset.act);
      if (b) { act.closest('.acts').querySelectorAll('button').forEach(x => x.disabled = true);
               acted = true; window.s2lPill('processing', 'Saving your answer…'); b.click(); }
      return;
    }
    const play = e.target.closest('[data-play]');
    if (play) {
      play.dataset.label = play.dataset.label || play.textContent;
      if (playing === play && player && !player.paused) { stopListen(); return; }
      stopListen();
      document.querySelectorAll('audio').forEach(a => a.pause());
      player = new Audio(play.dataset.play); playing = play;
      play.textContent = '❚❚ Pause';
      player.onended = () => { play.textContent = play.dataset.label; };
      player.onerror = () => { play.textContent = '▶ Listen (still preparing, try again)'; };
      player.play().catch(() => {});
      return;
    }
    const busy = e.target.closest('#check-btn, #typed-btn, #next-btn, #start-btn, #problem-btn');
    if (busy && !busy.disabled) window.s2lPill('processing', 'Processing: checking the line…');
  });
  // after every update: bring the newest line (and its comment) into view, and keep keyboard focus
  let lastSig = '';
  const motion = matchMedia('(prefers-reduced-motion: reduce)').matches ? 'auto' : 'smooth';
  const follow = () => {
    const lines = document.querySelectorAll('#doc .cell.line[data-line]');
    const last = lines[lines.length - 1];
    lines.forEach(l => l.classList.toggle('latest', l === last && lines.length > 1));
    const sig = last ? last.dataset.line + '|' + last.className + '|' + last.textContent + '|' + (last.nextElementSibling ? last.nextElementSibling.textContent : '') : '';
    if (sig === lastSig) return;
    const first = lastSig === ''; lastSig = sig;
    if (!last || first && lines.length < 2) return;
    const r = last.getBoundingClientRect();
    const nb0 = gbtn('next-btn');
    const tgt = (nb0 && nb0.offsetParent) ? nb0 : document.getElementById('status');
    const tr = tgt ? tgt.getBoundingClientRect() : r;
    if (r.top < 0 || tr.bottom > innerHeight - 8) last.scrollIntoView({block: 'start', behavior: motion});
    if (acted || !document.activeElement || document.activeElement === document.body) {
      const nb = gbtn('next-btn');
      if (nb && nb.offsetParent && !nb.disabled) nb.focus({preventScroll: true});
      else last.focus({preventScroll: true});
    }
    acted = false;
  };
  const watch = () => {
    const d = document.getElementById('doc');
    if (!d) { setTimeout(watch, 300); return; }
    new MutationObserver(() => setTimeout(follow, 50)).observe(d, {childList: true, subtree: true});
  };
  watch();
}
"""

HEADER = """<div id="hdr"><h1>Say your maths, one line at a time</h1>
<p>Each line is written as maths and checked against the line before it. Anything that needs a look
gets a comment beside its line.</p></div>"""

GUIDE = """<div class="guide"><b>How it works</b><ol>
<li>Press <b>Record</b> and say one line, for example <q>x squared plus five x equals fifty</q>.</li>
<li>Press <b>Stop</b>. The line is checked straight away.</li>
<li>Read or listen to the comment beside the line, answer it if it asks you something, then say the
next line.</li>
</ol>Typing? Open <b>Type LaTeX instead</b> below the microphone.</div>"""


def build():
    titles = list(examples())
    theme = gr.themes.Base(primary_hue="blue", neutral_hue="slate",
                           font=["system-ui", "-apple-system", "Segoe UI", "Roboto", "sans-serif"]).set(
        button_primary_background_fill="#0a5bc4", button_primary_background_fill_hover="#084a9f",
        button_primary_text_color="#ffffff", button_primary_border_color="#0a5bc4",
        button_cancel_background_fill="#ffffff", button_cancel_background_fill_hover="#fff0f0",
        button_cancel_text_color="#a40e17", button_cancel_border_color="#a40e17")
    with gr.Blocks(css=CSS, js=JS, theme=theme, title="Speech to LaTeX") as demo:
        state = gr.State(fresh())
        gr.HTML(HEADER)
        with gr.Row(elem_id="opts"):
            a11y = gr.Checkbox(False, label="Accessible view", elem_id="a11y", container=False)
            read_aloud = gr.Checkbox(True, label="Read comments aloud", container=False)
            show_history = gr.Checkbox(True, label="Show history", container=False)
            show_heard = gr.Checkbox(False, label="Show the words I said", container=False)
        doc = gr.HTML(render_doc(fresh()), elem_id="doc")
        with gr.Row(elem_id="inputrow", equal_height=False):
            with gr.Column(elem_id="inputcol", min_width=0):
                step = gr.HTML(step_html(fresh()))
                status = gr.HTML(ui.status_html([]), elem_id="status")
                next_btn = gr.Button("Play next line", variant="primary", size="lg", interactive=False,
                                     visible=False, elem_id="next-btn")
                clipbox = gr.HTML("")
                mic = gr.Audio(sources=["microphone"], type="numpy", label="Microphone", elem_id="mic",
                               show_download_button=False)
                check_btn = gr.Button("Check this line", variant="primary", size="lg", elem_id="check-btn")
                with gr.Accordion("Type LaTeX instead", open=False):
                    typed = gr.Textbox(label="LaTeX for this line", placeholder="e.g. x^2 + 5x = 50", lines=1,
                                       max_lines=3, elem_id="typed")
                    preview = gr.HTML(ui.preview_html(""))
                    with gr.Row():
                        typed_btn = gr.Button("Add typed line", variant="primary", elem_id="typed-btn")
                        problem_btn = gr.Button("Start over with this as the problem", elem_id="problem-btn")
                with gr.Row():
                    undo_btn = gr.Button("Remove last line", interactive=False)
                    reset_btn = gr.Button("Start over", variant="stop", elem_id="reset-btn")
            with gr.Column(elem_id="guidecol", min_width=0):
                gr.HTML(GUIDE)
                with gr.Accordion("Try an example", open=False):
                    gr.Markdown("Recorded examples play line by line, as if spoken into the microphone. "
                                "Use **Play next example line** beside the microphone to step through.")
                    pick = gr.Dropdown(titles, value=titles[0] if titles else None, label="Example",
                                       elem_id="pick")
                    start_btn = gr.Button("Start example", elem_id="start-btn")
                bridge_a = gr.Button("A", visible="hidden", elem_id="bridge-A")
                bridge_b = gr.Button("B", visible="hidden", elem_id="bridge-B")
        with gr.Accordion("Developer details", open=False):
            dev = gr.JSON(label="Latest line (pipeline output)", elem_id="dev")
        speaker = gr.HTML("", elem_id="speaker")

        opts = [show_history, show_heard]
        out = [state, doc, status, mic, next_btn, undo_btn, dev, step, speaker, clipbox, typed]
        quiet = dict(show_progress="hidden")   # the status box says what is happening; no Gradio timers
        say = dict(fn=fetch_tts, inputs=[state, read_aloud], outputs=speaker, **quiet)
        mic.start_recording(None, None, None, js="() => window.s2lPill('recording', 'Recording… press Stop when you have said the line.')")
        mic.stop_recording(None, None, None, js="() => window.s2lPill('processing', 'Processing: checking the line…')")
        mic.stop_recording(safe(add_line_auto), [state, mic] + opts, out, **quiet).then(**say)
        check_btn.click(safe(add_line), [state, mic] + opts, out, **quiet).then(**say)
        typed.change(ui.preview_html, typed, preview, trigger_mode="always_last", **quiet)
        typed.submit(safe(add_typed), [state, typed] + opts, out, **quiet).then(**say)
        typed_btn.click(safe(add_typed), [state, typed] + opts, out, **quiet).then(**say)
        problem_btn.click(safe(set_problem), [state, typed] + opts, out, **quiet)
        start_btn.click(safe(load_example), [state, pick] + opts, out, **quiet).then(**say)
        next_btn.click(safe(next_line), [state] + opts, out, **quiet).then(**say)
        bridge_a.click(safe(lambda s, sh, hd: resolve(s, "A", sh, hd)), [state] + opts, out, **quiet)
        bridge_b.click(safe(lambda s, sh, hd: resolve(s, "B", sh, hd)), [state] + opts, out, **quiet)
        undo_btn.click(safe(undo), [state] + opts, out, **quiet)
        reset_btn.click(safe(reset), [state] + opts, out, **quiet)
        for box in opts:
            box.change(rerender, [state] + opts, doc, **quiet)
        a11y.change(None, a11y, None, js="(v) => { document.body.classList.toggle('a11y', v); }")
        a11y.change(lambda v: gr.update(value=True) if v else gr.update(), a11y, read_aloud, **quiet)
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
                allowed_paths=[str(EXAMPLES), str(CACHE)], show_api=False)


if __name__ == "__main__":
    main()
