"""Client for the persistent TTS server (scripts/tts_server.py): spoken explanations in the demo.

src/tts.py pays the CosyVoice load on every batch; the demo needs one sentence in about a second, so
the model stays resident in a server process (tts-venv) and this client talks to its Unix socket.
Every call degrades gracefully: if the server is not running or fails, synth returns None and the
demo shows the explanation as text only.
"""

import json
import os
import socket
import subprocess
from pathlib import Path

from src.tts import TTS_PYTHON

SERVER = Path(__file__).resolve().parent.parent / "scripts" / "tts_server.py"
SOCKET = os.environ.get("TTS_SOCKET", "/tmp/speech-to-latex-tts.sock")


def _call(req, path=SOCKET, timeout=30.0):
    with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as s:
        s.settimeout(timeout)
        s.connect(path)
        s.sendall((json.dumps(req) + "\n").encode())
        buf = b""
        while not buf.endswith(b"\n"):
            chunk = s.recv(65536)
            if not chunk:
                break
            buf += chunk
    return json.loads(buf)


def available(path=SOCKET):
    """True if a server answers a ping."""
    try:
        return _call({"ping": True}, path, timeout=2.0).get("ok", False)
    except (OSError, ValueError):
        return False


def start_server(path=SOCKET, log=None):
    """Launch the server detached (no-op if one answers). Returns the Popen or None."""
    if available(path):
        return None
    out = open(log, "a") if log else subprocess.DEVNULL
    return subprocess.Popen([TTS_PYTHON, "-W", "ignore", str(SERVER), "--socket", path],
                            stdout=out, stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL,
                            start_new_session=True)


def synth(text, out_path, voice="v1", path=SOCKET, timeout=30.0):
    """Synthesise text to out_path via the server; the wav path, or None on any failure."""
    if not text:
        return None
    try:
        rep = _call({"text": text, "voice": voice, "out_path": str(out_path)}, path, timeout)
    except (OSError, ValueError):
        return None
    return rep.get("out_path") if "error" not in rep else None
