"""Safe LaTeX -> SymPy parsing shared by the eval metrics, System B and the checker."""

import re
import signal
import threading
from contextlib import contextmanager

import sympy
from sympy.parsing.latex import parse_latex

from src.config import SYMPY_TIMEOUT_S

_CMD = re.compile(r"\\([A-Za-z]+)")
_STRIP = re.compile(r"\\left|\\right|\\[,;:!]|\\quad|\\qquad|\\displaystyle")


class Timeout(Exception):
    pass


@contextmanager
def time_limit(seconds):
    """Raise Timeout after `seconds` (SIGALRM; a no-op outside the main thread)."""
    if threading.current_thread() is not threading.main_thread():
        yield
        return

    def handler(signum, frame):
        raise Timeout()

    old = signal.signal(signal.SIGALRM, handler)
    signal.setitimer(signal.ITIMER_REAL, seconds)
    try:
        yield
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
        signal.signal(signal.SIGALRM, old)


def clean_for_parse(latex):
    s = (latex or "").strip().strip("$").strip()
    s = _STRIP.sub(" ", s)
    s = s.rstrip(".,; ")
    return s.strip()


def _valid(expr, latex):
    if not isinstance(expr, sympy.Basic):
        return False   # lark returns a Tree for ambiguous input
    if isinstance(expr, sympy.core.relational.Relational) and any(
            isinstance(a, sympy.core.relational.Relational) for a in expr.args):
        return False   # "a = b = c" parses as Eq(Eq(a, b), c)
    # ANTLR turns unknown commands like \text into a Symbol named "text"
    names = {s.name for s in expr.free_symbols}
    return not any(cmd in names for cmd in _CMD.findall(latex))


def try_parse(latex, timeout=SYMPY_TIMEOUT_S):
    """Return a SymPy object, or None if `latex` is not parseable (ANTLR, then lark)."""
    s = clean_for_parse(latex)
    if not s:
        return None
    for kwargs in ({"strict": True}, {"backend": "lark"}):
        try:
            with time_limit(timeout):
                expr = parse_latex(s, **kwargs)
            if _valid(expr, s):
                return expr
        except Timeout:
            return None
        except Exception:
            continue
    return None


def parses(latex):
    return try_parse(latex) is not None
