"""Generate the 400 Phase 3 derivations (seed 42) -> data/derivations/derivations.json.

Usage: .venv/bin/python -W ignore scripts/gen_derivations.py
Every line must parse with parse_graded; the parse rate is asserted and printed.
"""

import json
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.config import DATA_DIR, SEED  # noqa: E402
from src.derivgen import generate, summary  # noqa: E402
from src.mathparse import parse_graded  # noqa: E402

OUT = DATA_DIR / "derivations" / "derivations.json"


def main():
    items = generate(SEED)
    levels = Counter(parse_graded(s)[1] for it in items for s in it["lines"] + [it["solution"]])
    n = sum(levels.values())
    rate = 1 - levels["none"] / n
    print(f"parse rate {rate:.4f} over {n} lines {dict(levels)}")
    assert rate == 1.0, "every derivation line must parse"
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(items, indent=1))
    s = summary(items)
    print(f"{len(items)} items -> {OUT}")
    print("types", dict(s["types"]))
    print("errors", dict(s["errors"]), f"rate {s['error_rate']:.2f}")
    print("splits", dict(s["splits"]))
    print("errors x split", dict(Counter((i["error"]["type"], i["split"]) for i in items if i["error"])))
    print("type x split", dict(Counter((i["type"], i["split"]) for i in items)))
    print("lines per item", dict(Counter(len(i["lines"]) for i in items)))


if __name__ == "__main__":
    main()
