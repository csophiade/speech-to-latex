"""Generate the homophone stress set (seed 42) -> data/derivations/homophone.json.

Usage: .venv/bin/python -W ignore scripts/gen_homophone.py
~100 calib + ~100 test derivations (ids hom-XXXX, problems distinct from the 400 in derivations.json)
whose last line t says the confusion partner of a correct number (see src/homophone.py). Every item is
re-checked with src.homophone.check (said line wrong, original line valid, one homophone swap).
"""

import json
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.config import DATA_DIR, SEED  # noqa: E402
from src.homophone import check, generate  # noqa: E402
from src.mathparse import parse_graded  # noqa: E402

OUT = DATA_DIR / "derivations" / "homophone.json"
PHASE3 = DATA_DIR / "derivations" / "derivations.json"


def main():
    old0 = {it["lines"][0] for it in json.loads(PHASE3.read_text())}
    items = generate(SEED, exclude_line0=old0)
    for it in items:
        check(it)
        assert it["lines"][0] not in old0
        assert all(parse_graded(s)[1] != "none" for s in it["lines"])
    assert len({it["lines"][0] for it in items}) == len(items)
    OUT.write_text(json.dumps(items, indent=1))
    print(f"{len(items)} items -> {OUT}")
    h = lambda k: Counter((it["split"], it["homophone"][k]) for it in items)  # noqa: E731
    print("split", dict(Counter(it["split"] for it in items)))
    print("type x split", dict(Counter((it["type"], it["split"]) for it in items)))
    print("position x split", dict(h("position")))
    print("direction x split", dict(h("direction")))
    print("t (error line)", dict(Counter(it["error"]["line"] for it in items)))
    print("type x position", dict(Counter((it["type"], it["homophone"]["position"]) for it in items)))
    print("correct->said", dict(Counter(f'{it["homophone"]["correct"]}->{it["homophone"]["said"]}' for it in items)))


if __name__ == "__main__":
    main()
