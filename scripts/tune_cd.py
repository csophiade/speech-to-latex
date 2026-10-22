"""Tune λ (Systems C, D), γ (phoneme weight) and ε (LLM rank prior) for D, on en_dev ONLY. Writes results/tuned.json.

Selection metric: CER (primary Speech2Latex metric); ties broken by SymEq on the
parseable subset. en_test_500 is never touched here.
"""

import itertools
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from run_ablation import TUNED, ensure_pools, evaluate, score  # noqa: E402

from src import config  # noqa: E402

LAMBDAS = [0.5, 1, 2, 4]
GAMMAS = [0.05, 0.1, 0.25, 0.5]
EPSILONS = [0.01, 0.05, 0.1, 0.25, 0.5]


def main():
    rows, pools = ensure_pools("s2l_dev")
    grid = []
    for system, lam, gamma, eps in [("C", lam, None, None) for lam in LAMBDAS] + \
            [("D", lam, g, e) for lam, g, e in itertools.product(LAMBDAS, GAMMAS, EPSILONS)]:
        kwargs = {"lam": lam} | ({"gamma": gamma, "eps": eps} if gamma is not None else {})
        m = score(evaluate(system, "s2l_dev", rows, pools, **kwargs))
        grid.append({"system": system, **kwargs, "cer": m["cer"],
                     "symeq_on_parseable": m.get("symeq_on_parseable", m["symeq"]), "nem": m["nem"]})
        print(json.dumps(grid[-1]))
    best = {}
    for system in ("C", "D"):
        rows_s = [g for g in grid if g["system"] == system]
        best[system] = min(rows_s, key=lambda g: (g["cer"], -g["symeq_on_parseable"]))
    TUNED.write_text(json.dumps({"dataset": "s2l_dev", "n": len(rows), "selection": "min CER, tie: max SymEq",
                                 "C": best["C"], "D": best["D"], "grid": grid,
                                 "llm": config.LLM_MODEL, "seed": config.SEED}, indent=2) + "\n")
    print("best:", json.dumps({k: best[k] for k in ("C", "D")}))


if __name__ == "__main__":
    main()
