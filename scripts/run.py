"""Uruchamia eksperyment albo test dymny.

  python scripts/run.py check qwen3.5-4b Q4_K_M           # surowe odpowiedzi, bez gramatyki
  python scripts/run.py p1 --model qwen3.5-4b --limit 20  # szybki test na 20 parach
  python scripts/run.py p1                                # cały P1 (wszystkie modele z configu)
  python scripts/run.py p2 --model qwen3.5-9b --quant Q8_0
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from judge.config import experiment  # noqa: E402
from judge.runner import run, smoke_check  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("exp", help="p1 / p2 / p3 / check")
    ap.add_argument("args", nargs="*", help="dla check: <model> <quant>")
    ap.add_argument("--model", nargs="+")
    ap.add_argument("--quant", nargs="+")
    ap.add_argument("--dataset", nargs="+")
    ap.add_argument("--limit", type=int)
    a = ap.parse_args()

    if a.exp == "check":
        model, quant = a.args
        smoke_check(model, quant)
        return

    exp = experiment(a.exp)
    for m in a.model or exp["models"]:
        for q in a.quant or exp["quants"]:
            print(f"== {a.exp}: {m} {q}")
            path = run(a.exp, m, q, limit=a.limit, datasets=a.dataset)
            print("zapisano", path)


if __name__ == "__main__":
    main()
