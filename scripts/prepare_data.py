"""Pobiera zbiory i zapisuje je we wspólnym formacie do data/processed/*.jsonl.

  python scripts/prepare_data.py              # wszystkie
  python scripts/prepare_data.py llmbar judgebench
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from judge.data import PREPARERS  # noqa: E402


def main():
    names = sys.argv[1:] or list(PREPARERS)
    for n in names:
        print(f"{n}: {PREPARERS[n]()} par")


if __name__ == "__main__":
    main()
