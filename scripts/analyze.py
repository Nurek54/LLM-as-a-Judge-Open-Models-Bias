"""Liczy metryki z results/<exp>/*.jsonl i zapisuje tabelę CSV.

  python scripts/analyze.py p1
  python scripts/analyze.py p2 --ci accuracy pc     # + przedziały ufności (bootstrap)
  python scripts/analyze.py p2 --mcnemar qwen3.5-9b BF16 Q4_K_M --dataset llmbar
  python scripts/analyze.py p3 --mcnemar-lang bielik-11b Q4_K_M
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from judge.config import RESULTS  # noqa: E402
from judge.metrics import GROUP, bootstrap_ci, load_results, mcnemar_paired, summarize  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("exp")
    ap.add_argument("--ci", nargs="*", default=[])
    ap.add_argument("--mcnemar", nargs=3, metavar=("MODEL", "QUANT_A", "QUANT_B"))
    ap.add_argument("--mcnemar-lang", nargs=2, metavar=("MODEL", "QUANT"))
    ap.add_argument("--dataset")
    a = ap.parse_args()

    df = load_results(sorted((RESULTS / a.exp).glob("*.jsonl")))
    if df.empty:
        sys.exit("brak wyników")

    if a.mcnemar:
        m, qa, qb = a.mcnemar
        sel = (df["model"] == m) & (df["dataset"] == a.dataset if a.dataset else True)
        print(mcnemar_paired(df[sel & (df["quant"] == qa)], df[sel & (df["quant"] == qb)]))
        return
    if a.mcnemar_lang:
        m, q = a.mcnemar_lang
        sel = (df["model"] == m) & (df["quant"] == q)
        print(mcnemar_paired(df[sel & (df["lang"] == "en")], df[sel & (df["lang"] == "pl")]))
        return

    summ = summarize(df)
    for metric in a.ci:
        lo, hi = [], []
        for _, row in summ.iterrows():
            g = df.loc[(df[GROUP] == row[GROUP]).all(axis=1)]
            l, h = bootstrap_ci(g, metric)
            lo.append(l)
            hi.append(h)
        summ[f"{metric}_lo"], summ[f"{metric}_hi"] = lo, hi
    out = RESULTS / a.exp / "summary.csv"
    summ.to_csv(out, index=False)
    with pd.option_context("display.max_columns", None, "display.width", 200):
        print(summ.round(3))
    print("zapisano", out)


if __name__ == "__main__":
    main()
