"""Raport P2: metryki w funkcji precyzji wag, testy względem BF16, wykresy i tabela LaTeX.

  python scripts/report_p2.py

Wyniki w results/p2/report/:
  summary.csv      metryki dla model x poziom x zbiór (+ bity/wagę z configs/models_info.json)
  tests.csv        porównania każdego poziomu z BF16 (test permutacyjny na poziomie par, Holm)
  fig_p2.pdf/png   małe wielokrotności: wiersze = metryki, kolumny = zbiory, x = bity/wagę
  table_p2.tex     tabela do raportu (accuracy, PC, first_rate, format_mass)

Test: jednostką jest PARA, nie wywołanie. Dla każdej pary liczymy średnią z obu kolejności
(greedy), różnicę względem BF16 i test permutacyjny ze zmianą znaku (sign-flip). Dzięki temu
dwie kolejności tej samej pary nie są liczone jako niezależne obserwacje.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import matplotlib.ticker  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from judge.config import CONFIGS, RESULTS  # noqa: E402
from judge.metrics import load_results, summarize  # noqa: E402

QUANTS = ["BF16", "Q8_0", "Q6_K", "Q4_K_M", "Q3_K_M"]
DATASETS = ["llmbar", "judgebench", "evalbiasbench"]
DS_LABEL = {"llmbar": "LLMBar", "judgebench": "JudgeBench", "evalbiasbench": "EvalBiasBench"}
METRICS = [  # (kolumna, etykieta osi, linia odniesienia)
    ("accuracy", "Trafność", None),
    ("pc", "Spójność po zamianie (PC)", None),
    ("first_rate", "Odsetek wyborów pozycji A", 0.5),
    ("format_mass", "Masa P(A)+P(B)", None),
]
# Paleta kategoryczna (slot 1 i 2, stała kolejność) + różne markery jako drugi kanał.
STYLE = {0: dict(color="#2a78d6", marker="o"), 1: dict(color="#eb6834", marker="s")}
INK, MUTED, GRID = "#1f1f1e", "#6b6a63", "#e4e3de"
OUT = RESULTS / "p2" / "report"


def bpw_table() -> pd.DataFrame:
    p = CONFIGS / "models_info.json"
    if not p.exists():
        sys.exit("brak configs/models_info.json: uruchom `python scripts/models.py info`")
    return pd.DataFrame(json.loads(p.read_text(encoding="utf-8")))[["model", "quant", "bpw"]]


# ----------------------------------------------------------------------- testy
def _pair_scores(df: pd.DataFrame, value: str) -> pd.Series:
    g = df[df["repeat"] == 0]
    if value == "correct":
        return g.groupby("pair_id")["correct"].mean()
    if value == "consistent":  # 1 gdy ten sam wybór w obu kolejnościach
        t = g.pivot_table(index="pair_id", columns="order", values="chosen", aggfunc="first")
        return (t["12"] == t["21"]).astype(float)
    if value == "first":
        return g.assign(a=(g["slot"] == "A").astype(float)).groupby("pair_id")["a"].mean()
    if value == "longer":  # odsetek wyborów dłuższej odpowiedzi (obie kolejności)
        return g.assign(l=g["chose_longer"].astype(float)).groupby("pair_id")["l"].mean()
    raise ValueError(value)


def sign_flip_test(d: np.ndarray, n_perm: int = 20000, seed: int = 0) -> float:
    """Dwustronny test permutacyjny dla średniej różnic sparowanych (zmiana znaku)."""
    d = d[~np.isnan(d)]
    if len(d) == 0 or np.allclose(d, 0):
        return 1.0
    rng = np.random.default_rng(seed)
    obs = abs(d.mean())
    signs = rng.choice([-1.0, 1.0], size=(n_perm, len(d)))
    null = np.abs((signs * d).mean(axis=1))
    return float((np.sum(null >= obs - 1e-12) + 1) / (n_perm + 1))


def holm(p: pd.Series) -> pd.Series:
    order = p.sort_values().index
    m = len(p)
    adj, running = pd.Series(index=p.index, dtype=float), 0.0
    for i, k in enumerate(order):
        running = max(running, min(1.0, (m - i) * p[k]))
        adj[k] = running
    return adj


def tests(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for model in sorted(df["model"].unique()):
        for ds in DATASETS:
            base = df[(df.model == model) & (df.dataset == ds) & (df.quant == "BF16")]
            if base.empty:
                continue
            for q in QUANTS[1:]:
                cur = df[(df.model == model) & (df.dataset == ds) & (df.quant == q)]
                if cur.empty:
                    continue
                for value, name in [("correct", "accuracy"), ("consistent", "pc"),
                                    ("first", "first_rate"), ("longer", "longer_rate")]:
                    a, b = _pair_scores(base, value), _pair_scores(cur, value)
                    common = a.index.intersection(b.index)
                    d = (b[common] - a[common]).to_numpy()
                    rows.append({"model": model, "dataset": ds, "quant": q, "metric": name,
                                 "n_pairs": len(common), "bf16": a[common].mean(),
                                 "value": b[common].mean(), "diff": d.mean(),
                                 "p": sign_flip_test(d)})
    t = pd.DataFrame(rows)
    # Holm w obrębie rodziny: model x zbiór x metryka (4 porównania z BF16)
    t["p_holm"] = t.groupby(["model", "dataset", "metric"])["p"].transform(holm)
    t["sig"] = np.where(t["p_holm"] < 0.05, "*", "")
    return t


# --------------------------------------------------------------------- wykres
def figure(summ: pd.DataFrame, path: Path) -> None:
    models = sorted(summ["model"].unique())
    fig, axes = plt.subplots(len(METRICS), len(DATASETS), figsize=(11, 11), sharex=True)
    for r, (col, label, ref) in enumerate(METRICS):
        for c, ds in enumerate(DATASETS):
            ax = axes[r, c]
            for i, m in enumerate(models):
                s = summ[(summ.model == m) & (summ.dataset == ds)].sort_values("bpw")
                if s.empty:
                    continue
                st = STYLE[i % len(STYLE)]
                ax.plot(s["bpw"], s[col], color=st["color"], lw=2, marker=st["marker"],
                        ms=7, mec="white", mew=1.5, label=m, zorder=3)
            if ref is not None:
                ax.axhline(ref, color=MUTED, lw=1, ls="--", zorder=1)
            ax.set_xscale("log", base=2)
            ax.set_xticks([4, 5, 6.5, 8.5, 16])
            ax.set_xticklabels(["4", "5", "6,5", "8,5", "16"])
            ax.grid(True, color=GRID, lw=0.8)
            ax.set_axisbelow(True)
            for sp in ("top", "right"):
                ax.spines[sp].set_visible(False)
            for sp in ("left", "bottom"):
                ax.spines[sp].set_color(MUTED)
            ax.tick_params(colors=MUTED, labelsize=9)
            if r == 0:
                ax.set_title(DS_LABEL[ds], color=INK, fontsize=11)
            if c == 0:
                ax.set_ylabel(label, color=INK, fontsize=10)
            if r == len(METRICS) - 1:
                ax.set_xlabel("bity na wagę (BF16 → Q3_K_M)", color=INK, fontsize=9)
    axes[0, 0].invert_xaxis()  # osie x są wspólne: odwracamy RAZ (BF16 po lewej)
    comma = matplotlib.ticker.FuncFormatter(lambda v, _: f"{v:g}".replace(".", ","))
    for ax in axes.flat:
        ax.yaxis.set_major_formatter(comma)
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=len(models), frameon=False,
               fontsize=10, bbox_to_anchor=(0.5, 1.0))
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    fig.savefig(path.with_suffix(".pdf"))
    fig.savefig(path.with_suffix(".png"), dpi=200)
    plt.close(fig)


# --------------------------------------------------------------------- LaTeX
def latex_table(summ: pd.DataFrame, t: pd.DataFrame, path: Path) -> None:
    sig = {(r.model, r.dataset, r.quant, r.metric): r.sig for r in t.itertuples()}
    lines = [r"\begin{tabular}{llrrrrrr}", r"\toprule",
             r"Model & Zbiór & Poziom (bpw) & Trafność & PC & Pozycja A & Dłuższa & Masa A+B \\",
             r"\midrule"]
    for m in sorted(summ["model"].unique()):
        for ds in DATASETS:
            s = summ[(summ.model == m) & (summ.dataset == ds)].copy()
            s["quant"] = pd.Categorical(s["quant"], QUANTS)
            for r in s.sort_values("quant").itertuples():
                def cell(metric, val):
                    return f"{val:.3f}".replace(".", ",") + sig.get((m, ds, r.quant, metric), "")
                quant = r.quant.replace("_", "\\_")
                bpw = f"{r.bpw:.1f}".replace(".", ",")
                mass = f"{r.format_mass:.3f}".replace(".", ",")
                lines.append(
                    f"{m} & {DS_LABEL[ds]} & {quant} ({bpw}) & {cell('accuracy', r.accuracy)} & "
                    f"{cell('pc', r.pc)} & {cell('first_rate', r.first_rate)} & "
                    f"{cell('longer_rate', r.longer_rate)} & {mass} " + r"\\")
            lines.append(r"\midrule")
    lines[-1] = r"\bottomrule"
    lines.append(r"\end{tabular}")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    df = load_results(sorted((RESULTS / "p2").glob("*.jsonl")))
    if df.empty:
        sys.exit("brak wyników w results/p2")
    summ = summarize(df).merge(bpw_table(), on=["model", "quant"], how="left")
    summ.to_csv(OUT / "summary.csv", index=False)
    t = tests(df)
    t.to_csv(OUT / "tests.csv", index=False)
    figure(summ, OUT / "fig_p2")
    latex_table(summ, t, OUT / "table_p2.tex")

    with pd.option_context("display.width", 200, "display.max_columns", None):
        show = t.copy()
        show[["bf16", "value", "diff"]] = show[["bf16", "value", "diff"]].round(3)
        show[["p", "p_holm"]] = show[["p", "p_holm"]].round(4)
        print(show.to_string(index=False))
    print("zapisano", OUT)


if __name__ == "__main__":
    main()
