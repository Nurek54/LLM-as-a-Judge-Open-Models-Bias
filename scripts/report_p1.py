"""Raport P1: position bias a rodzina i rozmiar modelu (wszystkie modele w Q4_K_M).

  python scripts/report_p1.py

Wyniki w results/p1/report/:
  summary.csv     metryki dla model x zbiór + 95% CI (bootstrap po parach)
  tests.csv       H1: mniejszy vs większy model w obrębie rodziny (test permutacyjny,
                  sparowany po parach, bo wszyscy sędziowie oceniają te same pary)
  fig_p1.pdf/png  wiersze = metryki, kolumny = zbiory, modele pogrupowane rodzinami
  table_p1.tex    tabela do raportu
"""
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import matplotlib.ticker  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from judge.config import RESULTS  # noqa: E402
from judge.metrics import load_results, summarize  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
from report_p2 import _pair_scores, holm, sign_flip_test  # noqa: E402

DATASETS = ["mtbench_human", "llmbar"]
DS_LABEL = {"mtbench_human": "MT-Bench (ludzie)", "llmbar": "LLMBar"}
METRICS = [  # (kolumna, etykieta, linia odniesienia)
    ("pc", "Spójność po zamianie (PC)", None),
    ("first_rate", "Odsetek wyborów pozycji A", 0.5),
    ("soft_pos_bias", "Miękki bias pozycji (z P(A))", 0.0),
    ("accuracy", "Trafność", None),
]
CI_METRICS = ["pc", "first_rate", "soft_pos_bias", "accuracy"]
# Rozmiar kodowany kolorem: mniejszy = slot 1 (niebieski, kółko), większy = slot 2 (pomarańczowy, kwadrat)
SIZE_STYLE = {"mniejszy": dict(color="#2a78d6", marker="o"),
              "większy": dict(color="#eb6834", marker="s")}
INK, MUTED, GRID = "#1f1f1e", "#6b6a63", "#e4e3de"
OUT = RESULTS / "p1" / "report"


def pair_level(g: pd.DataFrame) -> pd.DataFrame:
    """Jedna linia na parę (greedy). Każda metryka z CI_METRICS to średnia kolumny po parach."""
    g0 = g[g["repeat"] == 0]
    t = g0.pivot_table(index="pair_id", columns="order",
                       values=["chosen", "p_slot_a", "correct"], aggfunc="first")
    t.columns = [f"{a}_{b}" for a, b in t.columns]
    first = g0.assign(a=(g0["slot"] == "A").astype(float)).groupby("pair_id")["a"].mean()
    return pd.DataFrame({
        "pc": (t["chosen_12"] == t["chosen_21"]).astype(float),
        "first_rate": first,
        "soft_pos_bias": t["p_slot_a_12"] + t["p_slot_a_21"] - 1,
        "accuracy": t[["correct_12", "correct_21"]].astype(float).mean(axis=1),
    })


def bootstrap(g: pd.DataFrame, metrics: list[str], n_boot: int = 2000, seed: int = 0) -> dict:
    """95% CI: losowanie par ze zwracaniem, metryki liczone jako średnie po parach."""
    pl = pair_level(g)[metrics]
    x = pl.to_numpy(dtype=float)
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(x), size=(n_boot, len(x)))
    means = np.nanmean(x[idx], axis=1)  # (n_boot, n_metrics)
    out = {}
    for k, m in enumerate(metrics):
        out[f"{m}_lo"], out[f"{m}_hi"] = np.nanquantile(means[:, k], [0.025, 0.975])
    return out


def size_labels(summ: pd.DataFrame) -> pd.DataFrame:
    """W każdej rodzinie: model o najmniejszym size_b = 'mniejszy', reszta = 'większy'."""
    smallest = summ.groupby("family")["size_b"].transform("min")
    return summ.assign(size_class=np.where(summ["size_b"] == smallest, "mniejszy", "większy"))


def h1_tests(df: pd.DataFrame, summ: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for fam, s in summ.groupby("family"):
        models = s.sort_values("size_b")["model"].unique()
        if len(models) < 2:
            continue
        small, large = models[0], models[-1]
        for ds in DATASETS:
            a = df[(df.model == small) & (df.dataset == ds)]
            b = df[(df.model == large) & (df.dataset == ds)]
            if a.empty or b.empty:
                continue
            for value, name in [("consistent", "pc"), ("first", "first_rate"),
                                ("correct", "accuracy")]:
                x, y = _pair_scores(a, value), _pair_scores(b, value)
                common = x.index.intersection(y.index)
                d = (y[common] - x[common]).to_numpy()
                rows.append({"family": fam, "dataset": ds, "metric": name,
                             "small": small, "large": large, "n_pairs": len(common),
                             "small_value": x[common].mean(), "large_value": y[common].mean(),
                             "diff_large_minus_small": d.mean(), "p": sign_flip_test(d)})
            # |first_rate - 0.5|: czy większy model jest bliżej braku preferencji pozycji
            fa, fb = _pair_scores(a, "first"), _pair_scores(b, "first")
            common = fa.index.intersection(fb.index)
            rows.append({"family": fam, "dataset": ds, "metric": "pb",
                         "small": small, "large": large, "n_pairs": len(common),
                         "small_value": abs(fa[common].mean() - 0.5),
                         "large_value": abs(fb[common].mean() - 0.5),
                         "diff_large_minus_small": abs(fb[common].mean() - 0.5) - abs(fa[common].mean() - 0.5),
                         "p": np.nan})
    t = pd.DataFrame(rows)
    if t.empty:
        return t
    mask = t["p"].notna()
    t.loc[mask, "p_holm"] = t[mask].groupby(["dataset", "metric"])["p"].transform(holm)
    t["sig"] = np.where(t["p_holm"] < 0.05, "*", "")
    return t


def figure(summ: pd.DataFrame, path: Path) -> None:
    fams = list(dict.fromkeys(summ.sort_values(["family", "size_b"])["family"]))
    fig, axes = plt.subplots(len(METRICS), len(DATASETS), figsize=(10, 12), sharey="row")
    comma = matplotlib.ticker.FuncFormatter(lambda v, _: f"{v:g}".replace(".", ","))
    for r, (col, label, ref) in enumerate(METRICS):
        for c, ds in enumerate(DATASETS):
            ax = axes[r, c]
            s = summ[summ.dataset == ds]
            for i, fam in enumerate(fams):
                for j, row in enumerate(s[s.family == fam].sort_values("size_b").itertuples()):
                    st = SIZE_STYLE[row.size_class]
                    x = i + (-0.15 if row.size_class == "mniejszy" else 0.15)
                    lo, hi = getattr(row, f"{col}_lo", np.nan), getattr(row, f"{col}_hi", np.nan)
                    v = getattr(row, col)
                    if not np.isnan(lo):
                        ax.plot([x, x], [lo, hi], color=st["color"], lw=2, zorder=2,
                                solid_capstyle="round")
                    ax.plot(x, v, marker=st["marker"], color=st["color"], ms=8, mec="white",
                            mew=1.5, ls="none", zorder=3,
                            label=row.size_class if (i == 0 and r == 0 and c == 0) else None)
            if ref is not None:
                ax.axhline(ref, color=MUTED, lw=1, ls="--", zorder=1)
            ax.set_xticks(range(len(fams)))
            ax.set_xticklabels([
                f"{f}\n" + " / ".join(f"{v:g}B".replace(".", ",") for v in
                                      sorted(summ[summ.family == f]["size_b"].unique()))
                for f in fams], fontsize=9, color=INK)
            ax.set_xlim(-0.6, len(fams) - 0.4)
            ax.grid(True, axis="y", color=GRID, lw=0.8)
            ax.set_axisbelow(True)
            ax.yaxis.set_major_formatter(comma)
            for sp in ("top", "right"):
                ax.spines[sp].set_visible(False)
            for sp in ("left", "bottom"):
                ax.spines[sp].set_color(MUTED)
            ax.tick_params(colors=MUTED, labelsize=9)
            if r == 0:
                ax.set_title(DS_LABEL[ds], color=INK, fontsize=11)
            if c == 0:
                ax.set_ylabel(label, color=INK, fontsize=10)
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, [f"{l} model w rodzinie" for l in labels], loc="upper center",
               ncol=2, frameon=False, fontsize=10, bbox_to_anchor=(0.5, 1.0))
    fig.text(0.5, 0.005, "pionowe odcinki: 95% CI (bootstrap po parach); Q4_K_M dla wszystkich",
             ha="center", color=MUTED, fontsize=8)
    fig.tight_layout(rect=(0, 0.015, 1, 0.97))
    fig.savefig(path.with_suffix(".pdf"))
    fig.savefig(path.with_suffix(".png"), dpi=200)
    plt.close(fig)


def latex_table(summ: pd.DataFrame, path: Path) -> None:
    f = lambda v: f"{v:.3f}".replace(".", ",")  # noqa: E731
    lines = [r"\begin{tabular}{llrrrrrr}", r"\toprule",
             r"Zbiór & Model & PC & Pozycja A & Miękki bias & Trafność & $\kappa$ & Masa A+B \\",
             r"\midrule"]
    for ds in DATASETS:
        s = summ[summ.dataset == ds].sort_values(["family", "size_b"])
        for r in s.itertuples():
            lines.append(f"{DS_LABEL[ds]} & {r.model} & {f(r.pc)} & {f(r.first_rate)} & "
                         f"{f(r.soft_pos_bias)} & {f(r.accuracy)} & {f(r.kappa)} & "
                         f"{f(r.format_mass)} " + r"\\")
        lines.append(r"\midrule")
    lines[-1] = r"\bottomrule"
    lines.append(r"\end{tabular}")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    df = load_results(sorted((RESULTS / "p1").glob("*.jsonl")))
    if df.empty:
        sys.exit("brak wyników w results/p1")
    summ = summarize(df)
    cis = []
    for _, row in summ.iterrows():
        g = df[(df.model == row.model) & (df.dataset == row.dataset)]
        cis.append(bootstrap(g, CI_METRICS))
    summ = size_labels(pd.concat([summ, pd.DataFrame(cis)], axis=1))
    summ.to_csv(OUT / "summary.csv", index=False)
    t = h1_tests(df, summ)
    t.to_csv(OUT / "tests.csv", index=False)
    figure(summ, OUT / "fig_p1")
    latex_table(summ, OUT / "table_p1.tex")

    with pd.option_context("display.width", 200, "display.max_columns", None):
        cols = ["dataset", "model", "n_pairs", "pc", "first_rate", "soft_pos_bias",
                "accuracy", "kappa", "format_mass"]
        print(summ.sort_values(["dataset", "family", "size_b"])[cols].round(3).to_string(index=False))
        print()
        if not t.empty:
            print(t.round(4).to_string(index=False))
    print("zapisano", OUT)


if __name__ == "__main__":
    main()
