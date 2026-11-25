"""Raport P3: język ocenianych odpowiedzi (LLMBar EN vs LLMBar-PL), 4 sędziów w Q4_K_M.

  python scripts/report_p3.py
  python scripts/report_p3.py --exclude-lang-dependent   # bez par zależnych od języka

Wyniki w results/p3/report/:
  summary.csv     metryki dla model x język + 95% CI (bootstrap po parach)
  tests.csv       PL vs EN dla każdego modelu (test permutacyjny sparowany po parach, Holm)
                  + H3: czy luka PL-EN Bielika jest mniejsza niż u pozostałych (diff-in-diff)
  agreement.csv   odsetek identycznych werdyktów EN vs PL (ten sam model, ta sama para i kolejność)
  fig_p3.pdf/png  metryki EN vs PL dla każdego modelu
  table_p3.tex    tabela do raportu

Pary EN i PL łączymy po identyfikatorze bez prefiksu zbioru (llmbar/X == llmbar_pl/X).

Test odporności: --exclude-lang-dependent pomija pary, których zadanie zależy od angielskiej
formy słów (rymy, synonimy konkretnego słowa, kolejność alfabetyczna, liczba liter, wielkie
litery). Takich par nie da się wiernie przetłumaczyć; wyniki trafiają do report_robust/.
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
sys.path.insert(0, str(Path(__file__).resolve().parent))

from judge.config import RESULTS  # noqa: E402
from judge.metrics import load_results, summarize  # noqa: E402
from report_p1 import pair_level  # noqa: E402
from report_p2 import holm, sign_flip_test  # noqa: E402

PL_SPECIALIST = "bielik-11b"
METRICS = [  # (kolumna, etykieta, linia odniesienia)
    ("accuracy", "Trafność", None),
    ("pc", "Spójność po zamianie (PC)", None),
    ("first_rate", "Odsetek wyborów pozycji A", 0.5),
    ("format_mass", "Masa P(A)+P(B)", None),
]
TESTED = ["accuracy", "pc", "first_rate", "longer_rate"]
LANG_STYLE = {"en": dict(color="#2a78d6", marker="o", label="EN (LLMBar)"),
              "pl": dict(color="#eb6834", marker="s", label="PL (LLMBar-PL)")}
INK, MUTED, GRID = "#1f1f1e", "#6b6a63", "#e4e3de"
OUT = RESULTS / "p3" / "report"
LANG_DEPENDENT = {  # klucz pary bez prefiksu zbioru
    "Natural/20",              # rymy do "Moon"
    "Natural/35",              # synonim 'deceptive'
    "Natural/36",              # synonimy 'sad'
    "Natural/67",              # antonim "cogent"
    "Natural/75",              # synonim słowa w nawiasie
    "Adversarial/GPTInst/28",  # sortowanie alfabetyczne nazw owoców
    "Adversarial/Manual/25",   # błędy gramatyczne w angielskich zdaniach (tłumacz je poprawił)
    "Adversarial/Manual/39",   # Wielkie Litery Na Początku Słów
    "Adversarial/Neighbor/55", # rymowany dwuwiersz
    "Adversarial/Neighbor/133",  # słowa 8-literowe
}


def with_key(df: pd.DataFrame) -> pd.DataFrame:
    return df.assign(key=df["pair_id"].str.split("/", n=1).str[1])


def pair_table(df: pd.DataFrame, model: str, lang: str) -> pd.DataFrame:
    g = df[(df.model == model) & (df.lang == lang)]
    pl = pair_level(g)
    pl.index = pl.index.str.split("/", n=1).str[1]
    fm = g[g.repeat == 0].groupby("pair_id")["format_mass"].mean()
    fm.index = fm.index.str.split("/", n=1).str[1]
    return pl.assign(format_mass=fm)


def bootstrap_ci(pl: pd.DataFrame, cols: list[str], n_boot: int = 2000, seed: int = 0) -> dict:
    x = pl[cols].to_numpy(dtype=float)
    idx = np.random.default_rng(seed).integers(0, len(x), size=(n_boot, len(x)))
    means = np.nanmean(x[idx], axis=1)
    out = {}
    for k, c in enumerate(cols):
        out[f"{c}_lo"], out[f"{c}_hi"] = np.nanquantile(means[:, k], [0.025, 0.975])
    return out


def main():
    global OUT
    robust = "--exclude-lang-dependent" in sys.argv
    if robust:
        OUT = RESULTS / "p3" / "report_robust"
    OUT.mkdir(parents=True, exist_ok=True)
    df = load_results(sorted((RESULTS / "p3").glob("*.jsonl")))
    if df.empty:
        sys.exit("brak wyników w results/p3")
    if robust:
        key = df["pair_id"].str.split("/", n=1).str[1]
        df = df[~key.isin(LANG_DEPENDENT)].reset_index(drop=True)
        print(f"test odporności: pominięto {len(LANG_DEPENDENT)} par zależnych od języka")
    models = sorted(df["model"].unique(), key=lambda m: (m == PL_SPECIALIST, m))

    # ---------------------------------------------------------------- podsumowanie
    summ = summarize(df)
    cis = []
    for _, r in summ.iterrows():
        cis.append(bootstrap_ci(pair_table(df, r.model, r.lang), [m for m, *_ in METRICS]))
    summ = pd.concat([summ, pd.DataFrame(cis)], axis=1)
    summ.to_csv(OUT / "summary.csv", index=False)

    # ------------------------------------------------------- testy PL vs EN + H3
    tabs = {(m, l): pair_table(df, m, l) for m in models for l in ("en", "pl")}
    rows, gaps = [], {}
    for m in models:
        en, pl = tabs[(m, "en")], tabs[(m, "pl")]
        common = en.index.intersection(pl.index)
        for metric in TESTED:
            d = (pl.loc[common, metric] - en.loc[common, metric]).to_numpy()
            gaps[(m, metric)] = pd.Series(d, index=common)
            rows.append({"test": "PL-EN", "model": m, "metric": metric, "n_pairs": len(common),
                         "en": en.loc[common, metric].mean(), "pl": pl.loc[common, metric].mean(),
                         "diff": np.nanmean(d), "p": sign_flip_test(d)})
    t = pd.DataFrame(rows)
    t["p_holm"] = t.groupby("metric")["p"].transform(holm)

    # H3: luka Bielika vs luka każdego innego modelu (różnica różnic, sparowana po parach)
    h3 = []
    if PL_SPECIALIST in models:
        for other in [m for m in models if m != PL_SPECIALIST]:
            for metric in TESTED:
                gb, go = gaps[(PL_SPECIALIST, metric)], gaps[(other, metric)]
                common = gb.index.intersection(go.index)
                dd = (gb[common] - go[common]).to_numpy()
                h3.append({"test": "H3 luka(bielik)-luka(inny)", "model": f"{PL_SPECIALIST} vs {other}",
                           "metric": metric, "n_pairs": len(common), "en": np.nan, "pl": np.nan,
                           "diff": np.nanmean(dd), "p": sign_flip_test(dd)})
        h3 = pd.DataFrame(h3)
        h3["p_holm"] = h3.groupby("metric")["p"].transform(holm)
        t = pd.concat([t, h3], ignore_index=True)
    t["sig"] = np.where(t["p_holm"] < 0.05, "*", "")
    t.to_csv(OUT / "tests.csv", index=False)

    # ------------------------------------------- zgodność werdyktów EN vs PL
    g0 = with_key(df[df.repeat == 0])
    agr = []
    for m in models:
        a = g0[(g0.model == m) & (g0.lang == "en")].set_index(["key", "order"])["chosen"]
        b = g0[(g0.model == m) & (g0.lang == "pl")].set_index(["key", "order"])["chosen"]
        c = a.index.intersection(b.index)
        agr.append({"model": m, "n_calls": len(c), "same_verdict": float((a[c] == b[c]).mean())})
    agr = pd.DataFrame(agr)
    agr.to_csv(OUT / "agreement.csv", index=False)

    figure(summ, models, OUT / "fig_p3")
    latex_table(summ, t, models, OUT / "table_p3.tex")

    with pd.option_context("display.width", 200, "display.max_columns", None):
        cols = ["model", "lang", "n_pairs", "accuracy", "kappa", "pc", "first_rate",
                "soft_pos_bias", "longer_rate", "format_mass"]
        print(summ.sort_values(["model", "lang"])[cols].round(3).to_string(index=False))
        print()
        print(agr.round(3).to_string(index=False))
        print()
        show = t.copy()
        show[["en", "pl", "diff"]] = show[["en", "pl", "diff"]].round(3)
        show[["p", "p_holm"]] = show[["p", "p_holm"]].round(4)
        print(show.to_string(index=False))
    print("zapisano", OUT)


def figure(summ: pd.DataFrame, models: list[str], path: Path) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(11, 8))
    comma = matplotlib.ticker.FuncFormatter(lambda v, _: f"{v:g}".replace(".", ","))
    for ax, (col, label, ref) in zip(axes.flat, METRICS):
        for i, m in enumerate(models):
            for lang, dx in (("en", -0.15), ("pl", 0.15)):
                r = summ[(summ.model == m) & (summ.lang == lang)]
                if r.empty:
                    continue
                r = r.iloc[0]
                st = LANG_STYLE[lang]
                lo, hi = r.get(f"{col}_lo", np.nan), r.get(f"{col}_hi", np.nan)
                if not np.isnan(lo):
                    ax.plot([i + dx] * 2, [lo, hi], color=st["color"], lw=2, zorder=2)
                ax.plot(i + dx, r[col], marker=st["marker"], color=st["color"], ms=8, mec="white",
                        mew=1.5, ls="none", zorder=3,
                        label=st["label"] if (i == 0 and ax is axes.flat[0]) else None)
        if ref is not None:
            ax.axhline(ref, color=MUTED, lw=1, ls="--", zorder=1)
        ax.set_xticks(range(len(models)))
        ax.set_xticklabels(models, fontsize=9, color=INK)
        ax.set_xlim(-0.6, len(models) - 0.4)
        ax.set_title(label, color=INK, fontsize=10)
        ax.grid(True, axis="y", color=GRID, lw=0.8)
        ax.set_axisbelow(True)
        ax.yaxis.set_major_formatter(comma)
        for sp in ("top", "right"):
            ax.spines[sp].set_visible(False)
        for sp in ("left", "bottom"):
            ax.spines[sp].set_color(MUTED)
        ax.tick_params(colors=MUTED, labelsize=9)
    handles, labels = axes.flat[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=2, frameon=False, fontsize=10,
               bbox_to_anchor=(0.5, 1.0))
    fig.text(0.5, 0.005, "pionowe odcinki: 95% CI (bootstrap po parach); wszystkie modele Q4_K_M",
             ha="center", color=MUTED, fontsize=8)
    fig.tight_layout(rect=(0, 0.02, 1, 0.95))
    fig.savefig(path.with_suffix(".pdf"))
    fig.savefig(path.with_suffix(".png"), dpi=200)
    plt.close(fig)


def latex_table(summ: pd.DataFrame, t: pd.DataFrame, models: list[str], path: Path) -> None:
    f = lambda v: f"{v:.3f}".replace(".", ",")  # noqa: E731
    sig = {(r.model, r.metric): r.sig for r in t[t.test == "PL-EN"].itertuples()}
    lines = [r"\begin{tabular}{llrrrrrr}", r"\toprule",
             r"Model & Język & Trafność & $\kappa$ & PC & Pozycja A & Dłuższa & Masa A+B \\",
             r"\midrule"]
    for m in models:
        for lang in ("en", "pl"):
            r = summ[(summ.model == m) & (summ.lang == lang)]
            if r.empty:
                continue
            r = r.iloc[0]
            star = (lambda k: sig.get((m, k), "")) if lang == "pl" else (lambda k: "")
            lines.append(f"{m} & {lang.upper()} & {f(r.accuracy)}{star('accuracy')} & {f(r.kappa)} & "
                         f"{f(r.pc)}{star('pc')} & {f(r.first_rate)}{star('first_rate')} & "
                         f"{f(r.longer_rate)}{star('longer_rate')} & "
                         f"{f(r.format_mass)} " + r"\\")
        lines.append(r"\midrule")
    lines[-1] = r"\bottomrule"
    lines.append(r"\end{tabular}")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
