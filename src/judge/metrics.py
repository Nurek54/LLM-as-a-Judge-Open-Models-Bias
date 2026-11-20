"""Metryki na wynikach sędziów.

Oznaczenia (dla jednej konfiguracji model x kwantyzacja x zbiór x język):
  accuracy        odsetek wywołań, w których sędzia wybrał odpowiedź gold
  kappa           κ Cohena między wyborem sędziego a gold (wywołania z obu kolejności)
  pc              position consistency: odsetek par z tym samym wyborem w kolejności 12 i 21
  first_rate      odsetek wyborów pozycji A;  pb = |first_rate - 0.5|  (Norman i in.)
  soft_pos_bias   średnio p(A|12) + p(A|21) - 1  z logprobs; >0 = faworyzuje pozycję A
  longer_rate     odsetek wyborów dłuższej odpowiedzi
  acc_gold_longer / acc_gold_shorter   trafność, gdy gold jest dłuższa / krótsza
  format_mass     średnia masa P(A)+P(B) na pierwszym tokenie (zgodność z formatem)
  rs              repetition stability: odsetek (para, kolejność) z jednakowym wyborem
                  we wszystkich powtórzeniach z T>0
Przedziały ufności: bootstrap po parach (pary są jednostką losowania, nie wywołania).
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import cohen_kappa_score
from statsmodels.stats.contingency_tables import mcnemar

from .data import read_jsonl

GROUP = ["exp", "model", "family", "size_b", "quant", "dataset", "lang"]


def load_results(paths: list[Path]) -> pd.DataFrame:
    rows = [r for p in paths for r in read_jsonl(p)]
    df = pd.DataFrame(rows)
    if df.empty:
        return df
    df["correct"] = (df["chosen"] == df["gold"]).where(df["chosen"].notna())
    longer = np.where(df["len_1"] >= df["len_2"], 1, 2)
    df["chose_longer"] = (df["chosen"] == longer).where(df["chosen"].notna())
    df["gold_longer"] = df["gold"] == longer
    return df


def _pair_table(g: pd.DataFrame) -> pd.DataFrame:
    """Jedna linia na parę z pomiaru greedy (repeat 0): wybory i p(A) w obu kolejnościach."""
    g0 = g[g["repeat"] == 0]
    t = g0.pivot_table(index="pair_id", columns="order",
                       values=["chosen", "p_slot_a"], aggfunc="first")
    t.columns = [f"{a}_{b}" for a, b in t.columns]
    return t


def _summary(g: pd.DataFrame) -> dict:
    g0 = g[(g["repeat"] == 0) & g["chosen"].notna()]
    pairs = _pair_table(g)
    out = {
        "n_pairs": int(g0["pair_id"].nunique()),
        "n_calls": int(len(g[g["repeat"] == 0])),
        "invalid_rate": float(g[g["repeat"] == 0]["chosen"].isna().mean()),
        "accuracy": float(g0["correct"].mean()),
        "kappa": float(cohen_kappa_score(g0["gold"], g0["chosen"])) if g0["chosen"].nunique() > 1 else np.nan,
        "first_rate": float((g0["slot"] == "A").mean()),
        "longer_rate": float(g0["chose_longer"].mean()),
        "acc_gold_longer": float(g0[g0["gold_longer"]]["correct"].mean()),
        "acc_gold_shorter": float(g0[~g0["gold_longer"]]["correct"].mean()),
        "format_mass": float(g0["format_mass"].mean()),
    }
    out["pb"] = abs(out["first_rate"] - 0.5)
    if {"chosen_12", "chosen_21"} <= set(pairs.columns):
        both = pairs.dropna(subset=["chosen_12", "chosen_21"])
        out["pc"] = float((both["chosen_12"] == both["chosen_21"]).mean())
    if {"p_slot_a_12", "p_slot_a_21"} <= set(pairs.columns):
        sp = (pairs["p_slot_a_12"] + pairs["p_slot_a_21"] - 1).dropna()
        out["soft_pos_bias"] = float(sp.mean())
    rep = g[g["repeat"] > 0]
    if not rep.empty:
        stab = rep.groupby(["pair_id", "order"])["chosen"].nunique()
        out["rs"] = float((stab == 1).mean())
    return out


def summarize(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for key, g in df.groupby(GROUP, dropna=False):
        rows.append({**dict(zip(GROUP, key)), **_summary(g)})
    return pd.DataFrame(rows)


def bootstrap_ci(df: pd.DataFrame, metric: str, n_boot: int = 1000, seed: int = 0,
                 alpha: float = 0.05) -> tuple[float, float]:
    """CI dla metryki z _summary, losowanie par ze zwracaniem."""
    rng = np.random.default_rng(seed)
    ids = df["pair_id"].unique()
    by_pair = {k: v for k, v in df.groupby("pair_id")}
    vals = []
    for _ in range(n_boot):
        sample = rng.choice(ids, size=len(ids), replace=True)
        g = pd.concat([by_pair[i].assign(pair_id=f"{i}#{j}") for j, i in enumerate(sample)])
        v = _summary(g).get(metric)
        if v is not None and not np.isnan(v):
            vals.append(v)
    return float(np.quantile(vals, alpha / 2)), float(np.quantile(vals, 1 - alpha / 2))


def mcnemar_paired(df_a: pd.DataFrame, df_b: pd.DataFrame) -> dict:
    """Test McNemara na poprawności tych samych wywołań (para, kolejność) w dwóch konfiguracjach.

    P2: ten sam model w dwóch kwantyzacjach. P3: te same pary EN vs PL.
    Dla P3 pary łączymy po pair_id bez prefiksu zbioru (llmbar/... vs llmbar_pl/...).
    """
    def prep(d):
        d = d[(d["repeat"] == 0) & d["correct"].notna()].copy()
        d["k"] = d["pair_id"].str.split("/", n=1).str[1] + "|" + d["order"]
        return d.set_index("k")["correct"].astype(bool)
    a, b = prep(df_a), prep(df_b)
    common = a.index.intersection(b.index)
    a, b = a[common], b[common]
    table = [[int((a & b).sum()), int((a & ~b).sum())],
             [int((~a & b).sum()), int((~a & ~b).sum())]]
    res = mcnemar(table, exact=min(table[0][1], table[1][0]) < 25)
    return {"n": len(common), "acc_a": float(a.mean()), "acc_b": float(b.mean()),
            "b01": table[0][1], "b10": table[1][0], "p_value": float(res.pvalue)}
