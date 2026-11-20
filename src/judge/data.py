"""Pobieranie zbiorów i sprowadzenie ich do wspólnego formatu par.

Wspólny format (jedna linia JSONL = jedna para):
    pair_id      unikalny identyfikator w obrębie zbioru
    dataset      nazwa zbioru (mtbench_human, llmbar, judgebench, evalbiasbench, llmbar_pl)
    subset       podzbiór (np. Natural, Adversarial/GPTInst, mmlu-pro-law, length bias)
    lang         en / pl
    instruction  polecenie / pytanie
    response_1   pierwsza odpowiedź (w kolejności z oryginalnego zbioru)
    response_2   druga odpowiedź
    gold         1 albo 2: która odpowiedź jest lepsza (wg ludzi lub obiektywnie)
    meta         dodatkowe pola (modele-autorzy, liczba głosów...)
"""
from __future__ import annotations

import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Iterable, Iterator

import httpx

from .config import DATA_PROCESSED, DATA_RAW

LLMBAR_BASE = "https://raw.githubusercontent.com/princeton-nlp/LLMBar/main/Dataset/LLMBar"
LLMBAR_SUBSETS = ["Natural", "Adversarial/GPTInst", "Adversarial/GPTOut",
                  "Adversarial/Manual", "Adversarial/Neighbor"]
JUDGEBENCH_URL = ("https://raw.githubusercontent.com/ScalerLab/JudgeBench/main/data/"
                  "dataset=judgebench,response_model=gpt-4o-2024-05-13.jsonl")
EVALBIASBENCH_URL = ("https://raw.githubusercontent.com/ncsoft/offsetbias/master/"
                     "data/evalbiasbench/biasbench.json")


# ----------------------------------------------------------------------------- IO
def write_jsonl(path: Path, rows: Iterable[dict]) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    n = 0
    with open(path, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
            n += 1
    return n


def read_jsonl(path: Path) -> Iterator[dict]:
    with open(path, encoding="utf-8") as f:
        for line in f:
            if line.strip():
                yield json.loads(line)


def load_pairs(dataset: str, limit: int | None = None) -> list[dict]:
    path = DATA_PROCESSED / f"{dataset}.jsonl"
    if not path.exists():
        raise FileNotFoundError(f"{path} nie istnieje. Uruchom: python scripts/prepare_data.py")
    rows = list(read_jsonl(path))
    return rows[:limit] if limit else rows


def _download(url: str, dest: Path) -> Path:
    if dest.exists():
        return dest
    dest.parent.mkdir(parents=True, exist_ok=True)
    r = httpx.get(url, timeout=60, follow_redirects=True)
    r.raise_for_status()
    dest.write_bytes(r.content)
    return dest


# ------------------------------------------------------------------------- LLMBar
def prepare_llmbar() -> int:
    rows = []
    for subset in LLMBAR_SUBSETS:
        raw = _download(f"{LLMBAR_BASE}/{subset}/dataset.json",
                        DATA_RAW / "llmbar" / subset / "dataset.json")
        for i, ex in enumerate(json.loads(raw.read_text(encoding="utf-8"))):
            rows.append({
                "pair_id": f"llmbar/{subset}/{i}",
                "dataset": "llmbar", "subset": subset, "lang": "en",
                "instruction": ex["input"],
                "response_1": ex["output_1"], "response_2": ex["output_2"],
                "gold": int(ex["label"]), "meta": {},
            })
    return write_jsonl(DATA_PROCESSED / "llmbar.jsonl", rows)


# --------------------------------------------------------------------- JudgeBench
def prepare_judgebench() -> int:
    raw = _download(JUDGEBENCH_URL, DATA_RAW / "judgebench" / "gpt-4o.jsonl")
    rows = []
    for ex in (json.loads(l) for l in raw.read_text(encoding="utf-8").splitlines() if l.strip()):
        if ex["label"] not in ("A>B", "B>A"):
            continue
        rows.append({
            "pair_id": f"judgebench/{ex['pair_id']}",
            "dataset": "judgebench", "subset": ex["source"], "lang": "en",
            "instruction": ex["question"],
            "response_1": ex["response_A"], "response_2": ex["response_B"],
            "gold": 1 if ex["label"] == "A>B" else 2,
            "meta": {"response_model": ex.get("response_model")},
        })
    return write_jsonl(DATA_PROCESSED / "judgebench.jsonl", rows)


# ------------------------------------------------------------------ EvalBiasBench
def prepare_evalbiasbench() -> int:
    raw = _download(EVALBIASBENCH_URL, DATA_RAW / "evalbiasbench" / "biasbench.json")
    data = json.loads(raw.read_text(encoding="utf-8"))
    rows = []
    for bias, items in data.items():
        for i, ex in enumerate(items):
            rows.append({
                "pair_id": f"evalbiasbench/{bias}/{i}",
                "dataset": "evalbiasbench", "subset": bias, "lang": "en",
                "instruction": ex["instruction"],
                "response_1": ex["response1"], "response_2": ex["response2"],
                "gold": int(ex["label"]), "meta": {},
            })
    return write_jsonl(DATA_PROCESSED / "evalbiasbench.jsonl", rows)


# ------------------------------------------------------------ MT-Bench (ludzie)
def prepare_mtbench_human() -> int:
    """Tylko tura 1 (w turze 2 kontekst różni się między modelami).

    Ten sam (pytanie, para modeli) oceniało kilku ludzi: bierzemy większość głosów,
    remisy i brak większości odrzucamy.
    """
    from datasets import load_dataset

    ds = load_dataset("lmsys/mt_bench_human_judgments", split="human",
                      cache_dir=str(DATA_RAW / "hf_cache"))
    votes: dict[tuple, Counter] = defaultdict(Counter)
    texts: dict[tuple, dict] = {}
    for ex in ds:
        if ex["turn"] != 1:
            continue
        ma, mb, w = ex["model_a"], ex["model_b"], ex["winner"]
        a_txt, b_txt = ex["conversation_a"][1]["content"], ex["conversation_b"][1]["content"]
        if ma > mb:  # kanoniczna kolejność modeli
            ma, mb, a_txt, b_txt = mb, ma, b_txt, a_txt
            w = {"model_a": "model_b", "model_b": "model_a"}.get(w, w)
        key = (ex["question_id"], ma, mb)
        votes[key][w] += 1
        texts[key] = {"q": ex["conversation_a"][0]["content"], "a": a_txt, "b": b_txt}

    rows = []
    for key, c in sorted(votes.items()):
        (top, n_top), *rest = c.most_common()
        if top == "tie" or (rest and rest[0][1] == n_top):
            continue
        qid, ma, mb = key
        t = texts[key]
        rows.append({
            "pair_id": f"mtbench_human/{qid}/{ma}/{mb}",
            "dataset": "mtbench_human", "subset": str(qid), "lang": "en",
            "instruction": t["q"], "response_1": t["a"], "response_2": t["b"],
            "gold": 1 if top == "model_a" else 2,
            "meta": {"model_1": ma, "model_2": mb, "votes": dict(c)},
        })
    return write_jsonl(DATA_PROCESSED / "mtbench_human.jsonl", rows)


PREPARERS = {
    "llmbar": prepare_llmbar,
    "judgebench": prepare_judgebench,
    "evalbiasbench": prepare_evalbiasbench,
    "mtbench_human": prepare_mtbench_human,
}
