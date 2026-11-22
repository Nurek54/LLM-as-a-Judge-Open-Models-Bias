"""Tłumaczy LLMBar na polski -> data/processed/llmbar_pl.jsonl (do P3).

Tłumacz: dowolny endpoint zgodny z OpenAI (API albo lokalny llama-server), ustawiany w .env:
  TRANSLATE_BASE_URL  (np. https://api.openai.com/v1)
  TRANSLATE_MODEL     (model spoza zestawu sędziów, żeby nie faworyzować żadnej rodziny)
  TRANSLATE_API_KEY   (lub OPENAI_API_KEY)

Tłumaczenie musi zachować wady odpowiedzi (np. niewykonanie polecenia), bo od tego zależy
etykieta gold. Na koniec zapisuje losową próbę do ręcznej weryfikacji (CSV).

  python scripts/translate_llmbar.py --limit 5     # test
  python scripts/translate_llmbar.py               # całość, wznawialne
"""
from __future__ import annotations

import argparse
import json
import os
import random
import sys
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import pandas as pd
from openai import OpenAI
from tenacity import retry, stop_after_attempt, wait_exponential
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from judge.config import DATA_PROCESSED  # noqa: E402
from judge.data import load_pairs, read_jsonl  # noqa: E402

SYSTEM = (
    "You are a professional English-to-Polish translator. Translate the text the user gives "
    "you into natural Polish. Translate faithfully: do NOT correct errors, do NOT complete "
    "missing parts, do NOT improve quality, and keep any deviations from instructions exactly "
    "as they are. Preserve formatting, line breaks, lists, markdown, code and numbers. Do not "
    "translate code, variable names or text that must stay verbatim. Output only the translation."
)


def _request_kwargs() -> dict:
    """Parametry zależne od modelu, ustawiane w .env.

    TRANSLATE_SERVICE_TIER=flex      tańszy tryb OpenAI (wolniejszy, czasem 429 -> ponawiamy)
    TRANSLATE_REASONING_EFFORT=none  dla modeli z rozumowaniem (tłumaczenie go nie potrzebuje)
    TRANSLATE_TEMPERATURE=0          tylko dla modeli bez rozumowania (te z reasoning go nie przyjmują)
    """
    kw = {}
    if v := os.getenv("TRANSLATE_SERVICE_TIER"):
        kw["service_tier"] = v
    if v := os.getenv("TRANSLATE_REASONING_EFFORT"):
        kw["reasoning_effort"] = v
    if v := os.getenv("TRANSLATE_TEMPERATURE"):
        kw["temperature"] = float(v)
    return kw


@retry(stop=stop_after_attempt(8), wait=wait_exponential(min=5, max=300))
def translate(client: OpenAI, model: str, text: str) -> str:
    if not text.strip():
        return text
    r = client.chat.completions.create(
        model=model, **_request_kwargs(),
        messages=[{"role": "system", "content": SYSTEM}, {"role": "user", "content": text}],
    )
    out = r.choices[0].message.content
    if not out or not out.strip():
        raise RuntimeError("pusta odpowiedź tłumacza")
    return out


def translate_pair(client: OpenAI, model: str, p: dict) -> dict:
    return dict(p, pair_id=p["pair_id"].replace("llmbar/", "llmbar_pl/", 1),
                dataset="llmbar_pl", lang="pl",
                instruction=translate(client, model, p["instruction"]),
                response_1=translate(client, model, p["response_1"]),
                response_2=translate(client, model, p["response_2"]),
                meta={**p["meta"], "translator": model, **_request_kwargs()})


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int)
    ap.add_argument("--workers", type=int, default=8, help="równoległe pary (tryb flex jest wolny)")
    ap.add_argument("--sample", type=int, default=40, help="ile par do ręcznej weryfikacji")
    a = ap.parse_args()

    client = OpenAI(base_url=os.getenv("TRANSLATE_BASE_URL"),
                    api_key=os.getenv("TRANSLATE_API_KEY") or os.getenv("OPENAI_API_KEY") or "x",
                    timeout=900)
    model = os.environ["TRANSLATE_MODEL"]
    out = DATA_PROCESSED / "llmbar_pl.jsonl"
    done = {r["pair_id"] for r in read_jsonl(out)} if out.exists() else set()
    todo = [p for p in load_pairs("llmbar", a.limit)
            if p["pair_id"].replace("llmbar/", "llmbar_pl/", 1) not in done]
    print(f"model={model} {_request_kwargs()} | do przetłumaczenia: {len(todo)} par")

    lock = threading.Lock()
    failed = []
    with open(out, "a", encoding="utf-8") as f, ThreadPoolExecutor(a.workers) as ex:
        futs = {ex.submit(translate_pair, client, model, p): p["pair_id"] for p in todo}
        for fut in tqdm(as_completed(futs), total=len(futs)):
            try:
                row = fut.result()
            except Exception as e:  # noqa: BLE001
                failed.append((futs[fut], repr(e)))
                continue
            with lock:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
                f.flush()
    if failed:
        print(f"[UWAGA] {len(failed)} par się nie udało (odpal ponownie, zostaną dorobione):")
        for pid, err in failed[:10]:
            print("  ", pid, err[:200])

    en = {p["pair_id"].split("/", 1)[1]: p for p in load_pairs("llmbar")}
    pl = list(read_jsonl(out))
    random.Random(0).shuffle(pl)
    rows = []
    for p in pl[: a.sample]:
        e = en[p["pair_id"].split("/", 1)[1]]
        rows.append({"pair_id": p["pair_id"], "gold": p["gold"],
                     "instruction_en": e["instruction"], "instruction_pl": p["instruction"],
                     "response_1_en": e["response_1"], "response_1_pl": p["response_1"],
                     "response_2_en": e["response_2"], "response_2_pl": p["response_2"],
                     "ok (1/0)": "", "uwagi": ""})
    path = DATA_PROCESSED / "llmbar_pl_weryfikacja.csv"
    pd.DataFrame(rows).to_csv(path, index=False, encoding="utf-8-sig")
    print("próba do weryfikacji:", path)


if __name__ == "__main__":
    main()
