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


@retry(stop=stop_after_attempt(5), wait=wait_exponential(min=2, max=60))
def translate(client: OpenAI, model: str, text: str) -> str:
    if not text.strip():
        return text
    r = client.chat.completions.create(
        model=model, temperature=0,
        messages=[{"role": "system", "content": SYSTEM}, {"role": "user", "content": text}],
    )
    return r.choices[0].message.content


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int)
    ap.add_argument("--sample", type=int, default=40, help="ile par do ręcznej weryfikacji")
    a = ap.parse_args()

    client = OpenAI(base_url=os.getenv("TRANSLATE_BASE_URL"),
                    api_key=os.getenv("TRANSLATE_API_KEY") or os.getenv("OPENAI_API_KEY") or "x")
    model = os.environ["TRANSLATE_MODEL"]
    out = DATA_PROCESSED / "llmbar_pl.jsonl"
    done = {r["pair_id"] for r in read_jsonl(out)} if out.exists() else set()

    with open(out, "a", encoding="utf-8") as f:
        for p in tqdm(load_pairs("llmbar", a.limit)):
            pid = p["pair_id"].replace("llmbar/", "llmbar_pl/", 1)
            if pid in done:
                continue
            row = dict(p, pair_id=pid, dataset="llmbar_pl", lang="pl",
                       instruction=translate(client, model, p["instruction"]),
                       response_1=translate(client, model, p["response_1"]),
                       response_2=translate(client, model, p["response_2"]),
                       meta={**p["meta"], "translator": model})
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
            f.flush()

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
