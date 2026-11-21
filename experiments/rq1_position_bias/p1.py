"""P1: position bias a rodzina i rozmiar modelu.

Każda para jest oceniana dwa razy: w kolejności (1, 2) i (2, 1). Sędzia bez position
bias wybiera tę samą odpowiedź w obu kolejnościach.

  python experiments/rq1_position_bias/p1.py

Wyniki: results/rq1_position_bias/<model>.csv oraz summary.csv.
"""
from __future__ import annotations

import csv
import json
import os
import subprocess
import time

import httpx
from dotenv import load_dotenv

MODELS = [
    "qwen3.5-4b", "qwen3.5-9b",
    "gemma-4-e4b", "gemma-4-12b",
    "ministral-3-8b", "ministral-3-14b",
    "bielik-4.5b", "bielik-11b",
]
QUANT = "Q4_K_M"
DATASETS = ["mtbench_human", "llmbar"]
LIMIT = None  # np. 5 do szybkiego testu
RESULTS_DIR = "results/rq1_position_bias"
PORT = 8080

# Ten sam prompt co w src/judge/prompts.py.
SYSTEM = (
    "You are an impartial judge evaluating the quality of two AI assistant responses "
    "to the same user instruction. Choose the response that better follows the "
    "instruction and is more helpful, accurate and harmless. Do not let the order in "
    "which the responses are presented, their length or their formatting influence "
    "your decision. Answer with a single letter: A or B."
)
USER = (
    "[User instruction]\n{instruction}\n\n"
    "[Response A]\n{a}\n[End of Response A]\n\n"
    "[Response B]\n{b}\n[End of Response B]\n\n"
    "Which response is better? Answer with A or B only."
)


def load_pairs(dataset: str) -> list[dict]:
    with open(f"data/processed/{dataset}.jsonl", encoding="utf-8") as f:
        pairs = [json.loads(line) for line in f]
    return pairs[:LIMIT] if LIMIT else pairs


def start_server(model: str) -> subprocess.Popen | None:
    exe = "llama-server.exe" if os.name == "nt" else "llama-server"
    server_bin = os.path.join(os.environ["LLAMA_CPP_DIR"], exe)
    gguf = f"models/{model}/{model}-{QUANT}.gguf"
    if not os.path.exists(gguf):
        print(f"brak {gguf}, pomijam")
        return None

    cmd = [server_bin, "-m", gguf, "--port", str(PORT),
           "-c", "8192", "-np", "1", "-ngl", "999", "--jinja"]
    log = open(f"{RESULTS_DIR}/server_{model}.log", "w", encoding="utf-8")
    proc = subprocess.Popen(cmd, stdout=log, stderr=log)

    # Czekamy, aż model się załaduje (max 5 min).
    for _ in range(300):
        if proc.poll() is not None:
            raise SystemExit(f"llama-server się wyłączył, log: {log.name}")
        try:
            if httpx.get(f"http://127.0.0.1:{PORT}/health", timeout=2).status_code == 200:
                return proc
        except httpx.HTTPError:
            pass
        time.sleep(1)
    proc.kill()
    raise SystemExit("llama-server nie wstał w 5 min")


def ask_judge(instruction: str, a: str, b: str) -> str:
    """Zwraca "A" albo "B"."""
    body = {
        "messages": [
            {"role": "system", "content": SYSTEM},
            {"role": "user", "content": USER.format(instruction=instruction, a=a, b=b)},
        ],
        "temperature": 0,
        "max_tokens": 1,
        "grammar": 'root ::= "A" | "B"',
        "chat_template_kwargs": {"enable_thinking": False},
    }
    r = httpx.post(f"http://127.0.0.1:{PORT}/v1/chat/completions", json=body, timeout=120)
    return r.json()["choices"][0]["message"]["content"].strip()


def judge_pair(pair: dict) -> dict:
    """Ocena w obu kolejnościach; chosen_* to numer wybranej odpowiedzi (1 albo 2)."""
    q, r1, r2 = pair["instruction"], pair["response_1"], pair["response_2"]
    slot_12 = ask_judge(q, r1, r2)   # odpowiedź 1 jako A
    slot_21 = ask_judge(q, r2, r1)   # odpowiedź 1 jako B
    return {
        "slot_12": slot_12,
        "slot_21": slot_21,
        "chosen_12": 1 if slot_12 == "A" else 2,
        "chosen_21": 2 if slot_21 == "A" else 1,
    }


def compute_metrics(rows: list[dict]) -> dict:
    """consistency: % par z tym samym wyborem w obu kolejnościach (100 = brak biasu),
    pick_a: % odpowiedzi "A" (50 = brak preferencji pozycji),
    accuracy: % zgodności z gold."""
    n = len(rows)
    consistent = sum(r["chosen_12"] == r["chosen_21"] for r in rows)
    pick_a = sum((r["slot_12"] == "A") + (r["slot_21"] == "A") for r in rows)
    correct = sum((r["chosen_12"] == r["gold"]) + (r["chosen_21"] == r["gold"]) for r in rows)
    return {
        "n_pairs": n,
        "consistency": round(100 * consistent / n, 1),
        "pick_a": round(100 * pick_a / (2 * n), 1),
        "accuracy": round(100 * correct / (2 * n), 1),
    }


def read_csv(path: str) -> list[dict]:
    with open(path, encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    for r in rows:  # CSV zwraca tekst
        for k in ("gold", "chosen_12", "chosen_21"):
            r[k] = int(r[k])
    return rows


def write_csv(path: str, rows: list[dict]) -> None:
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)


def run_model(model: str, data: dict[str, list[dict]]) -> list[dict] | None:
    proc = start_server(model)
    if proc is None:
        return None
    rows = []
    try:
        for ds, pairs in data.items():
            t0 = time.time()
            for i, pair in enumerate(pairs, 1):
                rows.append({"model": model, "dataset": ds, "pair_id": pair["pair_id"],
                             "gold": pair["gold"], **judge_pair(pair)})
                if i % 50 == 0 or i == len(pairs):
                    print(f"  {ds}: {i}/{len(pairs)} ({time.time() - t0:.0f} s)")
    finally:
        proc.terminate()
        proc.wait()
    return rows


def main():
    load_dotenv()
    os.makedirs(RESULTS_DIR, exist_ok=True)
    data = {ds: load_pairs(ds) for ds in DATASETS}
    summary = []

    for model in MODELS:
        print(f"== {model} {QUANT}")
        path = f"{RESULTS_DIR}/{model}.csv"
        if os.path.exists(path):
            print(f"  {path} istnieje, pomijam (usuń plik, żeby policzyć od nowa)")
            rows = read_csv(path)
        else:
            rows = run_model(model, data)
            if not rows:
                continue
            write_csv(path, rows)

        for ds in DATASETS:
            ds_rows = [r for r in rows if r["dataset"] == ds]
            if ds_rows:
                m = {"model": model, "dataset": ds, **compute_metrics(ds_rows)}
                summary.append(m)
                print(f"  {ds}: {m}")

    if summary:
        write_csv(f"{RESULTS_DIR}/summary.csv", summary)


if __name__ == "__main__":
    main()
