"""Pętla eksperymentu: model x kwantyzacja x zbiór x para x kolejność x powtórzenie.

Wyniki dopisywane są do results/<exp>/<model>__<quant>.jsonl (jedna linia = jedno
wywołanie sędziego). Przerwany przebieg można wznowić: istniejące rekordy są pomijane.
"""
from __future__ import annotations

import json
import platform
import socket
import time
from pathlib import Path

from tqdm import tqdm

from .client import LlamaServer, check_build, judge
from .config import RESULTS, ModelSpec, experiment, models
from .data import load_pairs, read_jsonl
from .prompts import build_messages, slot_to_response


def _key(r: dict) -> tuple:
    return (r["dataset"], r["pair_id"], r["order"], r["repeat"])


def _plan(exp: dict) -> list[tuple[int, float]]:
    """Lista (repeat, temperature): powtórzenie 0 to zawsze pomiar greedy."""
    plan = [(0, 0.0)] if exp["greedy"] else []
    plan += [(i + 1, exp["sampled_temperature"]) for i in range(exp["sampled_repeats"])]
    return plan


def run(exp_name: str, model_key: str, quant: str, *, limit: int | None = None,
        datasets: list[str] | None = None) -> Path:
    exp = experiment(exp_name)
    model: ModelSpec = models()[model_key]
    out_path = RESULTS / exp_name / f"{model_key}__{quant}.jsonl"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    done = {_key(r) for r in read_jsonl(out_path)} if out_path.exists() else set()
    if out_path.exists() and out_path.stat().st_size:
        with open(out_path, "rb+") as fb:  # po przerwaniu: domknij uciętą linię
            fb.seek(-1, 2)
            if fb.read(1) != b"\n":
                fb.write(b"\n")

    build = check_build()
    host = f"{socket.gethostname()} ({platform.system()} {platform.machine()})"
    limit = limit or exp.get("max_pairs_per_dataset")
    plan = _plan(exp)

    with LlamaServer(model, quant, log_dir=RESULTS / exp_name / "logs") as srv, \
            open(out_path, "a", encoding="utf-8") as f:
        for ds in datasets or exp["datasets"]:
            pairs = load_pairs(ds, limit)
            jobs = [(p, o, rep, temp) for p in pairs for o in exp["orders"] for rep, temp in plan
                    if (ds, p["pair_id"], o, rep) not in done]
            for pair, order, rep, temp in tqdm(jobs, desc=f"{model_key}/{quant}/{ds}"):
                lang = pair["lang"] if exp["prompt_lang"] == "auto" else exp["prompt_lang"]
                msgs = build_messages(pair, order, lang, model.assistant_prefill)
                v = judge(srv.url, msgs, model, temperature=temp, seed=exp["seed"] + rep,
                          top_logprobs=exp["top_logprobs"])
                rec = {
                    "exp": exp_name, "model": model_key, "family": model.family,
                    "size_b": model.size_b, "quant": quant,
                    "dataset": ds, "subset": pair["subset"], "lang": lang,
                    "pair_id": pair["pair_id"], "order": order, "repeat": rep,
                    "temperature": temp, "gold": pair["gold"],
                    "len_1": len(pair["response_1"]), "len_2": len(pair["response_2"]),
                    "slot": v.slot,
                    "chosen": slot_to_response(v.slot, order) if v.slot else None,
                    "p_slot_a": v.p_a, "format_mass": v.format_mass,
                    "latency_s": round(v.latency_s, 3), "error": v.error,
                    "llama_build": build, "host": host, "ts": time.time(),
                }
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")
                f.flush()
    return out_path


def smoke_check(model_key: str, quant: str, n: int = 3, dataset: str = "llmbar") -> None:
    """Kilka par BEZ gramatyki: pokazuje surową odpowiedź i top logprobs.

    Służy do sprawdzenia, czy model nie generuje bloku myślenia albo innego prefiksu
    przed literą (wtedy trzeba ustawić assistant_prefill w configs/models.yaml).
    """
    model = models()[model_key]
    pairs = load_pairs(dataset, n)
    with LlamaServer(model, quant) as srv:
        for p in pairs:
            msgs = build_messages(p, "12", p["lang"], model.assistant_prefill)
            free = judge(srv.url, msgs, model, temperature=0.0, seed=0, top_logprobs=10,
                         use_grammar=False, max_tokens=32)
            forced = judge(srv.url, msgs, model, temperature=0.0, seed=0, top_logprobs=10)
            print(f"--- {p['pair_id']} (gold={p['gold']})")
            print(f"bez gramatyki: {free.raw!r}")
            print(f"z gramatyką:   slot={forced.slot} p_A={forced.p_a} "
                  f"masa_A+B={forced.format_mass} err={forced.error}")
