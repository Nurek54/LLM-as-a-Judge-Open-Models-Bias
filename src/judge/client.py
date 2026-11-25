"""Uruchamianie llama-server i zapytania sędziego.

Używamy natywnych parametrów llama-server w API zgodnym z OpenAI:
  grammar               wymusza odpowiedź "A" albo "B"
  logprobs/top_logprobs rozkład na pierwszej pozycji (liczony z logitów przed gramatyką)
  chat_template_kwargs  np. enable_thinking=false (wymaga --jinja)
"""
from __future__ import annotations

import math
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import httpx

from .config import ModelSpec, llama_bin, llama_cpp_cfg, server_cfg
from .prompts import VERDICT_GRAMMAR


# ------------------------------------------------------------------- serwer
class LlamaServer:
    """Kontekst: uruchamia llama-server dla danego pliku GGUF i zamyka go na końcu."""

    def __init__(self, model: ModelSpec, quant: str, log_dir: Path | None = None):
        self.model, self.quant = model, quant
        self.cfg = server_cfg()
        self.url = f"http://{self.cfg['host']}:{self.cfg['port']}"
        self.log_dir = log_dir
        self.proc: subprocess.Popen | None = None

    def command(self) -> list[str]:
        path = self.model.gguf_path(self.quant)
        if not path.exists():
            raise FileNotFoundError(f"Brak {path}. Uruchom scripts/models.py download/quantize.")
        ngl = self.model.ngl_by_quant.get(self.quant, self.cfg["ngl"])
        return [
            llama_bin("llama-server"), "-m", str(path),
            "--host", str(self.cfg["host"]), "--port", str(self.cfg["port"]),
            "-c", str(self.cfg["ctx"]), "-np", str(self.cfg["parallel"]),
            "-ngl", str(ngl), "--jinja",
            *[str(a) for a in self.cfg.get("extra_args", [])],
        ]

    def __enter__(self) -> "LlamaServer":
        cmd = self.command()
        print("[server]", " ".join(cmd), file=sys.stderr)
        out = None
        if self.log_dir:
            self.log_dir.mkdir(parents=True, exist_ok=True)
            out = open(self.log_dir / f"server_{self.model.key}_{self.quant}.log", "w", encoding="utf-8")
        self.proc = subprocess.Popen(cmd, stdout=out, stderr=subprocess.STDOUT)
        self._wait_ready()
        return self

    def _wait_ready(self, timeout: float = 600) -> None:
        t0 = time.time()
        while time.time() - t0 < timeout:
            if self.proc and self.proc.poll() is not None:
                raise RuntimeError(f"llama-server zakończył się (kod {self.proc.returncode}); patrz log.")
            try:
                if httpx.get(f"{self.url}/health", timeout=2).status_code == 200:
                    return
            except httpx.HTTPError:
                pass
            time.sleep(1)
        raise TimeoutError("llama-server nie wystartował w zadanym czasie")

    def __exit__(self, *exc) -> None:
        if self.proc and self.proc.poll() is None:
            self.proc.terminate()
            try:
                self.proc.wait(timeout=30)
            except subprocess.TimeoutExpired:
                self.proc.kill()


def server_version() -> str:
    try:
        out = subprocess.run([llama_bin("llama-server"), "--version"],
                             capture_output=True, text=True, timeout=30)
        return (out.stdout + out.stderr).strip().splitlines()[0]
    except Exception as e:  # noqa: BLE001
        return f"unknown ({e})"


def check_build() -> str:
    """Ostrzega, gdy build llama.cpp różni się od zapisanego w configs/models.yaml."""
    ver = server_version()
    expected = llama_cpp_cfg().get("build")
    if expected and expected != "bXXXX" and expected not in ver:
        print(f"[UWAGA] build llama.cpp: {ver}, a w configu: {expected}", file=sys.stderr)
    return ver


# ------------------------------------------------------------------- zapytania
@dataclass
class Verdict:
    slot: str | None          # "A" / "B" (None przy błędzie)
    p_a: float | None         # P(A) / (P(A)+P(B)) z logprobs
    format_mass: float | None # P(A)+P(B): ile masy model daje poprawnemu formatowi
    raw: str
    latency_s: float
    error: str | None = None


def _letter_probs(top: list[dict]) -> tuple[float, float]:
    pa = pb = 0.0
    for t in top:
        tok = t.get("token", "").strip()
        p = math.exp(t["logprob"])
        if tok == "A":
            pa += p
        elif tok == "B":
            pb += p
    return pa, pb


# Jedno trwałe połączenie (keep-alive) zamiast nowego gniazda na każdą ocenę. Bez tego Windows
# przy dziesiątkach tysięcy zapytań potrafi wyczerpać porty efemeryczne (WinError 10048).
_HTTP = httpx.Client(timeout=None)


def _post_with_retry(url: str, body: dict, timeout: float, attempts: int = 5) -> httpx.Response:
    """Ponawia wyłącznie błędy połączenia (np. chwilowy brak portu); błędy HTTP zwraca od razu."""
    for i in range(attempts):
        try:
            return _HTTP.post(url, json=body, timeout=timeout)
        except (httpx.ConnectError, httpx.RemoteProtocolError, httpx.ReadError,
                httpx.WriteError, OSError):
            if i == attempts - 1:
                raise
            time.sleep(2 ** i)
    raise RuntimeError("unreachable")


def judge(base_url: str, messages: list[dict], model: ModelSpec, *, temperature: float,
          seed: int, top_logprobs: int, use_grammar: bool = True,
          max_tokens: int = 1, timeout: float = 600) -> Verdict:
    body = {
        "messages": messages,
        "temperature": temperature,
        "seed": seed,
        "max_tokens": max_tokens,
        "logprobs": True,
        "top_logprobs": top_logprobs,
        "chat_template_kwargs": model.chat_template_kwargs,
        "cache_prompt": False,
    }
    if use_grammar:
        body["grammar"] = VERDICT_GRAMMAR
    t0 = time.time()
    try:
        r = _post_with_retry(f"{base_url}/v1/chat/completions", body, timeout)
        r.raise_for_status()
        ch = r.json()["choices"][0]
        text = ch["message"].get("content") or ""
        content = (ch.get("logprobs") or {}).get("content") or []
        p_a = mass = None
        if content:
            pa, pb = _letter_probs(content[0].get("top_logprobs", []))
            mass = pa + pb
            p_a = pa / mass if mass > 0 else None
        slot = text.strip()[:1] if text.strip()[:1] in ("A", "B") else None
        return Verdict(slot, p_a, mass, text, time.time() - t0)
    except Exception as e:  # noqa: BLE001
        return Verdict(None, None, None, "", time.time() - t0, error=repr(e))
