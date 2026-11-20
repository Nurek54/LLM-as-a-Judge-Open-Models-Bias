"""Wczytywanie konfiguracji i ścieżek projektu."""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml
from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parents[2]
CONFIGS = ROOT / "configs"
DATA_RAW = ROOT / "data" / "raw"
DATA_PROCESSED = ROOT / "data" / "processed"
MODELS_DIR = ROOT / "models"
RESULTS = ROOT / "results"

load_dotenv(ROOT / ".env")


def load_yaml(name: str) -> dict[str, Any]:
    with open(CONFIGS / name, encoding="utf-8") as f:
        return yaml.safe_load(f)


@dataclass
class ModelSpec:
    key: str
    family: str
    size_b: float
    original: str
    source: dict[str, Any]
    chat_template_kwargs: dict[str, Any] = field(default_factory=dict)
    assistant_prefill: str | None = None
    ngl_by_quant: dict[str, int] = field(default_factory=dict)
    optional: bool = False

    def gguf_path(self, quant: str) -> Path:
        return MODELS_DIR / self.key / f"{self.key}-{quant}.gguf"

    @property
    def source_dir(self) -> Path:
        return MODELS_DIR / self.key / "source"


def models() -> dict[str, ModelSpec]:
    cfg = load_yaml("models.yaml")
    out = {}
    for key, m in cfg["models"].items():
        out[key] = ModelSpec(
            key=key,
            family=m["family"],
            size_b=m["size_b"],
            original=m["original"],
            source=m["source"],
            chat_template_kwargs=m.get("chat_template_kwargs") or {},
            assistant_prefill=m.get("assistant_prefill"),
            ngl_by_quant=m.get("ngl_by_quant") or {},
            optional=bool(m.get("optional", False)),
        )
    return out


def server_cfg() -> dict[str, Any]:
    return load_yaml("models.yaml")["server"]


def llama_cpp_cfg() -> dict[str, Any]:
    cfg = load_yaml("models.yaml")["llama_cpp"]
    cfg["dir"] = os.getenv("LLAMA_CPP_DIR") or cfg.get("dir")
    return cfg


def llama_bin(name: str) -> str:
    """Pełna ścieżka do binarki llama.cpp (llama-server, llama-quantize...)."""
    d = llama_cpp_cfg()["dir"]
    if not d:
        return name  # liczymy na PATH
    p = Path(d) / name
    if os.name == "nt" and not p.suffix:
        p = p.with_suffix(".exe")
    return str(p)


def experiment(name: str) -> dict[str, Any]:
    cfg = load_yaml("experiments.yaml")
    exp = dict(cfg["defaults"])
    exp.update(cfg[name])
    exp["name"] = name
    return exp
