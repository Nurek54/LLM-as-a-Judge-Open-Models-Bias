"""Pobiera modele do P1 i robi Q4_K_M, trzymając na dysku naraz tylko jeden plik BF16.

Pobieranie i kwantyzacja idą przez scripts/models.py (config: configs/models.yaml),
więc pliki są takie same jak u reszty zespołu. Po kwantyzacji BF16 jest usuwany.
Wystarczy ok. 75 GB wolnego miejsca zamiast ok. 185 GB.

  python experiments/rq1_position_bias/download_models.py

Przerwany przebieg można wznowić: modele z gotowym Q4_K_M są pomijane.
"""
from __future__ import annotations

import glob
import os
import shutil
import subprocess
import sys

MODELS = [
    "qwen3.5-4b", "qwen3.5-9b",
    "gemma-4-e4b", "gemma-4-12b",
    "ministral-3-8b", "ministral-3-14b",
    "bielik-4.5b", "bielik-11b",
]
QUANT = "Q4_K_M"
DELETE_SOURCE = True  # False = zostaw BF16 (potrzeba ok. 185 GB)


def run(cmd: list[str]) -> None:
    print(">", " ".join(cmd))
    subprocess.run(cmd, check=True)


def delete_source(model: str) -> None:
    """Usuwa plik źródłowy (BF16/F16, także dzielony w source/), zostawia Q4."""
    files = [f"models/{model}/{model}-BF16.gguf", f"models/{model}/{model}-F16.gguf"]
    files += glob.glob(f"models/{model}/source/*.gguf")
    for f in files:
        if os.path.exists(f):
            size = os.path.getsize(f) / 1e9
            os.remove(f)
            print(f"usunięto {f} ({size:.1f} GB)")


def main():
    for model in MODELS:
        print(f"== {model}")
        q4 = f"models/{model}/{model}-{QUANT}.gguf"
        if os.path.exists(q4):
            print(f"{q4} istnieje, pomijam")
            continue

        print(f"wolne miejsce: {shutil.disk_usage('.').free / 1e9:.0f} GB")
        run([sys.executable, "scripts/models.py", "download", model])
        run([sys.executable, "scripts/models.py", "quantize", model, "--quants", QUANT])

        if not os.path.exists(q4):
            raise SystemExit(f"brak {q4} po kwantyzacji, BF16 zostaje")
        print(f"gotowe: {q4} ({os.path.getsize(q4) / 1e9:.1f} GB)")
        if DELETE_SOURCE:
            delete_source(model)

    print("\nTest modeli:")
    for model in MODELS:
        print(f"  python scripts/run.py check {model} {QUANT}")


if __name__ == "__main__":
    main()
