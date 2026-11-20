"""Zarządzanie modelami: sprawdzenie repo, pobranie BF16, kwantyzacja, informacje o plikach.

Przykłady:
  python scripts/models.py list
  python scripts/models.py check qwen3.5-9b              # pliki w repo HF
  python scripts/models.py download qwen3.5-9b           # BF16 -> models/qwen3.5-9b/
  python scripts/models.py quantize qwen3.5-9b --quants Q8_0 Q4_K_M
  python scripts/models.py quantize --all --quants Q4_K_M   # wszystkie modele do P1
  python scripts/models.py info                          # rozmiar, bity/wagę, SHA-256 -> configs/models_info.json
"""
from __future__ import annotations

import argparse
import fnmatch
import hashlib
import json
import shutil
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from judge.config import CONFIGS, llama_bin, load_yaml, models  # noqa: E402


def cmd_list(_):
    for k, m in models().items():
        have = [q for q in load_yaml("models.yaml")["quant_levels"] if m.gguf_path(q).exists()]
        opt = " (opcjonalny)" if m.optional else ""
        print(f"{k:18} {m.family:11} {m.size_b:>5}B  pliki: {', '.join(have) or '-'}{opt}")


def cmd_check(a):
    from huggingface_hub import list_repo_files
    for k in a.models:
        m = models()[k]
        files = list_repo_files(m.source["repo"])
        match = [f for f in files if fnmatch.fnmatch(f.lower(), m.source["pattern"].lower())]
        print(f"== {k}: {m.source['repo']}  (wzorzec {m.source['pattern']})")
        print("   pasujące:", match or "BRAK")
        if not match:
            print("   wszystkie .gguf:", [f for f in files if f.endswith(".gguf")])


def _src_precision(m) -> str:
    """Precyzja pliku źródłowego: BF16, a F16 tylko gdy w repo nie ma BF16."""
    return m.source.get("precision", "BF16")


def _bf16_target(m) -> Path:
    return m.gguf_path(_src_precision(m))


def cmd_download(a):
    from huggingface_hub import list_repo_files, hf_hub_download
    for k in a.models:
        m = models()[k]
        files = sorted(f for f in list_repo_files(m.source["repo"])
                       if fnmatch.fnmatch(f.lower(), m.source["pattern"].lower()))
        if not files:
            sys.exit(f"{k}: żaden plik nie pasuje do wzorca; uruchom `check {k}`")
        if len(files) > 1 and not any("00001-of" in f for f in files):
            sys.exit(f"{k}: wzorzec pasuje do kilku plików {files}; zawęź `source.pattern`")
        m.source_dir.mkdir(parents=True, exist_ok=True)
        local = [Path(hf_hub_download(m.source["repo"], f, local_dir=m.source_dir)) for f in files]
        target = _bf16_target(m)
        if len(local) == 1:
            if not target.exists():
                shutil.move(str(local[0]), target)
        else:
            # plik dzielony: llama.cpp wczytuje go po pierwszej części, zostawiamy w source/
            first = next(p for p in local if "00001-of" in p.name)
            print(f"{k}: plik dzielony, BF16 = {first}")
            (m.source_dir / "BF16_SPLIT").write_text(str(first))
        print(f"{k}: OK -> {target if target.exists() else m.source_dir}")


def _bf16_input(m) -> Path:
    if _bf16_target(m).exists():
        return _bf16_target(m)
    marker = m.source_dir / "BF16_SPLIT"
    if marker.exists():
        return Path(marker.read_text().strip())
    sys.exit(f"{m.key}: brak pliku źródłowego, najpierw `download {m.key}`")


def cmd_quantize(a):
    keys = list(models()) if a.all else a.models
    for k in keys:
        m = models()[k]
        if a.all and m.optional:
            continue
        src = _bf16_input(m)
        for q in a.quants:
            if q in ("BF16", "F16"):
                continue
            out = m.gguf_path(q)
            if out.exists() and not a.force:
                print(f"{k} {q}: istnieje, pomijam")
                continue
            cmd = [llama_bin("llama-quantize"), str(src), str(out), q]
            print("[quantize]", " ".join(cmd))
            subprocess.run(cmd, check=True)


def _n_params(path: Path) -> int:
    from gguf import GGUFReader
    r = GGUFReader(str(path))
    return int(sum(int(t.n_elements) for t in r.tensors))


def _sha256(path: Path, chunk: int = 1 << 24) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while b := f.read(chunk):
            h.update(b)
    return h.hexdigest()


def cmd_info(a):
    rows = []
    for k, m in models().items():
        for q in load_yaml("models.yaml")["quant_levels"]:
            p = m.gguf_path(q)
            if not p.exists():
                continue
            size = p.stat().st_size
            n = _n_params(p)
            rows.append({"model": k, "quant": q, "size_gb": round(size / 1e9, 2),
                         "n_params": n, "bpw": round(size * 8 / n, 3),
                         "sha256": _sha256(p) if a.sha else None})
            print(rows[-1])
    out = CONFIGS / "models_info.json"  # w repo: dokumentuje użyte pliki
    out.write_text(json.dumps(rows, indent=2), encoding="utf-8")
    print("zapisano", out)


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("list").set_defaults(f=cmd_list)
    for name, f in [("check", cmd_check), ("download", cmd_download)]:
        p = sub.add_parser(name)
        p.add_argument("models", nargs="+")
        p.set_defaults(f=f)
    p = sub.add_parser("quantize")
    p.add_argument("models", nargs="*")
    p.add_argument("--all", action="store_true")
    p.add_argument("--quants", nargs="+", required=True)
    p.add_argument("--force", action="store_true")
    p.set_defaults(f=cmd_quantize)
    p = sub.add_parser("info")
    p.add_argument("--sha", action="store_true", help="licz SHA-256 (wolne dla dużych plików)")
    p.set_defaults(f=cmd_info)
    a = ap.parse_args()
    a.f(a)


if __name__ == "__main__":
    main()
