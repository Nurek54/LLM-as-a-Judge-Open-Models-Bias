"""Punkt wejścia: patrz docs/setup.md. Skróty do skryptów:

  python main.py data          -> scripts/prepare_data.py
  python main.py run p1 ...    -> scripts/run.py p1 ...
  python main.py analyze p1    -> scripts/analyze.py p1
  python main.py models list   -> scripts/models.py list
"""
import runpy
import sys
from pathlib import Path

SCRIPTS = {"data": "prepare_data.py", "run": "run.py", "analyze": "analyze.py",
           "models": "models.py", "translate": "translate_llmbar.py"}

if __name__ == "__main__":
    if len(sys.argv) < 2 or sys.argv[1] not in SCRIPTS:
        print(__doc__)
        sys.exit(1)
    script = Path(__file__).parent / "scripts" / SCRIPTS[sys.argv[1]]
    sys.argv = [str(script)] + sys.argv[2:]
    runpy.run_path(str(script), run_name="__main__")
