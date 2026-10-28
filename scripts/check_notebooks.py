"""Cheap notebook sanity check (no Jupyter needed): valid JSON, nbformat 4, code cells parse.

Run: python scripts/check_notebooks.py
"""
from __future__ import annotations

import ast
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MAGIC = re.compile(r"^\s*[%!](?=[A-Za-z%!])")  # %matplotlib inline, !pip ... (not "% clf.score")


def check(path: Path) -> list[str]:
    try:
        nb = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        return [f"{path.name}: unreadable JSON ({exc})"]
    problems = []
    if nb.get("nbformat") != 4:
        problems.append(f"{path.name}: nbformat is {nb.get('nbformat')!r}, expected 4")
    for i, cell in enumerate(nb.get("cells", [])):
        if cell.get("cell_type") != "code":
            continue
        source = cell["source"] if isinstance(cell["source"], str) else "".join(cell["source"])
        lines = ["pass" if MAGIC.match(line) else line for line in source.splitlines()]
        try:
            ast.parse("\n".join(lines))
        except SyntaxError as exc:
            problems.append(f"{path.name}: cell {i}: {exc.msg} (line {exc.lineno})")
    return problems


def main() -> int:
    paths = sorted((ROOT / "notebooks").glob("*.ipynb")) + [ROOT / "check_env.ipynb"]
    problems = [p for path in paths for p in check(path)]
    for p in problems:
        print("PROBLEM:", p)
    print(f"Checked {len(paths)} notebooks, {len(problems)} problem(s).")
    return 1 if problems else 0


if __name__ == "__main__":
    sys.exit(main())
