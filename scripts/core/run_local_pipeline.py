from __future__ import annotations

import subprocess
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]


def run_step(name: str, script_path: Path) -> None:
    print(f"\n[local pipeline] {name}", flush=True)
    completed = subprocess.run(
        [sys.executable, str(script_path)],
        cwd=PROJECT_ROOT,
        text=True,
        encoding="utf-8",
    )
    if completed.returncode != 0:
        raise SystemExit(completed.returncode)


def main() -> None:
    run_step(
        "generate review evidence and product tag scores",
        PROJECT_ROOT / "scripts" / "WorldCloud_Analysis" / "generate_review_tag_evidence.py",
    )
    run_step(
        "import raw and processed data to SQLite",
        PROJECT_ROOT / "scripts" / "core" / "import_to_sqlite.py",
    )
    run_step(
        "build Chroma vector store from raw products and reviews",
        PROJECT_ROOT / "scripts" / "core" / "build_vector_store.py",
    )
    print("\n[local pipeline] done", flush=True)


if __name__ == "__main__":
    main()
