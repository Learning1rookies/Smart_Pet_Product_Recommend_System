from __future__ import annotations

from pathlib import Path
from typing import Iterable

from pet_agent.utils.config_loader import load_app_config
from pet_agent.utils.path_tool import get_abs_path


class ContentHashStore:
    def __init__(self, path: Path | None = None):
        default_path = load_app_config().get("content_hash_store", "data/runtime/content_md5.txt")
        self.path = path or get_abs_path(default_path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.touch(exist_ok=True)

    def load(self) -> set[str]:
        return {line.strip() for line in self.path.read_text(encoding="utf-8").splitlines() if line.strip()}

    def seen(self, digest: str) -> bool:
        return digest in self.load()

    def append(self, digest: str) -> None:
        if self.seen(digest):
            return
        with self.path.open("a", encoding="utf-8") as handle:
            handle.write(digest + "\n")

    def append_many(self, digests: Iterable[str]) -> None:
        existing = self.load()
        with self.path.open("a", encoding="utf-8") as handle:
            for digest in digests:
                if digest not in existing:
                    handle.write(digest + "\n")
                    existing.add(digest)

