from __future__ import annotations

from pathlib import Path


def display_executable(path: Path) -> str:
    return path.name or str(path)
