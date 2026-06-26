"""File hashing helpers."""

from __future__ import annotations

import hashlib
from pathlib import Path

from phonoflow.exceptions import ConfigError


def sha256_file(path: Path | None) -> str | None:
    """Return the SHA256 hash of a file, or ``None`` for a missing optional path."""

    if path is None:
        return None
    path = Path(path)
    if not path.exists():
        raise ConfigError(f"Cannot hash missing file: {path}")
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()
