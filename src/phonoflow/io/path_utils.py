"""Path helper functions."""

from __future__ import annotations

from pathlib import Path

from phonoflow.constants import SUPPORTED_STRUCTURE_NAMES, SUPPORTED_STRUCTURE_SUFFIXES


def ensure_dir(path: Path) -> Path:
    """Create a directory if needed and return it."""

    path.mkdir(parents=True, exist_ok=True)
    return path


def is_structure_file(path: Path) -> bool:
    """Return True if a path looks like a supported structure file."""

    return path.is_file() and (
        path.suffix.lower() in SUPPORTED_STRUCTURE_SUFFIXES or path.name.upper() in SUPPORTED_STRUCTURE_NAMES
    )


def find_structure_files(input_dir: Path) -> list[Path]:
    """Find supported structure files below an input directory."""

    return sorted(path for path in input_dir.iterdir() if is_structure_file(path))


def safe_stem(path: Path) -> str:
    """Return a filesystem-friendly stem for an output directory."""

    stem = path.stem if path.suffix else path.name
    return "".join(char if char.isalnum() or char in {"-", "_"} else "_" for char in stem)
