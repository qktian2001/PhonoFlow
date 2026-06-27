"""Runtime warning filters for optional backend probe noise."""

from __future__ import annotations

import contextlib
import os
import sys
import threading
from collections.abc import Iterator

_FILTER_INSTALLED = False


def is_optional_deepmd_cuda_probe_warning(line: str) -> bool:
    """Return True for known non-fatal DeepMD CUDA runtime probe messages."""

    lowered = line.strip().lower()
    if not lowered:
        return False
    known_prefix = lowered.startswith("deepmd-kit:") or lowered.startswith("implib-gen:")
    known_library = "libcudart" in lowered or "libcusparse" in lowered
    return known_prefix and known_library


def filter_optional_deepmd_cuda_probe_warnings(text: str) -> tuple[str, list[str]]:
    """Filter known DeepMD CUDA probe warnings from stderr text."""

    kept: list[str] = []
    suppressed: list[str] = []
    for line in text.splitlines():
        if is_optional_deepmd_cuda_probe_warning(line):
            suppressed.append(line)
        else:
            kept.append(line)
    filtered = "\n".join(kept)
    if text.endswith("\n") and filtered:
        filtered += "\n"
    return filtered, suppressed


@contextlib.contextmanager
def suppress_optional_deepmd_cuda_probe_warnings(*, enabled: bool = True) -> Iterator[None]:
    """Suppress known DeepMD native CUDA probe noise written directly to stderr."""

    if not enabled or os.environ.get("PHONOFLOW_SHOW_DEEPMD_CUDA_PROBE_WARNINGS"):
        yield
        return
    try:
        sys.stderr.flush()
        original_stderr_fd = os.dup(2)
        read_fd, write_fd = os.pipe()
        os.dup2(write_fd, 2)
        os.close(write_fd)
    except OSError:
        yield
        return

    def drain_stderr() -> None:
        with os.fdopen(read_fd, "r", encoding="utf-8", errors="replace") as stream:
            for line in stream:
                if not is_optional_deepmd_cuda_probe_warning(line):
                    os.write(original_stderr_fd, line.encode("utf-8", errors="replace"))

    reader = threading.Thread(target=drain_stderr, name="phonoflow-stderr-filter", daemon=True)
    reader.start()
    try:
        yield
    finally:
        try:
            sys.stderr.flush()
            os.dup2(original_stderr_fd, 2)
        finally:
            reader.join(timeout=2)
            os.close(original_stderr_fd)


def install_optional_deepmd_cuda_probe_warning_filter(*, enabled: bool = True) -> None:
    """Install a process-lifetime stderr filter for known DeepMD CUDA probe noise."""

    global _FILTER_INSTALLED
    if _FILTER_INSTALLED or not enabled or os.environ.get("PHONOFLOW_SHOW_DEEPMD_CUDA_PROBE_WARNINGS"):
        return
    try:
        sys.stderr.flush()
        original_stderr_fd = os.dup(2)
        read_fd, write_fd = os.pipe()
        os.dup2(write_fd, 2)
        os.close(write_fd)
    except OSError:
        return

    def drain_stderr() -> None:
        with os.fdopen(read_fd, "r", encoding="utf-8", errors="replace") as stream:
            for line in stream:
                if not is_optional_deepmd_cuda_probe_warning(line):
                    os.write(original_stderr_fd, line.encode("utf-8", errors="replace"))

    threading.Thread(target=drain_stderr, name="phonoflow-stderr-filter", daemon=True).start()
    _FILTER_INSTALLED = True
