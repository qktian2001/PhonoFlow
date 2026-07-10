"""Lightweight scheduler timing helpers."""

from __future__ import annotations

import time
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Iterator


@dataclass
class TimingProfile:
    """Named wall-time measurements."""

    sections: dict[str, float] = field(default_factory=dict)

    @contextmanager
    def measure(self, name: str) -> Iterator[None]:
        started = time.perf_counter()
        try:
            yield
        finally:
            self.sections[name] = self.sections.get(name, 0.0) + max(0.0, time.perf_counter() - started)

    def as_metadata(self) -> dict[str, float]:
        return {key: round(value, 6) for key, value in self.sections.items()}
