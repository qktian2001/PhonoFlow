"""Tiny profiler for CPU queue operations."""

from __future__ import annotations

import time
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Iterator


@dataclass
class CpuQueueProfile:
    timings: dict[str, float] = field(default_factory=dict)

    @contextmanager
    def measure(self, label: str) -> Iterator[None]:
        started = time.perf_counter()
        try:
            yield
        finally:
            self.timings[label] = self.timings.get(label, 0.0) + max(0.0, time.perf_counter() - started)

    def as_dict(self) -> dict[str, float]:
        return dict(self.timings)
