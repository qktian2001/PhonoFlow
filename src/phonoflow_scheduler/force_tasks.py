"""Force-displacement task/result data structures."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np


@dataclass(frozen=True)
class ForceDisplacementSpec:
    """Compact displacement payload reconstructed from a shared atoms template."""

    positions: np.ndarray
    cell: np.ndarray | None = None
    pbc: Any | None = None
    info: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_atoms(cls, atoms: Any) -> "ForceDisplacementSpec":
        """Build a compact payload from an ASE-like atoms object."""

        return cls(
            positions=np.asarray(atoms.get_positions(), dtype=float),
            cell=np.asarray(atoms.get_cell(), dtype=float),
            pbc=np.asarray(atoms.get_pbc(), dtype=bool),
            info=dict(getattr(atoms, "info", {}) or {}),
        )

    def to_atoms(self, base_atoms: Any) -> Any:
        """Reconstruct an ASE-like atoms object from the shared base atoms."""

        if base_atoms is None:
            raise ValueError("ForceDisplacementSpec requires base atoms for reconstruction")
        atoms = base_atoms.copy()
        atoms.set_positions(np.asarray(self.positions, dtype=float))
        if self.cell is not None:
            atoms.set_cell(np.asarray(self.cell, dtype=float), scale_atoms=False)
        if self.pbc is not None:
            atoms.set_pbc(self.pbc)
        atoms.info.update(dict(self.info))
        return atoms


@dataclass
class ForceTask:
    """One force-evaluation task with stable original ordering index."""

    index: int
    payload: Any
    label: str = "force"


@dataclass
class ForceResult:
    """One force-evaluation result."""

    index: int
    forces: np.ndarray
    energy: float | None = None
    audit_record: dict[str, Any] | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class ForceBatchResult:
    """A complete force-evaluation batch."""

    results: list[ForceResult]
    metadata: dict[str, Any]
    warnings: list[str] = field(default_factory=list)

    def ordered_results(self) -> list[ForceResult]:
        """Return results sorted by original task index."""

        return sorted(self.results, key=lambda result: result.index)


def resolve_force_task_payload(payload: Any, base_payload: Any | None = None) -> Any:
    """Return an evaluator-ready payload while preserving old direct payloads."""

    if isinstance(payload, ForceDisplacementSpec):
        return payload.to_atoms(base_payload)
    return payload
