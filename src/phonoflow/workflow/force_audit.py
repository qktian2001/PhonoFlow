"""Force-audit helpers for finite-displacement workflows."""

from __future__ import annotations

import csv
import hashlib
from pathlib import Path
from typing import Any

import numpy as np

from phonoflow.workflow.structure_provenance import cell_hash, positions_hash, structure_hash


def build_force_audit_record(index: int, atoms: Any, *, energy: float | None, forces: np.ndarray) -> dict[str, Any]:
    """Build one CSV-friendly force audit record."""

    force_array = np.asarray(forces, dtype=float)
    return {
        "index": int(index),
        "natoms": int(len(atoms)),
        "energy": float(energy) if energy is not None else None,
        "force_max_abs": float(np.max(np.abs(force_array))) if force_array.size else 0.0,
        "force_mean_abs": float(np.mean(np.abs(force_array))) if force_array.size else 0.0,
        "force_norm": float(np.linalg.norm(force_array.ravel())) if force_array.size else 0.0,
        "force_sha256": sha256_array(force_array),
        "structure_hash": structure_hash(atoms),
        "cell_hash": cell_hash(atoms),
        "positions_hash": positions_hash(atoms),
    }


def write_force_audit_files(
    outdir: Path,
    label: str,
    records: list[dict[str, Any]],
    raw_forces: np.ndarray,
) -> dict[str, str]:
    """Write stats CSV, hash CSV, and raw force array NPZ for one displacement family."""

    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    normalized = label.lower()
    stats_name = f"fd_{normalized}_forces_stats.csv"
    hashes_name = f"fd_{normalized}_force_hashes.csv"
    raw_name = f"fd_{normalized}_forces_raw.npz"
    stats_fields = [
        "index",
        "natoms",
        "energy",
        "force_max_abs",
        "force_mean_abs",
        "force_norm",
        "force_sha256",
        "structure_hash",
        "cell_hash",
        "positions_hash",
    ]
    hash_fields = ["index", "force_sha256", "structure_hash", "cell_hash", "positions_hash"]
    _write_csv(outdir / stats_name, stats_fields, records)
    _write_csv(outdir / hashes_name, hash_fields, records)
    np.savez_compressed(outdir / raw_name, forces=np.asarray(raw_forces, dtype=float))
    return {
        f"fd_{normalized}_forces_stats_csv": stats_name,
        f"fd_{normalized}_force_hashes_csv": hashes_name,
        f"fd_{normalized}_forces_raw_npz": raw_name,
    }


def sha256_array(array: np.ndarray) -> str:
    contiguous = np.ascontiguousarray(np.asarray(array))
    digest = hashlib.sha256()
    digest.update(str(contiguous.dtype).encode("utf-8"))
    digest.update(str(contiguous.shape).encode("utf-8"))
    digest.update(contiguous.tobytes())
    return digest.hexdigest()


def _write_csv(path: Path, fields: list[str], records: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for record in records:
            writer.writerow({field: record.get(field) for field in fields})
