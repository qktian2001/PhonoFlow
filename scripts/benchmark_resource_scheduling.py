"""Benchmark PhonoFlow force-loop resource scheduling.

This script creates a new timestamped output directory for every run and never
removes or rewrites existing benchmark outputs.
"""

from __future__ import annotations

import argparse
import json
import os
from datetime import datetime
from pathlib import Path
from typing import Any

from ase import Atoms

from phonoflow.config import WorkflowConfig
from phonoflow.workflow.force_eval import evaluate_forces_for_displacements


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n-displacements", type=int, default=16)
    parser.add_argument("--atoms", type=int, default=4)
    parser.add_argument("--sleep-seconds", type=float, default=0.05)
    parser.add_argument("--workers", type=int, default=min(4, os.cpu_count() or 1))
    parser.add_argument("--output-root", type=Path, default=Path("work") / "cpu_scheduling_benchmarks")
    args = parser.parse_args()

    if args.n_displacements < 1:
        raise SystemExit("--n-displacements must be >= 1")
    if args.atoms < 1:
        raise SystemExit("--atoms must be >= 1")
    if args.workers < 1:
        raise SystemExit("--workers must be >= 1")
    if args.sleep_seconds < 0:
        raise SystemExit("--sleep-seconds must be >= 0")

    outdir = _new_timestamp_dir(args.output_root)
    atoms_list = [_atoms(args.atoms, args.sleep_seconds) for _ in range(args.n_displacements)]

    serial_config = WorkflowConfig(
        backend="dummy",
        force_workers=1,
        force_parallel_backend="serial",
        deepmd_device="cpu",
    )
    parallel_config = WorkflowConfig(
        backend="dummy",
        force_workers=args.workers,
        force_parallel_backend="process",
        deepmd_device="cpu",
    )

    serial = evaluate_forces_for_displacements(
        atoms_list,
        backend_name="dummy",
        config=serial_config,
        force_workers=1,
        force_parallel_backend="serial",
        deepmd_device="cpu",
        audit_label="benchmark_serial",
    )
    parallel = evaluate_forces_for_displacements(
        atoms_list,
        backend_name="dummy",
        model_path=None,
        config=parallel_config,
        force_workers=args.workers,
        force_parallel_backend="process",
        deepmd_device="cpu",
        audit_label="benchmark_parallel",
    )

    serial_seconds = float(serial.metadata["wall_time"])
    parallel_seconds = float(parallel.metadata["wall_time"])
    speedup = serial_seconds / parallel_seconds if parallel_seconds > 0 else None
    payload: dict[str, Any] = {
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "parameters": {
            "n_displacements": args.n_displacements,
            "atoms": args.atoms,
            "sleep_seconds": args.sleep_seconds,
            "workers": args.workers,
            "os_cpu_count": os.cpu_count(),
        },
        "serial": serial.metadata,
        "parallel": parallel.metadata,
        "speedup": speedup,
        "note": (
            "Synthetic dummy benchmark for force-loop scheduling overhead. "
            "Use a real model benchmark for final production throughput decisions."
        ),
    }
    result_path = outdir / "benchmark_resource_scheduling.json"
    result_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    summary_path = outdir / "summary.txt"
    summary_path.write_text(
        "\n".join(
            [
                "PhonoFlow CPU resource scheduling benchmark",
                f"serial_seconds: {serial_seconds:.6f}",
                f"parallel_seconds: {parallel_seconds:.6f}",
                f"speedup: {speedup:.3f}x" if speedup is not None else "speedup: unavailable",
                f"workers: {args.workers}",
                f"result_json: {result_path}",
                "",
            ]
        ),
        encoding="utf-8",
    )
    print(summary_path.read_text(encoding="utf-8"))


def _atoms(n_atoms: int, sleep_seconds: float) -> Atoms:
    symbols = "H" * n_atoms
    positions = [(float(index), 0.0, 0.0) for index in range(n_atoms)]
    atoms = Atoms(symbols=symbols, positions=positions, cell=[20.0, 20.0, 20.0], pbc=True)
    atoms.info["phonoflow_dummy_force_sleep_seconds"] = sleep_seconds
    return atoms


def _new_timestamp_dir(root: Path) -> Path:
    root.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    candidate = root / stamp
    suffix = 1
    while candidate.exists():
        candidate = root / f"{stamp}_{suffix}"
        suffix += 1
    candidate.mkdir(parents=True, exist_ok=False)
    return candidate


if __name__ == "__main__":
    main()
