"""Sweep force_workers/chunk_size for a real FC3 finite-displacement run.

By default this script only writes the planned commands. Add ``--execute`` to
run the sweep. Each run gets a fresh timestamped output directory.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any


DEFAULT_WORKERS = [8, 12, 16, 20, 24, 30, 36, 48]
DEFAULT_CHUNKS = [1, 2]


def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark PhonoFlow FC3 force worker counts.")
    parser.add_argument("--execute", action="store_true", help="Run commands. Omit to only write the command plan.")
    parser.add_argument("--python", default=sys.executable, help="Python executable used to run phonoflow.")
    parser.add_argument("--input-path", type=Path, default=Path("examples/Si.vasp"))
    parser.add_argument("--backend", default="calorine")
    parser.add_argument("--model-path", type=Path, default=None)
    parser.add_argument("--outdir", type=Path, default=None)
    parser.add_argument("--workers", type=int, nargs="+", default=DEFAULT_WORKERS)
    parser.add_argument("--chunks", type=int, nargs="+", default=DEFAULT_CHUNKS)
    parser.add_argument("--kappa-mesh", type=int, nargs=3, default=[11, 11, 11])
    parser.add_argument("--temperatures", type=float, nargs="+", default=[300.0])
    parser.add_argument("--method", default="rta", choices=["rta", "lbte"])
    parser.add_argument("--max-fc3-displacements", type=int, default=None)
    parser.add_argument("--force-parallel-backend", default="process", choices=["process", "worker_queue"])
    parser.add_argument("--extra-arg", action="append", default=[], help="Additional raw argument passed to phonoflow.")
    args = parser.parse_args()

    root = args.outdir or Path("work") / "force_worker_sweep" / datetime.now().strftime("%Y%m%d_%H%M%S")
    root.mkdir(parents=True, exist_ok=False)
    commands = _build_commands(args, root)
    plan = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "execute": bool(args.execute),
        "workers": [int(value) for value in args.workers],
        "chunks": [int(value) for value in args.chunks],
        "runs": commands,
    }
    (root / "force_workers_benchmark_plan.json").write_text(json.dumps(plan, indent=2), encoding="utf-8")

    results: list[dict[str, Any]] = []
    if args.execute:
        for item in commands:
            results.append(_run_one(item))
    report = {
        **plan,
        "results": results,
        "note": "Commands are only executed when --execute is provided.",
    }
    (root / "force_workers_benchmark_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    (root / "force_workers_benchmark_report.md").write_text(_markdown_report(report), encoding="utf-8")
    print(json.dumps(report, indent=2))
    print(f"Wrote force worker benchmark report: {root / 'force_workers_benchmark_report.md'}")


def _build_commands(args: argparse.Namespace, root: Path) -> list[dict[str, Any]]:
    commands: list[dict[str, Any]] = []
    for workers in args.workers:
        for chunk in args.chunks:
            run_outdir = root / f"workers_{workers}_chunk_{chunk}"
            command = [
                str(args.python),
                "-m",
                "phonoflow",
                "single",
                "--input-path",
                str(args.input_path),
                "--backend",
                str(args.backend),
                "--outdir",
                str(run_outdir),
                "--compute-kappa",
                "--fc3-method",
                "finite-displacement",
                "--method",
                str(args.method),
                "--kappa-mesh",
                *[str(value) for value in args.kappa_mesh],
                "--temperatures",
                *[str(value) for value in args.temperatures],
                "--force-workers",
                str(workers),
                "--force-parallel-backend",
                str(args.force_parallel_backend),
                "--force-chunk-size",
                str(chunk),
            ]
            if args.model_path is not None:
                command.extend(["--model-path", str(args.model_path)])
            if args.max_fc3_displacements is not None:
                command.extend(["--max-fc3-displacements", str(args.max_fc3_displacements)])
            command.extend(str(value) for value in args.extra_arg)
            commands.append(
                {
                    "workers": int(workers),
                    "chunk_size": int(chunk),
                    "outdir": str(run_outdir),
                    "command": command,
                }
            )
    return commands


def _run_one(item: dict[str, Any]) -> dict[str, Any]:
    outdir = Path(str(item["outdir"]))
    outdir.mkdir(parents=True, exist_ok=False)
    started = time.perf_counter()
    completed = subprocess.run(
        item["command"],
        cwd=Path.cwd(),
        text=True,
        capture_output=True,
        check=False,
    )
    total_time = max(0.0, time.perf_counter() - started)
    (outdir / "benchmark_command.log").write_text(
        completed.stdout + "\n--- STDERR ---\n" + completed.stderr,
        encoding="utf-8",
    )
    metrics = _extract_metrics(outdir)
    return {
        **item,
        "returncode": int(completed.returncode),
        "total_time": total_time,
        **metrics,
    }


def _extract_metrics(outdir: Path) -> dict[str, Any]:
    timing = _read_json(outdir / "timing_breakdown.json")
    result = _read_json(outdir / "result.json")
    fc3_force_wall_time = _first_number(
        timing,
        [
            "fc3_force_wall_time",
            "fc3_force_evaluation_time",
            "fc3_force_evaluation_wall_time",
            "thermal.fc3_force_evaluation_wall_time",
        ],
    )
    n_displacements = _first_number(
        timing,
        [
            "n_fc3_displacements",
            "fc3_n_displacements",
            "thermal.n_fc3_displacements",
        ],
    )
    if n_displacements is None:
        n_displacements = _first_number(result, ["thermal.n_fc3_displacements", "n_fc3_displacements"])
    dps = float(n_displacements / fc3_force_wall_time) if n_displacements and fc3_force_wall_time else None
    return {
        "fc3_force_wall_time": fc3_force_wall_time,
        "displacements_per_second": dps,
        "n_fc3_displacements": n_displacements,
        "timing_breakdown_present": bool(timing),
        "result_json_present": bool(result),
    }


def _read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}


def _first_number(data: dict[str, Any], paths: list[str]) -> float | None:
    for path in paths:
        value: Any = data
        for key in path.split("."):
            if not isinstance(value, dict) or key not in value:
                value = None
                break
            value = value[key]
        if isinstance(value, (int, float)):
            return float(value)
    return None


def _markdown_report(report: dict[str, Any]) -> str:
    lines = [
        "# Force Workers Benchmark Report",
        "",
        f"- Execute: {report['execute']}",
        f"- Runs: {len(report['runs'])}",
        "",
        "| workers | chunk | returncode | FC3 force wall time (s) | displacements/s | total time (s) |",
        "|---:|---:|---:|---:|---:|---:|",
    ]
    for result in report.get("results", []):
        lines.append(
            "| {workers} | {chunk_size} | {returncode} | {fc3} | {dps} | {total} |".format(
                workers=result["workers"],
                chunk_size=result["chunk_size"],
                returncode=result.get("returncode", ""),
                fc3=_format_number(result.get("fc3_force_wall_time")),
                dps=_format_number(result.get("displacements_per_second")),
                total=_format_number(result.get("total_time")),
            )
        )
    if not report.get("results"):
        lines.append("| | | | | | |")
        lines.extend(["", "No commands were executed. Re-run with `--execute` to collect timings."])
    return "\n".join(lines) + "\n"


def _format_number(value: Any) -> str:
    if isinstance(value, (int, float)):
        return f"{float(value):.6f}"
    return ""


if __name__ == "__main__":
    main()
