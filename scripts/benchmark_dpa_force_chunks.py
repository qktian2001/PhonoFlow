"""Sweep DPA/DeepMD CPU force chunk settings for a real Si FC3 run."""

from __future__ import annotations

import argparse
import csv
import json
import os
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any


DEFAULT_WORKERS = [8, 12, 16, 20, 24, 30]
DEFAULT_CHUNKS = [1, 2, 4, 8]
DEFAULT_TORCH_THREADS = [1, 2]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--execute", action="store_true", help="Run the sweep. Omit to only write the command plan.")
    parser.add_argument("--root", type=Path, default=Path.cwd())
    parser.add_argument("--python", type=Path, default=Path(sys.executable))
    parser.add_argument("--input-path", type=Path, default=Path("examples/Si.vasp"))
    parser.add_argument("--backend", default="dpa4neo")
    parser.add_argument("--model-path", type=Path, required=True)
    parser.add_argument("--outdir", type=Path, default=None)
    parser.add_argument("--workers", type=int, nargs="+", default=DEFAULT_WORKERS)
    parser.add_argument("--chunks", type=int, nargs="+", default=DEFAULT_CHUNKS)
    parser.add_argument("--torch-threads", type=int, nargs="+", default=DEFAULT_TORCH_THREADS)
    parser.add_argument("--max-fc3-displacements", type=int, default=None)
    parser.add_argument("--kappa-mesh", type=int, nargs=3, default=[11, 11, 11])
    parser.add_argument("--temperatures", type=float, nargs="+", default=[300.0])
    parser.add_argument("--method", default="rta", choices=["rta", "lbte"])
    args = parser.parse_args()

    root = args.root.resolve()
    outdir = (args.outdir or root / "work" / f"dpa_force_chunk_sweep_{datetime.now().strftime('%Y%m%d_%H%M%S')}").resolve()
    outdir.mkdir(parents=True, exist_ok=False)
    plan = build_plan(
        root=root,
        python=args.python.resolve(),
        model_path=args.model_path.resolve(),
        outdir=outdir,
        input_path=args.input_path,
        backend=args.backend,
        workers=args.workers,
        chunks=args.chunks,
        torch_threads=args.torch_threads,
        max_fc3_displacements=args.max_fc3_displacements,
        kappa_mesh=args.kappa_mesh,
        temperatures=args.temperatures,
        method=args.method,
    )
    (outdir / "benchmark_plan.json").write_text(json.dumps(plan, indent=2), encoding="utf-8")

    results: list[dict[str, Any]] = []
    if args.execute:
        for idx, run in enumerate(plan["runs"], start=1):
            result = run_one(run, root=root, idx=idx, total=len(plan["runs"]))
            results.append(result)
            (outdir / "partial_results.json").write_text(json.dumps(results, indent=2), encoding="utf-8")

    report = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "execute": bool(args.execute),
        "outdir": str(outdir),
        "model_path": str(args.model_path.resolve()),
        "runs": plan["runs"],
        "results": results,
        "summary": summarize_results(results),
    }
    (outdir / "benchmark_results.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    write_csv(outdir / "benchmark_results.csv", results)
    (outdir / "benchmark_report.md").write_text(markdown_report(report), encoding="utf-8")
    print(json.dumps(report, indent=2))
    print(f"FINAL_REPORT={outdir / 'benchmark_report.md'}")


def build_plan(
    *,
    root: Path,
    python: Path,
    model_path: Path,
    outdir: Path,
    input_path: Path,
    backend: str,
    workers: list[int],
    chunks: list[int],
    torch_threads: list[int],
    max_fc3_displacements: int | None = None,
    kappa_mesh: list[int] | None = None,
    temperatures: list[float] | None = None,
    method: str = "rta",
) -> dict[str, Any]:
    mesh = kappa_mesh or [11, 11, 11]
    temps = temperatures or [300.0]
    runs: list[dict[str, Any]] = []
    for worker_count in workers:
        for chunk_size in chunks:
            for torch_thread_count in torch_threads:
                run_name = f"workers_{worker_count}_chunk_{chunk_size}_torch_{torch_thread_count}"
                run_outdir = outdir / run_name
                command = [
                    str(python),
                    "-m",
                    "phonoflow",
                    "single",
                    "--input-path",
                    str(_resolve_input(root, input_path)),
                    "--backend",
                    str(backend),
                    "--model-path",
                    str(model_path),
                    "--outdir",
                    str(run_outdir),
                    "--compute-kappa",
                    "--fc3-method",
                    "finite-displacement",
                    "--method",
                    str(method),
                    "--temperatures",
                    *[str(value) for value in temps],
                    "--kappa-mesh",
                    *[str(value) for value in mesh],
                    "--deepmd-device",
                    "cpu",
                    "--deepmd-torch-threads",
                    str(torch_thread_count),
                    "--force-workers",
                    str(worker_count),
                    "--force-parallel-backend",
                    "process",
                    "--force-chunk-size",
                    str(chunk_size),
                ]
                if max_fc3_displacements is not None:
                    command.extend(["--max-fc3-displacements", str(max_fc3_displacements)])
                runs.append(
                    {
                        "name": run_name,
                        "workers": int(worker_count),
                        "chunk_size": int(chunk_size),
                        "deepmd_torch_threads": int(torch_thread_count),
                        "outdir": str(run_outdir),
                        "command": command,
                    }
                )
    return {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "root": str(root),
        "model_path": str(model_path),
        "input_path": str(_resolve_input(root, input_path)),
        "runs": runs,
    }


def run_one(run: dict[str, Any], *, root: Path, idx: int, total: int) -> dict[str, Any]:
    run_outdir = Path(str(run["outdir"]))
    run_outdir.mkdir(parents=True, exist_ok=False)
    (run_outdir / "benchmark_command.json").write_text(json.dumps(run, indent=2), encoding="utf-8")
    print(f"[{idx}/{total}] START {run['name']}", flush=True)
    started = time.perf_counter()
    completed = subprocess.run(
        run["command"],
        cwd=root,
        env={**os.environ, "PYTHONPATH": str(root / "src")},
        text=True,
        capture_output=True,
        check=False,
    )
    elapsed = max(0.0, time.perf_counter() - started)
    (run_outdir / "benchmark_stdout.log").write_text(completed.stdout, encoding="utf-8")
    (run_outdir / "benchmark_stderr.log").write_text(completed.stderr, encoding="utf-8")
    result_json = read_json(run_outdir / "result.json")
    timing_json = read_json(run_outdir / "timing_breakdown.json")
    record = {
        **run,
        "returncode": int(completed.returncode),
        "total_wall_time_s": elapsed,
        "fc3_force_wall_time_s": fc3_force_wall_time(result_json, timing_json),
        "fc3_displacements_per_second": fc3_displacements_per_second(result_json),
        "n_fc3_displacements": n_fc3_displacements(result_json),
        "kappa_scalar_300K": extract_kappa(result_json),
    }
    print(
        f"[{idx}/{total}] DONE rc={record['returncode']} total={elapsed:.3f}s "
        f"fc3={record['fc3_force_wall_time_s']} kappa={record['kappa_scalar_300K']}",
        flush=True,
    )
    return record


def summarize_results(results: list[dict[str, Any]]) -> dict[str, Any]:
    ok_rows = [row for row in results if row.get("returncode") == 0]
    fc3_rows = [row for row in ok_rows if isinstance(row.get("fc3_force_wall_time_s"), (int, float))]
    total_rows = [row for row in ok_rows if isinstance(row.get("total_wall_time_s"), (int, float))]
    kappas = [float(row["kappa_scalar_300K"]) for row in ok_rows if isinstance(row.get("kappa_scalar_300K"), (int, float))]
    best_fc3 = min(fc3_rows, key=lambda row: row["fc3_force_wall_time_s"]) if fc3_rows else None
    best_total = min(total_rows, key=lambda row: row["total_wall_time_s"]) if total_rows else None
    return {
        "successful_runs": len(ok_rows),
        "failed_runs": len(results) - len(ok_rows),
        "best_fc3_run": _run_summary(best_fc3),
        "best_total_run": _run_summary(best_total),
        "kappa_min": min(kappas) if kappas else None,
        "kappa_max": max(kappas) if kappas else None,
        "kappa_spread_abs": max(kappas) - min(kappas) if kappas else None,
    }


def read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        return {"_error": str(exc)}


def first(data: dict[str, Any], paths: list[str]) -> Any:
    for path in paths:
        current: Any = data
        ok = True
        for key in path.split("."):
            if isinstance(current, dict) and key in current:
                current = current[key]
            else:
                ok = False
                break
        if ok:
            return current
    return None


def extract_kappa(result_json: dict[str, Any]) -> float | None:
    rows = first(result_json, ["thermal.thermal_conductivity.rows", "thermal_conductivity.rows"])
    if isinstance(rows, list) and rows:
        row = rows[0]
        if isinstance(row, dict):
            for key in ("kappa_total", "kappa_average", "kappa_iso", "kappa"):
                if isinstance(row.get(key), (int, float)):
                    return float(row[key])
    summary = first(result_json, ["thermal.summary", "summary"])
    if isinstance(summary, dict):
        for key in ("kappa_trace_over_3_300K", "kappa_average_300K", "kappa_300K"):
            if isinstance(summary.get(key), (int, float)):
                return float(summary[key])
    for path in (
        "thermal_conductivity.summary.300.kappa_trace_over_3",
        "thermal.summary.300.kappa_trace_over_3",
    ):
        value = first(result_json, [path])
        if isinstance(value, (int, float)):
            return float(value)
    return None


def fc3_force_wall_time(result_json: dict[str, Any], timing_json: dict[str, Any]) -> float | None:
    value = first(
        result_json,
        [
            "thermal_conductivity.timing_breakdown.fc3_wall_time",
            "thermal.force_evaluation.fc3_wall_time",
            "force_evaluation.fc3_wall_time",
            "thermal.fc3_wall_time",
        ],
    )
    if isinstance(value, (int, float)):
        return float(value)
    value = first(timing_json, ["thermal.force_evaluation.fc3_wall_time", "fc3_wall_time"])
    return float(value) if isinstance(value, (int, float)) else None


def fc3_displacements_per_second(result_json: dict[str, Any]) -> float | None:
    value = first(
        result_json,
        [
            "thermal_conductivity.timing_breakdown.fc3_displacements_per_second",
            "thermal.force_evaluation.fc3_displacements_per_second",
            "force_evaluation.fc3_displacements_per_second",
            "thermal.fc3_displacements_per_second",
        ],
    )
    return float(value) if isinstance(value, (int, float)) else None


def n_fc3_displacements(result_json: dict[str, Any]) -> float | None:
    value = first(
        result_json,
        [
            "thermal_conductivity.n_fc3_displacements",
            "thermal.n_fc3_displacements",
            "n_fc3_displacements",
        ],
    )
    return float(value) if isinstance(value, (int, float)) else None


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fields = [
        "name",
        "returncode",
        "workers",
        "chunk_size",
        "deepmd_torch_threads",
        "total_wall_time_s",
        "fc3_force_wall_time_s",
        "fc3_displacements_per_second",
        "n_fc3_displacements",
        "kappa_scalar_300K",
        "outdir",
    ]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field) for field in fields})


def markdown_report(report: dict[str, Any]) -> str:
    lines = [
        "# DPA DeepMD Force Chunk Benchmark",
        "",
        f"- Execute: {report['execute']}",
        f"- Runs: {len(report['runs'])}",
        f"- Model: `{report['model_path']}`",
        "",
        "## Summary",
        "",
        "```json",
        json.dumps(report["summary"], indent=2),
        "```",
        "",
        "## Runs",
        "",
        "| name | rc | workers | chunk | torch threads | total s | FC3 force s | disp/s | kappa |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in report.get("results", []):
        lines.append(
            "| {name} | {returncode} | {workers} | {chunk} | {torch} | {total} | {fc3} | {dps} | {kappa} |".format(
                name=row.get("name", ""),
                returncode=row.get("returncode", ""),
                workers=row.get("workers", ""),
                chunk=row.get("chunk_size", ""),
                torch=row.get("deepmd_torch_threads", ""),
                total=format_number(row.get("total_wall_time_s")),
                fc3=format_number(row.get("fc3_force_wall_time_s")),
                dps=format_number(row.get("fc3_displacements_per_second")),
                kappa=format_number(row.get("kappa_scalar_300K"), digits=10),
            )
        )
    if not report.get("results"):
        lines.append("| | | | | | | | | |")
        lines.extend(["", "No commands were executed. Re-run with `--execute` to collect timings."])
    return "\n".join(lines) + "\n"


def format_number(value: Any, *, digits: int = 6) -> str:
    if isinstance(value, (int, float)):
        return f"{float(value):.{digits}f}"
    return ""


def _resolve_input(root: Path, input_path: Path) -> Path:
    return input_path if input_path.is_absolute() else root / input_path


def _run_summary(row: dict[str, Any] | None) -> dict[str, Any] | None:
    if row is None:
        return None
    return {
        "name": row.get("name"),
        "workers": row.get("workers"),
        "chunk_size": row.get("chunk_size"),
        "deepmd_torch_threads": row.get("deepmd_torch_threads"),
        "total_wall_time_s": row.get("total_wall_time_s"),
        "fc3_force_wall_time_s": row.get("fc3_force_wall_time_s"),
        "kappa_scalar_300K": row.get("kappa_scalar_300K"),
    }


if __name__ == "__main__":
    main()
