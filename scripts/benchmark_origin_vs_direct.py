"""Compare current direct force loop with old-version-style origin loop."""

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


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--root", type=Path, default=Path.cwd())
    parser.add_argument("--python", type=Path, default=Path(sys.executable))
    parser.add_argument("--outdir", type=Path, default=None)
    parser.add_argument("--input-path", type=Path, default=Path("examples/Si.vasp"))
    parser.add_argument("--nep-model-path", type=Path, required=True)
    parser.add_argument("--dpa-model-path", type=Path, required=True)
    args = parser.parse_args()

    root = args.root.resolve()
    outdir = (args.outdir or root / "work" / f"origin_vs_direct_{datetime.now().strftime('%Y%m%d_%H%M%S')}").resolve()
    outdir.mkdir(parents=True, exist_ok=False)
    plan = build_plan(
        root=root,
        python=args.python.resolve(),
        outdir=outdir,
        input_path=args.input_path,
        nep_model_path=args.nep_model_path.resolve(),
        dpa_model_path=args.dpa_model_path.resolve(),
    )
    (outdir / "benchmark_plan.json").write_text(json.dumps(plan, indent=2), encoding="utf-8")
    results: list[dict[str, Any]] = []
    if args.execute:
        for idx, run in enumerate(plan["runs"], start=1):
            results.append(run_one(run, root=root, idx=idx, total=len(plan["runs"])))
            (outdir / "partial_results.json").write_text(json.dumps(results, indent=2), encoding="utf-8")
    report = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "execute": bool(args.execute),
        "outdir": str(outdir),
        "results": results,
        "summary": summarize(results),
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
    outdir: Path,
    input_path: Path,
    nep_model_path: Path,
    dpa_model_path: Path,
) -> dict[str, Any]:
    runs: list[dict[str, Any]] = []
    for model_name, backend, model_path, extra in [
        ("nep89", "calorine", nep_model_path, []),
        ("dpa4neo", "dpa4neo", dpa_model_path, ["--deepmd-device", "cpu", "--deepmd-torch-threads", "1"]),
    ]:
        for force_backend in ["direct", "origin"]:
            name = f"{model_name}_{force_backend}"
            run_outdir = outdir / name
            command = [
                str(python),
                "-m",
                "phonoflow",
                "single",
                "--input-path",
                str(_resolve_input(root, input_path)),
                "--backend",
                backend,
                "--model-path",
                str(model_path),
                "--outdir",
                str(run_outdir),
                "--compute-kappa",
                "--fc3-method",
                "finite-displacement",
                "--method",
                "rta",
                "--temperatures",
                "300",
                "--kappa-mesh",
                "11",
                "11",
                "11",
                "--force-workers",
                "1",
                "--force-parallel-backend",
                force_backend,
                *extra,
            ]
            runs.append(
                {
                    "name": name,
                    "model": model_name,
                    "force_parallel_backend": force_backend,
                    "outdir": str(run_outdir),
                    "command": command,
                }
            )
    return {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "root": str(root),
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
    record = {
        **run,
        "returncode": int(completed.returncode),
        "total_wall_time_s": elapsed,
        "fc3_force_wall_time_s": first(
            result_json,
            ["thermal_conductivity.timing_breakdown.fc3_wall_time", "thermal.force_evaluation.fc3_wall_time"],
        ),
        "fc3_displacements_per_second": first(
            result_json,
            [
                "thermal_conductivity.timing_breakdown.fc3_displacements_per_second",
                "thermal.force_evaluation.fc3_displacements_per_second",
            ],
        ),
        "n_fc3_displacements": first(result_json, ["thermal_conductivity.n_fc3_displacements", "thermal.n_fc3_displacements"]),
        "kappa_scalar_300K": first(
            result_json,
            ["thermal_conductivity.summary.300.kappa_trace_over_3", "thermal.summary.300.kappa_trace_over_3"],
        ),
    }
    print(
        f"[{idx}/{total}] DONE rc={record['returncode']} total={elapsed:.3f}s "
        f"fc3={record['fc3_force_wall_time_s']} kappa={record['kappa_scalar_300K']}",
        flush=True,
    )
    return record


def summarize(results: list[dict[str, Any]]) -> dict[str, Any]:
    summary: dict[str, Any] = {}
    for model in sorted({str(row.get("model")) for row in results}):
        rows = [row for row in results if row.get("model") == model and row.get("returncode") == 0]
        direct = next((row for row in rows if row.get("force_parallel_backend") == "direct"), None)
        origin = next((row for row in rows if row.get("force_parallel_backend") == "origin"), None)
        summary[model] = {
            "direct": compact_row(direct),
            "origin": compact_row(origin),
            "kappa_abs_delta": abs(float(direct["kappa_scalar_300K"]) - float(origin["kappa_scalar_300K"]))
            if direct and origin and isinstance(direct.get("kappa_scalar_300K"), (int, float)) and isinstance(origin.get("kappa_scalar_300K"), (int, float))
            else None,
            "fc3_force_time_delta_s": float(origin["fc3_force_wall_time_s"]) - float(direct["fc3_force_wall_time_s"])
            if direct and origin and isinstance(direct.get("fc3_force_wall_time_s"), (int, float)) and isinstance(origin.get("fc3_force_wall_time_s"), (int, float))
            else None,
        }
    return summary


def compact_row(row: dict[str, Any] | None) -> dict[str, Any] | None:
    if row is None:
        return None
    return {
        "returncode": row.get("returncode"),
        "total_wall_time_s": row.get("total_wall_time_s"),
        "fc3_force_wall_time_s": row.get("fc3_force_wall_time_s"),
        "kappa_scalar_300K": row.get("kappa_scalar_300K"),
        "outdir": row.get("outdir"),
    }


def read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}


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


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fields = [
        "name",
        "model",
        "force_parallel_backend",
        "returncode",
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
        "# Origin Vs Direct Benchmark",
        "",
        "## Summary",
        "",
        "```json",
        json.dumps(report["summary"], indent=2),
        "```",
        "",
        "## Runs",
        "",
        "| name | rc | total s | FC3 force s | disp/s | kappa |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for row in report.get("results", []):
        lines.append(
            "| {name} | {rc} | {total} | {fc3} | {dps} | {kappa} |".format(
                name=row.get("name", ""),
                rc=row.get("returncode", ""),
                total=format_number(row.get("total_wall_time_s")),
                fc3=format_number(row.get("fc3_force_wall_time_s")),
                dps=format_number(row.get("fc3_displacements_per_second")),
                kappa=format_number(row.get("kappa_scalar_300K"), digits=10),
            )
        )
    return "\n".join(lines) + "\n"


def format_number(value: Any, *, digits: int = 6) -> str:
    if isinstance(value, (int, float)):
        return f"{float(value):.{digits}f}"
    return ""


def _resolve_input(root: Path, input_path: Path) -> Path:
    return input_path if input_path.is_absolute() else root / input_path


if __name__ == "__main__":
    main()
