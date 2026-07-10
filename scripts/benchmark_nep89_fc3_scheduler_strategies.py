"""Run NEP89 FC3 scheduler strategy benchmarks for example structures."""

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


STRUCTURES = {
    "BAs": {"input": "examples/BAs.vasp", "legacy_chunk": 6},
    "Diamond": {"input": "examples/Diamond.vasp", "legacy_chunk": 2},
    "GaN": {"input": "examples/GaN.vasp", "legacy_chunk": 10},
    "SiC": {"input": "examples/SiC.vasp", "legacy_chunk": 16},
}

STRATEGIES = [
    {"name": "new_auto_process_w30", "workers": 30, "backend": "process", "chunk": None},
    {"name": "process_chunk2_w30", "workers": 30, "backend": "process", "chunk": 2},
    {"name": "worker_queue_chunk1_w30", "workers": 30, "backend": "worker_queue", "chunk": 1},
    {"name": "process_chunk1_w24", "workers": 24, "backend": "process", "chunk": 1},
    {"name": "process_chunk1_w36", "workers": 36, "backend": "process", "chunk": 1},
]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path.cwd())
    parser.add_argument("--python", type=Path, default=Path(sys.executable))
    parser.add_argument("--model-path", type=Path, required=True)
    parser.add_argument("--outdir", type=Path, required=True)
    args = parser.parse_args()

    root = args.root.resolve()
    outdir = args.outdir.resolve()
    model_path = args.model_path.resolve()
    outdir.mkdir(parents=True, exist_ok=False)
    results: list[dict[str, Any]] = []
    plan = _build_plan(root, args.python, model_path, outdir)
    (outdir / "benchmark_plan.json").write_text(json.dumps(plan, indent=2), encoding="utf-8")
    print(f"Benchmark plan written: {outdir / 'benchmark_plan.json'}", flush=True)

    for idx, run in enumerate(plan["runs"], start=1):
        result = _run_one(run, root, idx=idx, total=len(plan["runs"]))
        results.append(result)
        (outdir / "partial_results.json").write_text(json.dumps(results, indent=2), encoding="utf-8")

    report = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "outdir": str(outdir),
        "model": str(model_path),
        "results": results,
        "summary": _summarize(results),
    }
    (outdir / "benchmark_results.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    _write_csv(outdir / "benchmark_results.csv", results)
    (outdir / "benchmark_report.md").write_text(_markdown_report(report), encoding="utf-8")
    print(f"FINAL_REPORT={outdir / 'benchmark_report.md'}", flush=True)


def _build_plan(root: Path, python: Path, model_path: Path, outdir: Path) -> dict[str, Any]:
    runs: list[dict[str, Any]] = []
    for structure, info in STRUCTURES.items():
        strategies = list(STRATEGIES)
        strategies.append(
            {
                "name": f"legacy_auto_equiv_chunk{info['legacy_chunk']}_w30",
                "workers": 30,
                "backend": "process",
                "chunk": info["legacy_chunk"],
            }
        )
        for strategy in strategies:
            run_outdir = outdir / structure / str(strategy["name"])
            command = [
                str(python),
                "-m",
                "phonoflow",
                "single",
                "--input-path",
                str(root / str(info["input"])),
                "--backend",
                "calorine",
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
                str(strategy["workers"]),
                "--force-parallel-backend",
                str(strategy["backend"]),
            ]
            if strategy["chunk"] is not None:
                command.extend(["--force-chunk-size", str(strategy["chunk"])])
            runs.append(
                {
                    "structure": structure,
                    "strategy": strategy["name"],
                    "command": command,
                    "outdir": str(run_outdir),
                }
            )
    return {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "outdir": str(outdir),
        "model": str(model_path),
        "structures": STRUCTURES,
        "strategies": STRATEGIES,
        "runs": runs,
    }


def _run_one(run: dict[str, Any], root: Path, *, idx: int, total: int) -> dict[str, Any]:
    run_outdir = Path(str(run["outdir"]))
    run_outdir.mkdir(parents=True, exist_ok=False)
    (run_outdir / "benchmark_command.json").write_text(json.dumps(run, indent=2), encoding="utf-8")
    print(f"[{idx}/{total}] START {run['structure']} {run['strategy']}", flush=True)
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
    result_json = _read_json(run_outdir / "result.json")
    timing_json = _read_json(run_outdir / "timing_breakdown.json")
    record = {
        "structure": run["structure"],
        "strategy": run["strategy"],
        "returncode": int(completed.returncode),
        "total_wall_time_s": elapsed,
        "fc3_force_wall_time_s": _first(
            result_json,
            [
                "thermal.force_evaluation.fc3_wall_time",
                "force_evaluation.fc3_wall_time",
                "thermal.fc3_wall_time",
            ],
        )
        or _first(timing_json, ["thermal.force_evaluation.fc3_wall_time", "fc3_wall_time"]),
        "fc3_displacements_per_second": _first(
            result_json,
            [
                "thermal.force_evaluation.fc3_displacements_per_second",
                "force_evaluation.fc3_displacements_per_second",
                "thermal.fc3_displacements_per_second",
            ],
        ),
        "n_fc3_displacements": _first(result_json, ["thermal.n_fc3_displacements", "n_fc3_displacements"]),
        "kappa_scalar_300K": _extract_kappa(result_json),
        "chunk_size": _first(result_json, ["thermal.force_evaluation.chunk_size", "force_evaluation.chunk_size"]),
        "force_parallel_backend": _first(
            result_json,
            ["thermal.force_evaluation.force_parallel_backend", "force_evaluation.force_parallel_backend"],
        ),
        "effective_force_workers": _first(
            result_json,
            ["thermal.force_evaluation.fc3_force_workers", "force_evaluation.fc3_force_workers"],
        ),
        "outdir": str(run_outdir),
    }
    print(
        f"[{idx}/{total}] DONE rc={record['returncode']} total={elapsed:.3f}s "
        f"fc3={record['fc3_force_wall_time_s']} kappa={record['kappa_scalar_300K']}",
        flush=True,
    )
    return record


def _read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        return {"_error": str(exc)}


def _first(data: dict[str, Any], paths: list[str]) -> Any:
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


def _extract_kappa(result_json: dict[str, Any]) -> float | None:
    rows = _first(result_json, ["thermal.thermal_conductivity.rows", "thermal_conductivity.rows"])
    if isinstance(rows, list) and rows:
        row = rows[0]
        if isinstance(row, dict):
            for key in ("kappa_total", "kappa_average", "kappa_iso", "kappa"):
                if isinstance(row.get(key), (int, float)):
                    return float(row[key])
    candidates: list[float] = []

    def walk(value: Any) -> None:
        if isinstance(value, dict):
            for key, item in value.items():
                if isinstance(item, (int, float)) and "kappa" in str(key).lower() and "mesh" not in str(key).lower():
                    candidates.append(float(item))
                walk(item)
        elif isinstance(value, list):
            for item in value:
                walk(item)

    walk(result_json)
    return candidates[0] if candidates else None


def _summarize(results: list[dict[str, Any]]) -> list[dict[str, Any]]:
    by_structure: dict[str, list[dict[str, Any]]] = {}
    for record in results:
        by_structure.setdefault(str(record["structure"]), []).append(record)
    summary: list[dict[str, Any]] = []
    for structure, rows in by_structure.items():
        ok_rows = [row for row in rows if row["returncode"] == 0]
        kappas = [float(row["kappa_scalar_300K"]) for row in ok_rows if isinstance(row.get("kappa_scalar_300K"), (int, float))]
        fc3_rows = [row for row in ok_rows if isinstance(row.get("fc3_force_wall_time_s"), (int, float))]
        best_total = min(ok_rows, key=lambda row: row["total_wall_time_s"]) if ok_rows else None
        best_fc3 = min(fc3_rows, key=lambda row: row["fc3_force_wall_time_s"]) if fc3_rows else None
        summary.append(
            {
                "structure": structure,
                "successful_runs": len(ok_rows),
                "failed_runs": len(rows) - len(ok_rows),
                "kappa_min": min(kappas) if kappas else None,
                "kappa_max": max(kappas) if kappas else None,
                "kappa_spread_abs": max(kappas) - min(kappas) if kappas else None,
                "best_total_strategy": best_total["strategy"] if best_total else None,
                "best_total_wall_time_s": best_total["total_wall_time_s"] if best_total else None,
                "best_fc3_strategy": best_fc3["strategy"] if best_fc3 else None,
                "best_fc3_force_wall_time_s": best_fc3["fc3_force_wall_time_s"] if best_fc3 else None,
            }
        )
    return summary


def _write_csv(path: Path, results: list[dict[str, Any]]) -> None:
    if not results:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(results[0].keys()))
        writer.writeheader()
        writer.writerows(results)


def _markdown_report(report: dict[str, Any]) -> str:
    lines = [
        "# NEP89 FC3 Scheduler Benchmark",
        "",
        f"- Output: {report['outdir']}",
        f"- Model: {report['model']}",
        "- Method: finite-displacement FC3, RTA, kappa mesh 11 11 11, T=300 K",
        "",
        "## Summary",
        "",
        "| Structure | Successful | Failed | kappa min | kappa max | spread | Best total strategy | Best total s | Best FC3 strategy | Best FC3 s |",
        "|---|---:|---:|---:|---:|---:|---|---:|---|---:|",
    ]
    for row in report["summary"]:
        lines.append(
            "| {structure} | {successful_runs} | {failed_runs} | {kappa_min} | {kappa_max} | {kappa_spread_abs} | {best_total_strategy} | {best_total_wall_time_s:.3f} | {best_fc3_strategy} | {best_fc3_force_wall_time_s} |".format(
                **row
            )
        )
    lines.extend(
        [
            "",
            "## Full Runs",
            "",
            "| Structure | Strategy | rc | total s | FC3 force s | disp/s | kappa | chunk | backend | workers |",
            "|---|---|---:|---:|---:|---:|---:|---:|---|---:|",
        ]
    )
    for record in report["results"]:
        lines.append(
            "| {structure} | {strategy} | {returncode} | {total_wall_time_s:.3f} | {fc3_force_wall_time_s} | {fc3_displacements_per_second} | {kappa_scalar_300K} | {chunk_size} | {force_parallel_backend} | {effective_force_workers} |".format(
                **record
            )
        )
    return "\n".join(lines) + "\n"


if __name__ == "__main__":
    main()
