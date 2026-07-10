"""Summarize a completed NEP89 FC3 scheduler benchmark directory."""

from __future__ import annotations

import argparse
import csv
import json
import re
from pathlib import Path
from typing import Any


FC3_RE = re.compile(r"FC3 force evaluation completed: wall_time=([0-9.]+)s, displacements_per_second=([0-9.]+)")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("benchmark_dir", type=Path)
    args = parser.parse_args()

    benchmark_dir = args.benchmark_dir.resolve()
    plan = _read_json(benchmark_dir / "benchmark_plan.json")
    measured = _read_json(benchmark_dir / "benchmark_results.json")
    measured_by_run = {
        (row["structure"], row["strategy"]): row
        for row in measured.get("results", [])
        if isinstance(row, dict)
    }
    rows = []
    for run in plan.get("runs", []):
        if not isinstance(run, dict):
            continue
        run_outdir = Path(str(run["outdir"]))
        rows.append(_summarize_run(run, run_outdir, measured_by_run.get((run["structure"], run["strategy"]), {})))
    summary = _summarize_by_structure(rows)
    report = {
        "benchmark_dir": str(benchmark_dir),
        "model": plan.get("model"),
        "results": rows,
        "summary": summary,
    }
    (benchmark_dir / "benchmark_results_corrected.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    _write_csv(benchmark_dir / "benchmark_results_corrected.csv", rows)
    (benchmark_dir / "benchmark_report_corrected.md").write_text(_markdown(report), encoding="utf-8")
    print(benchmark_dir / "benchmark_report_corrected.md")


def _summarize_run(run: dict[str, Any], run_outdir: Path, measured: dict[str, Any]) -> dict[str, Any]:
    timing = _read_json(run_outdir / "timing_breakdown.json")
    diagnostics = _read_json(run_outdir / "fd_phono3py_input_diagnostics.json")
    fc3_wall, dps = _parse_fc3_force_log(run_outdir / "run.log")
    command = run.get("command", [])
    return {
        "structure": run["structure"],
        "strategy": run["strategy"],
        "returncode": measured.get("returncode"),
        "total_wall_time_s": measured.get("total_wall_time_s"),
        "pipeline_total_seconds": timing.get("total_seconds"),
        "fc3_force_wall_time_s": fc3_wall,
        "fc3_displacements_per_second": dps,
        "fc3_stage_seconds": _stage_seconds(timing, "fc3"),
        "thermal_solver_seconds": _stage_seconds(timing, "thermal_lifetime"),
        "fc2_phonon_seconds": _stage_seconds(timing, "fc2_phonon"),
        "fc3_thermal_seconds": _stage_seconds(timing, "fc3_thermal"),
        "n_fc3_displacements": diagnostics.get("n_fc3_displacements"),
        "n_atoms_supercell": diagnostics.get("n_atoms_supercell"),
        "kappa_trace_over_3_300K": _read_kappa(run_outdir / "thermal_conductivity.csv"),
        "force_parallel_backend": _command_value(command, "--force-parallel-backend"),
        "force_workers": _command_value(command, "--force-workers"),
        "force_chunk_size": _command_value(command, "--force-chunk-size") or "auto",
        "outdir": str(run_outdir),
    }


def _summarize_by_structure(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    by_structure: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        by_structure.setdefault(str(row["structure"]), []).append(row)
    summary = []
    for structure, records in by_structure.items():
        ok = [row for row in records if row["returncode"] == 0]
        kappas = [float(row["kappa_trace_over_3_300K"]) for row in ok if isinstance(row.get("kappa_trace_over_3_300K"), (int, float))]
        best_total = min(ok, key=lambda row: float(row["total_wall_time_s"])) if ok else None
        best_fc3 = min(
            [row for row in ok if isinstance(row.get("fc3_force_wall_time_s"), (int, float))],
            key=lambda row: float(row["fc3_force_wall_time_s"]),
            default=None,
        )
        legacy = next((row for row in ok if str(row["strategy"]).startswith("legacy_auto_equiv")), None)
        auto = next((row for row in ok if row["strategy"] == "new_auto_process_w30"), None)
        summary.append(
            {
                "structure": structure,
                "successful_runs": len(ok),
                "failed_runs": len(records) - len(ok),
                "kappa_min": min(kappas) if kappas else None,
                "kappa_max": max(kappas) if kappas else None,
                "kappa_spread_abs": max(kappas) - min(kappas) if kappas else None,
                "best_total_strategy": best_total["strategy"] if best_total else None,
                "best_total_wall_time_s": best_total["total_wall_time_s"] if best_total else None,
                "best_fc3_strategy": best_fc3["strategy"] if best_fc3 else None,
                "best_fc3_force_wall_time_s": best_fc3["fc3_force_wall_time_s"] if best_fc3 else None,
                "legacy_total_wall_time_s": legacy.get("total_wall_time_s") if legacy else None,
                "legacy_fc3_force_wall_time_s": legacy.get("fc3_force_wall_time_s") if legacy else None,
                "new_auto_total_wall_time_s": auto.get("total_wall_time_s") if auto else None,
                "new_auto_fc3_force_wall_time_s": auto.get("fc3_force_wall_time_s") if auto else None,
                "best_total_speedup_vs_legacy": _speedup(legacy, best_total, "total_wall_time_s"),
                "best_fc3_speedup_vs_legacy": _speedup(legacy, best_fc3, "fc3_force_wall_time_s"),
                "new_auto_total_speedup_vs_legacy": _speedup(legacy, auto, "total_wall_time_s"),
                "new_auto_fc3_speedup_vs_legacy": _speedup(legacy, auto, "fc3_force_wall_time_s"),
            }
        )
    return summary


def _parse_fc3_force_log(path: Path) -> tuple[float | None, float | None]:
    if not path.exists():
        return None, None
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        match = FC3_RE.search(line)
        if match:
            return float(match.group(1)), float(match.group(2))
    return None, None


def _read_kappa(path: Path) -> float | None:
    if not path.exists():
        return None
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            value = row.get("kappa_trace_over_3")
            return float(value) if value not in (None, "") else None
    return None


def _stage_seconds(timing: dict[str, Any], name: str) -> float | None:
    stage = timing.get("stages", {}).get(name)
    if isinstance(stage, dict) and isinstance(stage.get("seconds"), (int, float)):
        return float(stage["seconds"])
    return None


def _command_value(command: list[Any], option: str) -> str | None:
    try:
        index = command.index(option)
    except ValueError:
        return None
    if index + 1 >= len(command):
        return None
    return str(command[index + 1])


def _speedup(base: dict[str, Any] | None, other: dict[str, Any] | None, key: str) -> float | None:
    if not base or not other:
        return None
    base_value = base.get(key)
    other_value = other.get(key)
    if not isinstance(base_value, (int, float)) or not isinstance(other_value, (int, float)) or other_value <= 0:
        return None
    return float(base_value / other_value)


def _read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def _markdown(report: dict[str, Any]) -> str:
    lines = [
        "# NEP89 FC3 Scheduler Benchmark Corrected Report",
        "",
        f"- Benchmark dir: {report['benchmark_dir']}",
        f"- Model: {report['model']}",
        "- Method: finite-displacement FC3, RTA, kappa mesh 11 11 11, T=300 K",
        "- Kappa source: thermal_conductivity.csv / kappa_trace_over_3",
        "- FC3 force timing source: run.log / FC3 force evaluation completed",
        "",
        "## Summary",
        "",
        "| Structure | runs ok | kappa spread | best total strategy | best total s | best FC3 strategy | best FC3 force s | best total speedup vs legacy | best FC3 speedup vs legacy | new auto FC3 speedup vs legacy |",
        "|---|---:|---:|---|---:|---|---:|---:|---:|---:|",
    ]
    for row in report["summary"]:
        lines.append(
            "| {structure} | {successful_runs} | {kappa_spread_abs} | {best_total_strategy} | {best_total_wall_time_s:.3f} | {best_fc3_strategy} | {best_fc3_force_wall_time_s} | {best_total_speedup_vs_legacy} | {best_fc3_speedup_vs_legacy} | {new_auto_fc3_speedup_vs_legacy} |".format(
                **row
            )
        )
    lines.extend(
        [
            "",
            "## Full Runs",
            "",
            "| Structure | Strategy | rc | total s | pipeline s | FC3 force s | disp/s | kappa | chunk | backend | workers |",
            "|---|---|---:|---:|---:|---:|---:|---:|---:|---|---:|",
        ]
    )
    for row in report["results"]:
        lines.append(
            "| {structure} | {strategy} | {returncode} | {total_wall_time_s:.3f} | {pipeline_total_seconds} | {fc3_force_wall_time_s} | {fc3_displacements_per_second} | {kappa_trace_over_3_300K} | {force_chunk_size} | {force_parallel_backend} | {force_workers} |".format(
                **row
            )
        )
    return "\n".join(lines) + "\n"


if __name__ == "__main__":
    main()
