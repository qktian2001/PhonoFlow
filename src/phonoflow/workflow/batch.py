"""Batch workflow skeleton."""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

from phonoflow.config import WorkflowConfig
from phonoflow.exceptions import ConfigError
from phonoflow.io.path_utils import ensure_dir, find_structure_files, safe_stem
from phonoflow.reporting.summary_csv import write_summary_csv
from phonoflow.workflow.pipeline import run_single_workflow


def _run_one(config: WorkflowConfig, structure_path: Path, outdir: Path) -> dict[str, Any]:
    single_config = config.model_copy(update={"input_path": structure_path, "outdir": outdir})

    if config.resume and (outdir / "stability_report.json").exists():
        return {
            "structure_name": structure_path.name,
            "status": "skipped",
            "outdir": str(outdir),
            "dynamically_stable": "",
            "minimum_frequency_THz": "",
            "error_message": "Skipped because resume=True and stability_report.json exists.",
        }

    try:
        return run_single_workflow(single_config)
    except Exception as exc:
        ensure_dir(outdir)
        return {
            "structure_name": structure_path.name,
            "status": "failed",
            "outdir": str(outdir),
            "dynamically_stable": "",
            "minimum_frequency_THz": "",
            "error_message": str(exc),
        }


def run_batch_workflow(config: WorkflowConfig) -> list[dict[str, Any]]:
    """Run the v0.1 batch workflow skeleton."""

    if config.input_dir is None:
        raise ConfigError("batch workflow requires input_dir.")

    input_dir = Path(config.input_dir)
    if not input_dir.is_dir():
        raise ConfigError(f"input_dir does not exist or is not a directory: {input_dir}")

    root_outdir = ensure_dir(Path(config.outdir or "results"))
    structure_files = find_structure_files(input_dir)
    if not structure_files:
        raise ConfigError(f"No supported structure files found in input_dir: {input_dir}")

    results: list[dict[str, Any]] = []
    if config.max_workers == 1:
        for structure_path in structure_files:
            outdir = root_outdir / safe_stem(structure_path)
            results.append(_run_one(config, structure_path, outdir))
    else:
        with ThreadPoolExecutor(max_workers=config.max_workers) as executor:
            future_map = {
                executor.submit(_run_one, config, structure_path, root_outdir / safe_stem(structure_path)): structure_path
                for structure_path in structure_files
            }
            for future in as_completed(future_map):
                results.append(future.result())

    results = sorted(results, key=lambda item: item.get("structure_name", ""))
    write_summary_csv(results, root_outdir / "summary.csv")
    return results
