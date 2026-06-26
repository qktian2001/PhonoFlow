"""Run-folder reporting helpers for validation and audit workflows."""

from __future__ import annotations

import csv
import json
import platform
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable

from phonoflow.config import WorkflowConfig
from phonoflow.constants import VERSION


def create_run_folder(
    base_dir: Path | str = Path("results"),
    prefix: str = "dpa_deepmd_audit",
    timestamp: str | None = None,
) -> Path:
    """Create one timestamped folder for a complete validation/audit run."""

    stamp = timestamp or datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = Path(base_dir) / f"{prefix}_{stamp}"
    run_dir.mkdir(parents=True, exist_ok=False)
    return run_dir


class StepReporter:
    """Write readable step output to stdout and a validation log."""

    def __init__(self, *, total: int, log_path: Path | None = None, console: Any | None = None) -> None:
        self.total = int(total)
        self.log_path = Path(log_path) if log_path is not None else None
        self.console = console
        self.started_at = time.perf_counter()
        self.last_step_at = self.started_at
        self.records: list[dict[str, Any]] = []
        if self.log_path is not None:
            self.log_path.parent.mkdir(parents=True, exist_ok=True)
            self.log_path.write_text("", encoding="utf-8")

    def step(
        self,
        number: int,
        title: str,
        *,
        details: Iterable[str] | None = None,
        status: str | None = None,
        warning: str | None = None,
    ) -> None:
        """Record and print one workflow step."""

        now = time.perf_counter()
        elapsed = now - self.last_step_at
        self.last_step_at = now
        header = f"[{number}/{self.total}] {title}"
        suffix = f" | status={status}" if status else ""
        elapsed_text = f" | elapsed={elapsed:.2f}s"
        lines = [f"{header}{suffix}{elapsed_text}"]
        for detail in details or []:
            lines.append(f"  - {detail}")
        if warning:
            lines.append(f"  - warning: {warning}")
        text = "\n".join(lines)
        if self.console is not None:
            self.console.print(text)
        else:
            print(text)
        if self.log_path is not None:
            with self.log_path.open("a", encoding="utf-8") as handle:
                handle.write(text + "\n")
        self.records.append(
            {
                "step": int(number),
                "title": title,
                "status": status,
                "elapsed_since_previous_seconds": round(elapsed, 3),
                "details": list(details or []),
                "warning": warning,
            }
        )


def build_default_audit_table() -> list[dict[str, str]]:
    """Return the NEP89/DPA default-policy audit table used in run reports."""

    base = WorkflowConfig()
    rows = [
        _audit_row("supercell_dim", "auto", "auto", True, "Aligned with NEP89 auto supercell policy."),
        _audit_row(
            "target_supercell_length",
            str(base.target_supercell_length),
            str(base.target_supercell_length),
            True,
            "Same auto-supercell target length.",
        ),
        _audit_row("fc3_supercell_dim", "auto", "auto", True, "Aligned with NEP89 FC3 auto policy."),
        _audit_row("kappa_mesh", "auto", "auto", True, "Both resolve auto kappa mesh in the thermal backend."),
        _audit_row("displacement", str(base.displacement), str(base.displacement), True, "Same FC2 displacement default."),
        _audit_row(
            "fc3_displacement",
            str(base.fc3_displacement),
            str(base.fc3_displacement),
            True,
            "Same FC3 displacement default.",
        ),
        _audit_row(
            "phonopy_symprec",
            "1e-5",
            "1e-5",
            True,
            "Aligned with phonopy SYMMETRY_TOLERANCE / API symprec default.",
        ),
        _audit_row("phono3py_symprec", "1e-5", "1e-5", True, "PhonoFlow records the phono3py API default explicitly instead of inheriting phonopy_symprec."),
        _audit_row("phono3py_cutoff_frequency", "1e-4", "1e-4", True, "PhonoFlow records the phono3py API cutoff-frequency default explicitly."),
        _audit_row(
            "export_fc2_text",
            str(base.export_fc2_text),
            str(base.export_fc2_text),
            True,
            "Same force-constant text export policy.",
        ),
        _audit_row(
            "phono3py_symmetrize_fc2",
            "False",
            "True",
            False,
            "Intentional DPA difference from ASR repeatability validation.",
        ),
        _audit_row(
            "deepmd_deterministic",
            "False / not applicable",
            "True",
            False,
            "Intentional DPA runtime reproducibility setting.",
        ),
        _audit_row(
            "deepmd_reuse_calculator",
            "True / not applicable",
            "True",
            True,
            "DeepMD calculator reuse is enabled for DPA efficiency.",
        ),
        _audit_row(
            "save_force_audit",
            "False",
            "True",
            False,
            "Intentional DPA provenance and debugging default.",
        ),
        _audit_row(
            "relax",
            "True",
            "False",
            False,
            "Intentional DPA policy: avoid silent mixed-model relaxation unless the user requests relaxation.",
        ),
        _audit_row(
            "relax_backend",
            "force backend",
            "NEP89/Calorine when relaxation is explicitly enabled",
            False,
            "Intentional DPA safety policy for explicit relaxation.",
        ),
    ]
    return rows


def write_run_report(
    run_dir: Path | str,
    *,
    title: str,
    summary: dict[str, Any],
    commands: Iterable[str] = (),
    validation_lines: Iterable[str] = (),
) -> dict[str, Any]:
    """Write bilingual run reports and machine-readable indexes into one folder."""

    run_path = Path(run_dir)
    run_path.mkdir(parents=True, exist_ok=True)
    commands_list = list(commands)
    validation_list = list(validation_lines)
    default_audit = list(summary.get("default_audit") or build_default_audit_table())
    full_summary = {
        **summary,
        "title": title,
        "run_dir": str(run_path),
        "phonoflow_version": VERSION,
        "default_audit": default_audit,
    }
    (run_path / "commands.log").write_text("\n".join(commands_list) + ("\n" if commands_list else ""), encoding="utf-8")
    existing_validation = ""
    validation_path = run_path / "validation.log"
    if validation_path.exists():
        existing_validation = validation_path.read_text(encoding="utf-8")
    validation_text = existing_validation + ("\n".join(validation_list) + ("\n" if validation_list else ""))
    validation_path.write_text(validation_text, encoding="utf-8")
    environment = collect_environment()
    (run_path / "environment.json").write_text(json.dumps(environment, indent=2, ensure_ascii=False), encoding="utf-8")
    artifacts = collect_artifacts_index(run_path)
    _write_artifacts_csv(run_path / "artifacts_index.csv", artifacts)
    (run_path / "artifacts_index.json").write_text(json.dumps(artifacts, indent=2, ensure_ascii=False), encoding="utf-8")
    full_summary["environment_file"] = "environment.json"
    full_summary["artifacts_index_csv"] = "artifacts_index.csv"
    full_summary["artifacts_index_json"] = "artifacts_index.json"
    (run_path / "summary.json").write_text(json.dumps(full_summary, indent=2, ensure_ascii=False), encoding="utf-8")
    (run_path / "report_en.md").write_text(_markdown_report(full_summary, language="en"), encoding="utf-8")
    (run_path / "report_zh.md").write_text(_markdown_report(full_summary, language="zh"), encoding="utf-8")
    return full_summary


def collect_environment() -> dict[str, Any]:
    """Collect a small reproducibility environment snapshot."""

    return {
        "python_executable": sys.executable,
        "python_version": sys.version.replace("\n", " "),
        "platform": platform.platform(),
        "phonoflow_version": VERSION,
        "cwd": str(Path.cwd()),
    }


def collect_artifacts_index(run_dir: Path | str) -> list[dict[str, Any]]:
    """Index files under a run folder."""

    run_path = Path(run_dir)
    rows: list[dict[str, Any]] = []
    for path in sorted(item for item in run_path.rglob("*") if item.is_file()):
        rel = path.relative_to(run_path).as_posix()
        rows.append(
            {
                "path": rel,
                "bytes": path.stat().st_size,
                "suffix": path.suffix,
            }
        )
    return rows


def _audit_row(setting: str, nep89: str, dpa: str, consistent: bool, action: str) -> dict[str, str]:
    intended = "yes" if consistent or setting in {
        "phono3py_symmetrize_fc2",
        "deepmd_deterministic",
        "deepmd_reuse_calculator",
        "save_force_audit",
        "relax",
        "relax_backend",
    } else "no"
    return {
        "setting": setting,
        "NEP89 default": nep89,
        "DPA default": dpa,
        "DPA3 default": dpa,
        "DPA4 default": dpa,
        "consistent?": "yes" if consistent else "no",
        "intended?": intended,
        "action": action,
    }


def _write_artifacts_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fieldnames = ["path", "bytes", "suffix"]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _markdown_report(summary: dict[str, Any], *, language: str) -> str:
    title = str(summary.get("title") or "Run Report")
    if language == "zh":
        lines = [
            f"# {title}",
            "",
            "## 运行摘要",
            "",
            f"- 运行目录: `{summary.get('run_dir')}`",
            f"- 状态: `{summary.get('status', 'unknown')}`",
            f"- PhonoFlow 版本: `{summary.get('phonoflow_version')}`",
            "",
            "## 默认参数审计",
        ]
    else:
        lines = [
            f"# {title}",
            "",
            "## Summary",
            "",
            f"- Run directory: `{summary.get('run_dir')}`",
            f"- Status: `{summary.get('status', 'unknown')}`",
            f"- PhonoFlow version: `{summary.get('phonoflow_version')}`",
            "",
            "## Default Audit",
        ]
    lines.extend(
        [
            "",
            "| setting | NEP89 default | DPA default | DPA3 default | DPA4 default | consistent? | intended? | action |",
            "| --- | --- | --- | --- | --- | --- | --- | --- |",
        ]
    )
    for row in summary.get("default_audit") or []:
        lines.append(
            "| {setting} | {nep89} | {dpa} | {dpa3} | {dpa4} | {consistent} | {intended} | {action} |".format(
                setting=_escape_md(row.get("setting")),
                nep89=_escape_md(row.get("NEP89 default")),
                dpa=_escape_md(row.get("DPA default")),
                dpa3=_escape_md(row.get("DPA3 default")),
                dpa4=_escape_md(row.get("DPA4 default")),
                consistent=_escape_md(row.get("consistent?")),
                intended=_escape_md(row.get("intended?")),
                action=_escape_md(row.get("action")),
            )
        )
    lines.extend(["", "## Artifacts" if language == "en" else "## 输出文件", "", "- `summary.json`", "- `commands.log`", "- `validation.log`", "- `environment.json`", "- `artifacts_index.csv`"])
    if summary.get("models"):
        lines.extend(["", "## Model Results" if language == "en" else "## 模型结果", ""])
        for row in summary["models"]:
            lines.append(
                f"- `{row.get('model')}`: status=`{row.get('status')}`, kavg=`{row.get('kavg')}`, outdir=`{row.get('outdir')}`"
            )
    return "\n".join(lines) + "\n"


def _escape_md(value: Any) -> str:
    return str(value if value is not None else "").replace("|", "/")
