from __future__ import annotations

import json
from pathlib import Path

from phonoflow.reporting.run_report import build_default_audit_table, create_run_folder, write_run_report


def test_run_report_writes_all_required_files_in_one_folder(tmp_path: Path) -> None:
    run_dir = create_run_folder(base_dir=tmp_path, prefix="dpa_deepmd_audit", timestamp="20260102_030405")
    (run_dir / "artifact.dat").write_text("data", encoding="utf-8")

    write_run_report(
        run_dir,
        title="DPA DeepMD Audit",
        summary={
            "status": "success",
            "default_audit": [
                {
                    "setting": "supercell_dim",
                    "NEP89 default": "auto",
                    "DPA default": "auto",
                    "DPA3 default": "auto",
                    "DPA4 default": "auto",
                    "consistent?": "yes",
                    "intended?": "yes",
                    "action": "keep aligned",
                }
            ],
        },
        commands=["python -m phonoflow single --help"],
        validation_lines=["[1/8] Resolve backend and model"],
    )

    assert run_dir.name == "dpa_deepmd_audit_20260102_030405"
    for filename in (
        "report_en.md",
        "report_zh.md",
        "summary.json",
        "commands.log",
        "validation.log",
        "environment.json",
        "artifacts_index.csv",
        "artifacts_index.json",
    ):
        assert (run_dir / filename).exists(), filename

    summary = json.loads((run_dir / "summary.json").read_text(encoding="utf-8"))
    assert summary["default_audit"][0]["setting"] == "supercell_dim"
    assert "artifact.dat" in (run_dir / "artifacts_index.csv").read_text(encoding="utf-8")


def test_default_audit_table_records_intended_dpa_differences() -> None:
    rows = build_default_audit_table()
    by_setting = {row["setting"]: row for row in rows}

    assert by_setting["supercell_dim"]["consistent?"] == "yes"
    assert by_setting["fc3_supercell_dim"]["consistent?"] == "yes"
    assert by_setting["kappa_mesh"]["consistent?"] == "yes"
    assert by_setting["phono3py_symmetrize_fc2"]["consistent?"] == "no"
    assert by_setting["phono3py_symmetrize_fc2"]["intended?"] == "yes"
    assert by_setting["deepmd_deterministic"]["intended?"] == "yes"
    assert by_setting["save_force_audit"]["intended?"] == "yes"
