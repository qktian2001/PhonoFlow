from __future__ import annotations

import importlib.util
import json
from pathlib import Path


def _load_validator():
    script_path = Path(__file__).resolve().parents[1] / "scripts" / "validate_output.py"
    spec = importlib.util.spec_from_file_location("validate_output", script_path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_validate_output_checks_fc2_text_files(tmp_path: Path):
    validator = _load_validator()
    outdir = tmp_path
    (outdir / "FORCE_CONSTANTS_2ND").write_text("2\n", encoding="utf-8")
    result = {
        "export_fc2_text": True,
        "force_constants_text_exported": True,
        "n_atoms_supercell": 2,
        "output_files": {
            "shengbte_force_constants_2nd": "FORCE_CONSTANTS_2ND",
        },
    }

    checks = []
    validator._validate_fc2_text_exports(outdir, result, lambda name, passed, message: checks.append(passed))

    assert checks
    assert all(checks)
