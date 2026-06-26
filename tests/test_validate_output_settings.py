import json
from pathlib import Path


def test_validate_settings_helper_accepts_settings_files(tmp_path: Path):
    from importlib import util

    script_path = Path(__file__).resolve().parents[1] / "scripts" / "validate_output.py"
    spec = util.spec_from_file_location("validate_output", script_path)
    validator = util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(validator)

    (tmp_path / "resolved_settings.json").write_text(
        json.dumps({"backend_resolved": {"value": "calorine", "source": "auto", "note": ""}}),
        encoding="utf-8",
    )
    (tmp_path / "resolved_settings.yaml").write_text("backend_resolved:\n  value: calorine\n", encoding="utf-8")
    (tmp_path / "run_command.txt").write_text("python -m phonoflow run\n", encoding="utf-8")
    result = {
        "settings_summary": {"backend_resolved": {"value": "calorine", "source": "auto", "note": ""}},
        "input_file_hash": "a" * 64,
        "model_file_hash": "b" * 64,
        "software_versions": {"PhonoFlow": "0.2.0a9"},
    }
    checks = []
    validator._validate_settings_exports(tmp_path, result, lambda name, passed, message: checks.append(passed))
    assert checks
    assert all(checks)
