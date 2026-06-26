from __future__ import annotations

import json
from pathlib import Path


def test_inventory_contains_single_run_compare_and_required_fields() -> None:
    data = json.loads(Path("docs/parameter_inventory_full.json").read_text())
    params = data["parameters"]
    scopes = " ".join(item["command_scope"] for item in params)
    assert "single" in scopes
    assert "run" in scopes
    assert "compare-models" in scopes
    required = {"category", "current_phonoflow_default", "official_origin", "stage_used_in"}
    for item in params:
        for field in required:
            assert field in item
            assert item[field] not in (None, "")


def test_deprecated_phono3py_fc2_asr_is_not_formal_parameter() -> None:
    data = json.loads(Path("docs/parameter_inventory_full.json").read_text())
    names = {item["parameter_name"] for item in data["parameters"]}
    assert "phono3py_fc2_asr" not in names
    assert "phono3py_symmetrize_fc2" in names
