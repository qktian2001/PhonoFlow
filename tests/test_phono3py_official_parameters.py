from __future__ import annotations

from pathlib import Path

from phonoflow.compare_models import _model_command
from phonoflow.config import WorkflowConfig
from phonoflow.thermal.fc3_finite_displacement import _apply_phono3py_symmetrize_fc2


ROOT = Path(__file__).resolve().parents[1]


def test_workflow_config_uses_new_symmetrize_fc2_field_and_maps_legacy_alias() -> None:
    fields = set(WorkflowConfig.model_fields)
    assert "phono3py_symmetrize_fc2" in fields
    assert "phono3py_symmetrize_fc3" in fields
    assert "phono3py_fc2_asr" not in fields

    config = WorkflowConfig(phono3py_fc2_asr=True)
    assert config.phono3py_symmetrize_fc2 is True


def test_cli_help_exposes_new_fc2_name_and_marks_legacy_alias_deprecated() -> None:
    cli_source = (ROOT / "src/phonoflow/cli.py").read_text(encoding="utf-8")
    assert "--phono3py-symmetrize-fc2" in cli_source
    assert "--phono3py-symmetrize-fc3" in cli_source
    assert "--phono3py-fc2-asr" in cli_source
    assert "Deprecated alias" in cli_source


def test_compare_child_command_uses_new_symmetrize_fc2_option() -> None:
    command = _model_command(
        input_path=Path("POSCAR"),
        outdir=Path("out"),
        backend="deepmd",
        compute_kappa=True,
        relax=False,
        dry_run=False,
        overwrite=True,
        phono3py_symmetrize_fc2=True,
        phono3py_symmetrize_fc3=False,
    )
    assert "--phono3py-symmetrize-fc2" in command
    assert "--no-phono3py-symmetrize-fc3" in command
    assert "--phono3py-fc2-asr" not in command
    assert "--no-phono3py-fc2-asr" not in command


def test_result_serialization_uses_new_symmetrize_fields() -> None:
    data = WorkflowConfig(phono3py_symmetrize_fc2=True, phono3py_symmetrize_fc3=True).to_dict()
    assert data["phono3py_symmetrize_fc2"] is True
    assert data["phono3py_symmetrize_fc3"] is True
    assert "phono3py_fc2_asr" not in data


def test_parameter_chain_docs_keep_old_name_only_as_deprecated_compatibility() -> None:
    for rel in ("docs/parameter_chain_audit.md", "docs/parameter_chain_audit_zh.md"):
        text = (ROOT / rel).read_text(encoding="utf-8")
        before_compatibility = text.split("Deprecated", 1)[0].split("已废弃", 1)[0]
        assert "phono3py_symmetrize_fc2" in text
        assert "phono3py_fc2_asr" not in before_compatibility
        if "phono3py_fc2_asr" in text:
            window = text[max(0, text.index("phono3py_fc2_asr") - 120) : text.index("phono3py_fc2_asr") + 160]
            assert "deprecated" in window.lower() or "已废弃" in window


def test_thermal_fc2_symmetrization_flag_calls_official_method() -> None:
    class FakePhono3py:
        def __init__(self) -> None:
            self.fc2 = [[[[0.0]]]]
            self.called = False

        def symmetrize_fc2(self) -> None:
            self.called = True

    fake = FakePhono3py()
    info = _apply_phono3py_symmetrize_fc2(fake, enabled=True)
    assert fake.called is True
    assert info["phono3py_symmetrize_fc2"] is True
    assert info["phono3py_symmetrize_fc2_applied"] is True