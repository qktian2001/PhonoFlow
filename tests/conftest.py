from __future__ import annotations

from pathlib import Path

import pytest


@pytest.fixture(autouse=True)
def public_model_placeholders(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Provide private-model placeholders so public tests do not need model files."""

    import phonoflow.compare_models as compare_models
    import phonoflow.defaults as defaults

    model_dir = tmp_path / "model_placeholders"
    model_dir.mkdir()
    for spec in defaults.DPA_MODEL_REGISTRY.values():
        (model_dir / str(spec["filename"])).write_text("placeholder\n", encoding="utf-8")

    nep_model = tmp_path / "nep_model.txt"
    nep_model.write_text("placeholder\n", encoding="utf-8")

    monkeypatch.setattr(defaults, "DEFAULT_DPA_MODEL_DIR", model_dir)
    monkeypatch.setattr(defaults, "DEFAULT_NEP89_MODEL_PATH", nep_model)
    monkeypatch.setattr(compare_models, "DEFAULT_NEP89_MODEL_PATH", nep_model)
    compare_models.MODEL_SPECS["nep89"]["model_path"] = nep_model
