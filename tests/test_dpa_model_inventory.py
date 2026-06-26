from __future__ import annotations

from pathlib import Path

import pytest

from phonoflow.defaults import (
    DPA_MODEL_REGISTRY,
    discover_bundled_dpa_models,
    resolve_dpa_model_path,
)
from phonoflow.exceptions import ConfigError


EXPECTED_MODELS = {
    "dpa31": "DPA-3.1-3M.pt",
    "dpa32": "DPA-3.2-5M.pt",
    "dpa33": "DPA-3.3-1M.pt",
    "dpa4neo": "DPA4-Neo-OMat24-v20260528_rc.pt",
}
EXPECTED_HEADS = {
    "dpa31": "Omat24",
    "dpa32": "OMat24",
    "dpa33": "Omat24",
    "dpa4neo": None,
}


def test_registry_uses_latest_four_dpa_models() -> None:
    assert {key: value["filename"] for key, value in DPA_MODEL_REGISTRY.items()} == EXPECTED_MODELS
    assert {key: value["model_head"] for key, value in DPA_MODEL_REGISTRY.items()} == EXPECTED_HEADS
    assert all("Pro-MPtrj" not in value["filename"] for value in DPA_MODEL_REGISTRY.values())


def test_latest_four_dpa_models_are_discovered(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    for filename in EXPECTED_MODELS.values():
        (tmp_path / filename).write_bytes(filename.encode("utf-8"))
    monkeypatch.setattr("phonoflow.defaults.DEFAULT_DPA_MODEL_DIR", tmp_path)

    inventory = discover_bundled_dpa_models()

    assert {item["alias"]: item["filename"] for item in inventory} == EXPECTED_MODELS
    assert all(item["available"] for item in inventory)
    assert all(Path(item["path"]).exists() for item in inventory)


@pytest.mark.parametrize("alias,filename", EXPECTED_MODELS.items())
def test_each_latest_dpa_alias_resolves_exact_filename(
    alias: str,
    filename: str,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    model_path = tmp_path / filename
    model_path.write_bytes(b"model")
    monkeypatch.setattr("phonoflow.defaults.DEFAULT_DPA_MODEL_DIR", tmp_path)

    resolved = resolve_dpa_model_path(alias, None)

    assert resolved.backend_alias == alias
    assert resolved.model_name == filename
    assert resolved.model_path == model_path


def test_legacy_aliases_resolve_to_current_defaults(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    dpa32 = tmp_path / EXPECTED_MODELS["dpa32"]
    dpa4neo = tmp_path / EXPECTED_MODELS["dpa4neo"]
    dpa32.write_bytes(b"dpa32")
    dpa4neo.write_bytes(b"dpa4neo")
    monkeypatch.setattr("phonoflow.defaults.DEFAULT_DPA_MODEL_DIR", tmp_path)

    assert resolve_dpa_model_path("dpa3", None).model_path == dpa32
    assert resolve_dpa_model_path("dpa4", None).model_path == dpa4neo


def test_missing_latest_model_does_not_fallback_to_old_dpa4(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    (tmp_path / "DPA-4.0-Pro-MPtrj.pt").write_bytes(b"old")
    monkeypatch.setattr("phonoflow.defaults.DEFAULT_DPA_MODEL_DIR", tmp_path)

    with pytest.raises(ConfigError, match="DPA4-Neo-OMat24-v20260528_rc.pt"):
        resolve_dpa_model_path("dpa4", None)
