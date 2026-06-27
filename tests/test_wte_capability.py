from __future__ import annotations

import sys
from types import SimpleNamespace

from phonoflow.thermal import wte_backend


def _registered_wte_methods() -> list[str]:
    return ["std-rta", "wte-lbte", "wte-rta"]


def test_wte_capability_accepts_wte_module(monkeypatch) -> None:
    monkeypatch.setattr(wte_backend, "_find_module", lambda name: name == "wte")
    monkeypatch.setattr(
        wte_backend,
        "_import_module",
        lambda name: SimpleNamespace(__name__=name, register=lambda: None),
    )
    monkeypatch.setattr(wte_backend, "_registered_conductivity_methods", _registered_wte_methods)

    capability = wte_backend.get_wte_backend_capability()

    assert capability["available"] is True
    assert capability["module_name"] == "wte"
    assert capability["wte_module_found"] is True
    assert capability["wte_registered"] is True
    assert capability["registered_methods"] == ["std-rta", "wte-lbte", "wte-rta"]


def test_wte_capability_accepts_phono3py_wte_module_when_wte_is_missing(monkeypatch) -> None:
    monkeypatch.setattr(wte_backend, "_find_module", lambda name: name == "phono3py_wte")
    monkeypatch.setattr(
        wte_backend,
        "_import_module",
        lambda name: SimpleNamespace(__name__=name, register=lambda: None),
    )
    monkeypatch.setattr(wte_backend, "_registered_conductivity_methods", _registered_wte_methods)

    capability = wte_backend.get_wte_backend_capability()

    assert capability["available"] is True
    assert capability["module_name"] == "phono3py_wte"
    assert capability["phono3py_wte_module_found"] is True
    assert capability["wte_registered"] is True


def test_wte_capability_requires_registered_wte_methods(monkeypatch) -> None:
    monkeypatch.setattr(wte_backend, "_find_module", lambda name: name == "wte")
    monkeypatch.setattr(
        wte_backend,
        "_import_module",
        lambda name: SimpleNamespace(__name__=name, register=lambda: None),
    )
    monkeypatch.setattr(wte_backend, "_registered_conductivity_methods", lambda: ["std-rta"])

    capability = wte_backend.get_wte_backend_capability()

    assert capability["available"] is False
    assert capability["module_name"] == "wte"
    assert capability["wte_importable"] is True
    assert capability["wte_registered"] is False
    assert capability["registered_methods"] == ["std-rta"]
    assert "wte-rta" in capability["reason"]
    assert "wte-lbte" in capability["reason"]


def test_wte_capability_missing_reason_contains_install_command(monkeypatch) -> None:
    monkeypatch.setattr(wte_backend, "_find_module", lambda name: False)
    monkeypatch.setattr(wte_backend, "_registered_conductivity_methods", lambda: [])

    capability = wte_backend.get_wte_backend_capability()

    assert capability["available"] is False
    assert "python -m pip install" in capability["reason"]
    assert "phono3py" in capability["reason"]
    assert capability["module_name"] is None
    assert capability["python_executable"] == sys.executable
    assert capability["missing_modules"] == ["wte", "phono3py_wte"]
    assert capability["wte_registered"] is False
    assert capability["registered_methods"] == []
    assert "git clone" in capability["installation_hint"]
    assert "restart" in capability["installation_hint"].lower()
