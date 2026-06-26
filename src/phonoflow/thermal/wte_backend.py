"""WTE backend capability detection for phono3py v4."""

from __future__ import annotations

from importlib import import_module
from importlib import metadata as importlib_metadata
from importlib.util import find_spec
import sys
from typing import Any


WTE_INSTALLATION_HINT = (
    "Install from source in this exact Python environment: "
    "`git clone https://github.com/MSimoncelli/phono3py-wte.git .codex/vendor/phono3py-wte`, "
    "apply the phonopy 4.1 compatibility import documented in `docs/wte_installation.md`, "
    "then run `python -m pip install -e .codex/vendor/phono3py-wte` and restart the Web service."
)


def _find_module(name: str) -> bool:
    return find_spec(name) is not None


def _import_module(name: str) -> Any:
    return import_module(name)


def get_wte_backend_capability() -> dict[str, Any]:
    """Return server-side WTE backend availability without running a calculation."""

    phono3py_version = _package_version("phono3py")
    phonopy_version = _package_version("phonopy")
    distribution_version = _package_version("phono3py-wte")
    module_status = {
        "wte": _find_module("wte"),
        "phono3py_wte": _find_module("phono3py_wte"),
    }
    importable = False
    import_reason = None
    module_name = None
    for candidate in ("wte", "phono3py_wte"):
        if not module_status[candidate]:
            continue
        try:
            _import_module(candidate)
            importable = True
            module_name = candidate
            break
        except Exception as exc:  # pragma: no cover - environment dependent
            import_reason = f"{candidate}: {type(exc).__name__}: {exc}"
    reason = None
    if not importable:
        missing_modules = [name for name, found in module_status.items() if not found]
        reason = (
            "WTE is unavailable in "
            f"`{sys.executable}` because the required plugin modules are not importable. "
            f"Missing module checks: {', '.join(missing_modules) or 'none found but import failed'}. "
            f"{WTE_INSTALLATION_HINT}"
        )
        if import_reason:
            reason = f"{reason} Import failed: {import_reason}"
    else:
        missing_modules = [name for name, found in module_status.items() if not found]
    return {
        "available": bool(importable),
        "backend": "phono3py_wte_plugin" if importable else None,
        "transport_type": "WTE" if importable else None,
        "module_name": module_name,
        "wte_module_found": bool(module_status["wte"]),
        "phono3py_wte_module_found": bool(module_status["phono3py_wte"]),
        "wte_importable": bool(importable),
        "phono3py_wte_distribution_version": distribution_version,
        "phono3py_version": phono3py_version,
        "phonopy_version": phonopy_version,
        "python_executable": sys.executable,
        "sys_path": list(sys.path),
        "missing_modules": missing_modules,
        "installation_hint": None if importable else WTE_INSTALLATION_HINT,
        "reason": reason,
    }


def _package_version(name: str) -> str | None:
    try:
        return importlib_metadata.version(name)
    except importlib_metadata.PackageNotFoundError:
        return None
