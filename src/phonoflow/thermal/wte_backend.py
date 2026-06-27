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


def _registered_conductivity_methods() -> list[str]:
    try:
        from phono3py.conductivity import factory
    except Exception:  # pragma: no cover - environment dependent
        return []
    return sorted(str(key) for key in getattr(factory, "_REGISTRY", {}).keys())


def _register_wte_plugin(module: Any) -> tuple[bool, str | None]:
    register = getattr(module, "register", None)
    if not callable(register):
        return False, "WTE plugin module does not expose a callable register()."
    try:
        register()
    except Exception as exc:  # pragma: no cover - environment dependent
        return False, f"{type(exc).__name__}: {exc}"
    methods = _registered_conductivity_methods()
    if "wte-rta" not in methods or "wte-lbte" not in methods:
        return (
            False,
            "WTE plugin imported but did not register wte-rta/wte-lbte. "
            f"Registered methods: {methods}",
        )
    return True, None


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
    registration_error = None
    wte_registered = False
    for candidate in ("wte", "phono3py_wte"):
        if not module_status[candidate]:
            continue
        try:
            module = _import_module(candidate)
            importable = True
            module_name = candidate
            wte_registered, registration_error = _register_wte_plugin(module)
            break
        except Exception as exc:  # pragma: no cover - environment dependent
            import_reason = f"{candidate}: {type(exc).__name__}: {exc}"
    registered_methods = _registered_conductivity_methods()
    available = bool(importable and wte_registered)
    reason = None
    if not available:
        missing_modules = [name for name, found in module_status.items() if not found]
        if not importable:
            reason = (
                "WTE is unavailable in "
                f"`{sys.executable}` because the required plugin modules are not importable. "
                f"Missing module checks: {', '.join(missing_modules) or 'none found but import failed'}. "
                f"{WTE_INSTALLATION_HINT}"
            )
            if import_reason:
                reason = f"{reason} Import failed: {import_reason}"
        else:
            reason = (
                "WTE is unavailable because the phono3py-wte module was found but did not "
                f"register wte-rta/wte-lbte in phono3py. {registration_error or ''}"
            ).strip()
            if import_reason:
                reason = f"{reason} Import failed: {import_reason}"
            if missing_modules:
                reason = f"{reason} Missing alternate module checks: {', '.join(missing_modules)}."
            reason = f"{reason} {WTE_INSTALLATION_HINT}"
        if import_reason and "Import failed:" not in reason:
            reason = f"{reason} Import failed: {import_reason}"
    else:
        missing_modules = [name for name, found in module_status.items() if not found]
    return {
        "available": available,
        "backend": "phono3py_wte_plugin" if available else None,
        "transport_type": "WTE" if available else None,
        "module_name": module_name,
        "wte_module_found": bool(module_status["wte"]),
        "phono3py_wte_module_found": bool(module_status["phono3py_wte"]),
        "wte_importable": bool(importable),
        "wte_registered": bool(wte_registered),
        "registered_methods": registered_methods,
        "registration_error": registration_error,
        "phono3py_wte_distribution_version": distribution_version,
        "phono3py_version": phono3py_version,
        "phonopy_version": phonopy_version,
        "python_executable": sys.executable,
        "sys_path": list(sys.path),
        "missing_modules": missing_modules,
        "installation_hint": None if available else WTE_INSTALLATION_HINT,
        "reason": reason,
    }


def _package_version(name: str) -> str | None:
    try:
        return importlib_metadata.version(name)
    except importlib_metadata.PackageNotFoundError:
        return None
