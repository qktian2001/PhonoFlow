"""Shared thermal workflow helpers."""

from __future__ import annotations

from typing import Any


def disabled_thermal_result(reason: str = "Thermal conductivity calculation was not requested.") -> dict[str, Any]:
    """Return the stable result schema used when kappa is disabled or unavailable."""

    return {
        "enabled": False,
        "available": False,
        "reason": reason,
    }


def unavailable_thermal_result(
    *,
    enabled: bool = True,
    reason: str,
    warnings: list[str] | None = None,
    fc3_method: str | None = None,
    kappa_method: str | None = None,
    **metadata: Any,
) -> dict[str, Any]:
    """Return a non-fatal unavailable thermal result."""

    result: dict[str, Any] = {
        "enabled": bool(enabled),
        "available": False,
        "reason": reason,
        "warnings": list(warnings or []),
    }
    if fc3_method:
        result["fc3_method"] = fc3_method
    if kappa_method:
        result["kappa_method"] = kappa_method
        result["solver_flags"] = ["--method", kappa_method]
        result["method_flags"] = ["--method", kappa_method]
    result.update(metadata)
    return result
