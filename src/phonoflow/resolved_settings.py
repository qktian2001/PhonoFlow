"""Resolved workflow settings with source tracking."""

from __future__ import annotations

import json
import shlex
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml
from rich.console import Console
from rich.table import Table


@dataclass(frozen=True)
class ResolvedSetting:
    """One resolved workflow setting."""

    name: str
    value: Any
    source: str
    note: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "value": _jsonable(self.value),
            "source": self.source,
            "note": self.note,
        }


class ResolvedSettings:
    """Collection of resolved workflow settings."""

    def __init__(self, settings: dict[str, ResolvedSetting] | None = None) -> None:
        self.settings = settings or {}

    def add(self, name: str, value: Any, source: str, note: str = "") -> None:
        self.settings[name] = ResolvedSetting(name=name, value=_jsonable(value), source=source, note=note)

    def to_dict(self) -> dict[str, dict[str, Any]]:
        return {name: setting.to_dict() for name, setting in self.settings.items()}

    def write_json(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8") as handle:
            json.dump(self.to_dict(), handle, indent=2)
            handle.write("\n")

    def write_yaml(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8") as handle:
            yaml.safe_dump(self.to_dict(), handle, sort_keys=False, allow_unicode=True)

    def print_table(self, console: Console | None = None) -> None:
        (console or Console()).print(self._table())

    def render_table(self) -> str:
        """Render the settings table as plain terminal text for run artifacts."""

        console = Console(record=True, width=132, color_system=None)
        with console.capture() as capture:
            console.print(self._table())
        return capture.get()

    def write_table(self, path: Path) -> str:
        """Write and return the rendered settings table."""

        text = self.render_table()
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text, encoding="utf-8")
        return text

    def _table(self) -> Table:
        table = Table(title="Resolved PhonoFlow settings")
        table.add_column("Parameter")
        table.add_column("Value")
        table.add_column("Source")
        table.add_column("Note")
        for setting in self.settings.values():
            table.add_row(setting.name, _format_value(setting.value), setting.source, setting.note)
        return table


def build_run_command(argv: list[str] | None, fallback: list[str]) -> str:
    """Return a shell-readable command from argv or a fallback command list."""

    items = argv if argv else fallback
    return " ".join(shlex.quote(str(item)) for item in items)


def _format_value(value: Any) -> str:
    if isinstance(value, (list, dict)):
        return json.dumps(value, ensure_ascii=False)
    if value is None:
        return ""
    return str(value)


def _jsonable(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, tuple):
        return [_jsonable(item) for item in value]
    if isinstance(value, list):
        return [_jsonable(item) for item in value]
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    return value
