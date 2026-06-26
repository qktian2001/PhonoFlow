from __future__ import annotations

import subprocess
import sys


def _run_module_help(module: str, subcommand: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, "-m", module, subcommand, "--help"],
        check=False,
        capture_output=True,
        text=True,
    )


def test_phonoflow_module_run_and_single_help() -> None:
    for subcommand in ("run", "single"):
        result = _run_module_help("phonoflow", subcommand)
        assert result.returncode == 0, result.stderr
        assert subcommand in result.stdout


def test_legacy_source_package_directory_removed() -> None:
    repo_root = __import__("pathlib").Path(__file__).resolve().parents[1]
    legacy_package = "nep" + "phononflow"
    assert not (repo_root / "src" / legacy_package).exists()
    assert (repo_root / "src" / "phonoflow").exists()
