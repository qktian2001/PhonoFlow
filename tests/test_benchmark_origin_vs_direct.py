from __future__ import annotations

import importlib.util
from pathlib import Path


def _load_script_module():
    script_path = Path(__file__).resolve().parents[1] / "scripts" / "benchmark_origin_vs_direct.py"
    spec = importlib.util.spec_from_file_location("benchmark_origin_vs_direct", script_path)
    assert spec is not None
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_build_plan_creates_nep_and_dpa_direct_origin_runs(tmp_path: Path) -> None:
    module = _load_script_module()

    plan = module.build_plan(
        root=Path("/repo"),
        python=Path("/env/bin/python"),
        outdir=tmp_path,
        input_path=Path("examples/Si.vasp"),
        nep_model_path=Path("/repo/nep89/nep.txt"),
        dpa_model_path=Path("/repo/models/dpa.pt"),
    )

    assert [run["name"] for run in plan["runs"]] == [
        "nep89_direct",
        "nep89_origin",
        "dpa4neo_direct",
        "dpa4neo_origin",
    ]
    origin = plan["runs"][1]["command"]
    assert origin[origin.index("--force-parallel-backend") + 1] == "origin"
    assert origin[origin.index("--backend") + 1] == "calorine"
    dpa = plan["runs"][2]["command"]
    assert dpa[dpa.index("--backend") + 1] == "dpa4neo"
    assert dpa[dpa.index("--deepmd-device") + 1] == "cpu"
