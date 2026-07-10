from __future__ import annotations

import importlib.util
from pathlib import Path


def _load_script_module():
    script_path = Path(__file__).resolve().parents[1] / "scripts" / "benchmark_dpa_force_chunks.py"
    spec = importlib.util.spec_from_file_location("benchmark_dpa_force_chunks", script_path)
    assert spec is not None
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_build_plan_sweeps_workers_chunks_and_torch_threads(tmp_path: Path) -> None:
    module = _load_script_module()

    plan = module.build_plan(
        root=Path("/repo"),
        python=Path("/env/bin/python"),
        model_path=Path("/repo/models/dpa.pt"),
        outdir=tmp_path,
        input_path=Path("examples/Si.vasp"),
        backend="dpa4neo",
        workers=[8, 12],
        chunks=[1, 4],
        torch_threads=[1, 2],
        max_fc3_displacements=20,
    )

    assert len(plan["runs"]) == 8
    first = plan["runs"][0]
    command = first["command"]
    assert first["workers"] == 8
    assert first["chunk_size"] == 1
    assert first["deepmd_torch_threads"] == 1
    assert "--backend" in command
    assert command[command.index("--backend") + 1] == "dpa4neo"
    assert command[command.index("--deepmd-device") + 1] == "cpu"
    assert command[command.index("--deepmd-torch-threads") + 1] == "1"
    assert command[command.index("--force-workers") + 1] == "8"
    assert command[command.index("--force-chunk-size") + 1] == "1"
    assert command[command.index("--max-fc3-displacements") + 1] == "20"
    assert command[command.index("--kappa-mesh") + 1 : command.index("--kappa-mesh") + 4] == ["11", "11", "11"]


def test_extracts_current_result_json_force_and_kappa_paths() -> None:
    module = _load_script_module()
    result_json = {
        "thermal_conductivity": {
            "n_fc3_displacements": 181,
            "summary": {
                "300": {
                    "kappa_trace_over_3": 117.5,
                }
            },
            "timing_breakdown": {
                "fc3_wall_time": 24.8,
                "fc3_displacements_per_second": 7.3,
            },
        }
    }

    assert module.extract_kappa(result_json) == 117.5
    assert module.fc3_force_wall_time(result_json, {}) == 24.8
    assert module.fc3_displacements_per_second(result_json) == 7.3
    assert module.n_fc3_displacements(result_json) == 181
