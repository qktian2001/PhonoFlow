from __future__ import annotations

from phonoflow.runtime_warnings import filter_optional_deepmd_cuda_probe_warnings


def test_filter_optional_deepmd_cuda_probe_warnings_keeps_real_errors() -> None:
    raw = "\n".join(
        [
            "DeePMD-kit: Cannot find libcudart.so.12",
            "DeePMD-kit: Error message: libcudart.so.12: cannot open shared object file",
            "implib-gen: libcudart.so.12: failed to resolve symbol '__cudaRegisterFatBinary'",
            "RuntimeError: real model failure",
            "",
        ]
    )

    filtered, suppressed = filter_optional_deepmd_cuda_probe_warnings(raw)

    assert len(suppressed) == 3
    assert "libcudart" not in filtered
    assert "RuntimeError: real model failure" in filtered
