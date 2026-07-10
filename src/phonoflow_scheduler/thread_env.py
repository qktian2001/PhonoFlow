"""Thread-pool environment helpers for CPU-bound scheduler subprocesses."""

from __future__ import annotations

import os

from phonoflow_scheduler.config import clean_positive_int


THREAD_ENV_VARS = [
    "OMP_NUM_THREADS",
    "MKL_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
    "DP_INTRA_OP_PARALLELISM_THREADS",
    "DP_INTER_OP_PARALLELISM_THREADS",
    "TF_INTRA_OP_PARALLELISM_THREADS",
    "TF_INTER_OP_PARALLELISM_THREADS",
]


def build_thread_env(deepmd_torch_threads: int | None = 1) -> dict[str, str]:
    """Return env vars that cap BLAS/OpenMP/DeepMD/TensorFlow thread pools."""

    value = str(clean_positive_int(deepmd_torch_threads or 1))
    env = {name: value for name in THREAD_ENV_VARS}
    env["PHONOFLOW_THREAD_POOL_LIMIT"] = value
    return env


def apply_thread_env(env: dict[str, str]) -> None:
    """Apply thread-pool env vars to the current process."""

    for key, value in env.items():
        os.environ[str(key)] = str(value)
