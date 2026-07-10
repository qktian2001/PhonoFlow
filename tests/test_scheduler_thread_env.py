from __future__ import annotations

import os

from phonoflow_scheduler.thread_env import THREAD_ENV_VARS, apply_thread_env, build_thread_env


def test_build_thread_env_sets_all_thread_pool_variables() -> None:
    env = build_thread_env(3)

    for name in THREAD_ENV_VARS:
        assert env[name] == "3"
    assert env["PHONOFLOW_THREAD_POOL_LIMIT"] == "3"


def test_build_thread_env_clamps_invalid_values_to_one() -> None:
    env = build_thread_env(0)

    for name in THREAD_ENV_VARS:
        assert env[name] == "1"


def test_apply_thread_env_updates_process_environment(monkeypatch) -> None:
    monkeypatch.delenv("OMP_NUM_THREADS", raising=False)

    apply_thread_env({"OMP_NUM_THREADS": "2"})

    assert os.environ["OMP_NUM_THREADS"] == "2"
