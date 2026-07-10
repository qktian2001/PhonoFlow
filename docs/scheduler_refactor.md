# PhonoFlow Scheduler Refactor

This document records the CPU/force-task scheduler split introduced for local
CLI workflows.

## Package Boundary

The scheduler lives in `src/phonoflow_scheduler` and is intentionally separate
from the scientific workflow code in `src/phonoflow`.

- `phonoflow_scheduler.config`: cleaned scheduler dataclasses.
- `phonoflow_scheduler.resources`: CPU budget estimation and recommendations.
- `phonoflow_scheduler.thread_env`: shared BLAS/OpenMP/DeepMD thread env vars.
- `phonoflow_scheduler.force_tasks`: order-preserving force task/result models.
- `phonoflow_scheduler.process_pool`: serial/process force-task execution with
  worker calculator reuse, chunking, and bounded pending futures.
- `phonoflow_scheduler.profiling`: small timing helper for future diagnostics.

The old `phonoflow.resources.estimate_cpu_budget()` API remains available and
now delegates to `phonoflow_scheduler.resources`.

## Core Integration

`src/phonoflow/workflow/force_eval.py` is the compatibility adapter between
PhonoFlow workflows and the independent scheduler. It still returns
`ForceEvaluationResult`, but internally it now builds `ForceTask` objects and
calls `evaluate_force_tasks()`.

The same adapter is used by:

- FC2/phonopy finite-displacement force evaluation.
- FC3/phono3py finite-displacement force evaluation.
- HiPhive training-structure force evaluation.

The scheduler preserves displacement order by carrying a task index through the
process pool and sorting `ForceResult` objects before returning forces.

## Thread Environment

`phonoflow_scheduler.thread_env.build_thread_env()` centralizes the
BLAS/OpenMP/DeepMD thread environment used by CLI subprocesses and worker
processes. The scheduler can set:

- `OMP_NUM_THREADS`
- `MKL_NUM_THREADS`
- `OPENBLAS_NUM_THREADS`
- `NUMEXPR_NUM_THREADS`
- `VECLIB_MAXIMUM_THREADS`
- `DP_INTRA_OP_PARALLELISM_THREADS`
- `DP_INTER_OP_PARALLELISM_THREADS`
- `TF_INTRA_OP_PARALLELISM_THREADS`
- `TF_INTER_OP_PARALLELISM_THREADS`
- `PHONOFLOW_THREAD_POOL_LIMIT`

## CLI/Internal Knobs

The existing options are unchanged:

- `--force-workers`
- `--force-parallel-backend`
- `--deepmd-torch-threads`

Two internal scheduler options are available for local benchmarking and tuning:

- `--force-chunk-size`
- `--force-max-pending-tasks`

Defaults are `None`, meaning the scheduler chooses conservative automatic
values.

## Safety

This refactor does not change phonopy/phono3py displacement generation,
force-constant assembly, thermal conductivity formulas, model parameters, or
physical defaults. It changes only the CPU scheduling layer used to evaluate
many independent force tasks.
