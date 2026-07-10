# FC3 Displacement Acceleration Notes

Large FC3 jobs can contain hundreds or thousands of displaced supercells. Each
displacement force calculation is independent, so PhonoFlow can parallelize the
force-evaluation stage without changing the final physics.

## Current Strategy

The current scheduler uses process-level parallelism:

1. PhonoFlow generates FC3 displaced supercells with phono3py.
2. `force_eval.py` wraps each displaced structure in a `ForceTask(index, payload)`.
3. `phonoflow_scheduler.process_pool.evaluate_force_tasks()` groups tasks into
   small chunks.
4. A `ProcessPoolExecutor` evaluates bounded chunks instead of submitting all
   displacements at once.
5. Each worker initializes its calculator once and reuses it for multiple tasks.
6. Results are sorted by displacement index before forces are returned to
   phono3py.

This improves FC3 throughput mainly by reducing repeated calculator/model
initialization and by lowering future scheduling overhead for very large
displacement sets.

## What This Optimizes

- Repeated model/calculator creation inside process workers.
- Excessive pending future count for large FC3 displacement lists.
- Pickle/scheduling overhead by sending chunks instead of one future per
  displacement.
- Result ordering consistency for `force_workers=1` and `force_workers>1`.
- CLI resource-budget consistency.

## What This Does Not Optimize Yet

- Model-level batch inference.
- DeepMD/DPA multi-frame inference through a native backend.
- NEP/Calorine multi-structure batch evaluation.
- phono3py thermal solver threading and BLAS tuning.
- Reducing the number of physical FC3 displacements through additional symmetry
  or cutoff choices.

## Why FC3 Can Still Be Slow

If a job has 4000 displaced supercells and `force_workers=24`, at most 24 worker
processes evaluate force tasks at the same time. The remaining displacements wait
in the bounded scheduler queue. This is expected: the CPU budget controls
concurrency, not the total number of queued structures.

The largest remaining costs are usually:

- The model inference time per displaced structure.
- ASE `Atoms` serialization between processes.
- FC3 force-constant assembly.
- phono3py thermal conductivity solver time, especially LBTE on dense meshes.

## Benchmark

Use the lightweight benchmark to compare legacy-style one-future-per-displacement
scheduling with the current chunked worker-reuse scheduler:

```bash
PYTHONPATH=src python scripts/benchmark_force_scheduler.py --tasks 240 --workers 4
```

The script writes `force_scheduler_benchmark.json` into a timestamped directory
under `work/scheduler_benchmarks/`.
