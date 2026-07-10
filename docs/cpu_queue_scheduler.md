# CPU Queue Scheduler

PhonoFlow has an optional local CPU-slot queue for PBS/Slurm-style resource
accounting on one host. It is disabled by default and does not change normal
CLI calculations unless explicitly enabled by the caller.

## Concepts

- `total_cpu_slots`: total CPU slots managed by the local queue.
- `cpu_queue_job_slots`: CPU slots one job requests from the queue.
- `cpu_queue_max_running_jobs`: maximum jobs allowed to hold leases at the same
  time.
- `force_workers`: force-evaluation workers used inside one PhonoFlow job.

`total_cpu_slots` and `force_workers` are deliberately separate. The queue first
allocates a lease, then PhonoFlow maps the allocated slots to an upper bound for
per-job force workers.

## CLI

Example:

```bash
python -m phonoflow single \
  --input-path examples/Si.vasp \
  --backend dummy \
  --outdir work/cpu_queue_example \
  --cpu-queue \
  --cpu-queue-total-slots 36 \
  --cpu-queue-max-running-jobs 2 \
  --cpu-queue-job-slots 18 \
  --force-parallel-backend process \
  --overwrite
```

If `--force-workers` is not explicitly set, the CLI uses the allocated queue
slots as the force-worker count. If `--force-workers` is explicitly set, it is
treated as a conservative upper bound and will not exceed the allocated slots.

For a single fastest local job on a 36-core machine:

```bash
phonoflow single \
  --input-path examples/Si.vasp \
  --backend dummy \
  --outdir work/cpu_queue_single \
  --cpu-queue \
  --cpu-queue-total-slots 36 \
  --cpu-queue-max-running-jobs 1 \
  --cpu-queue-job-slots 36 \
  --force-parallel-backend process \
  --overwrite
```

For two concurrent local jobs on the same 36-core machine, use
`--cpu-queue-max-running-jobs 2` and `--cpu-queue-job-slots 18`.

## State Directory

The queue uses a file lock and a JSON state file under `cpu_queue_state_dir`.
When the field is omitted, the scheduler chooses a local state directory inside
the run context. Set `--cpu-queue-state-dir` explicitly when several independent
CLI processes should coordinate through the same queue.

## Notes

- The queue controls job admission and slot accounting; it does not change
  Phonopy, Phono3py, or model physics.
- `force_workers` controls finite-displacement force concurrency inside an
  admitted job.
- DeepMD/DPA CPU jobs often need `--deepmd-torch-threads 1` when
  `force_workers` is large, otherwise worker processes may oversubscribe CPU
  threads.
