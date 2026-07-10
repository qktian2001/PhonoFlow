# CPU Queue Scheduler

PhonoFlow has an optional local CPU-slot queue for PBS/Slurm-style resource
accounting on one host. It is disabled by default and does not change existing
CLI calculations unless explicitly enabled by the CLI caller.

## Concepts

- `total_cpu_slots`: total CPU slots managed by the local queue.
- `cpu_queue_job_slots`: CPU slots one job requests from the queue.
- `max_running_jobs`: maximum jobs allowed to hold leases at the same time.
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
  --model-path examples/Si.vasp \
  --outdir work/cpu_queue_example \
  --cpu-queue \
  --cpu-queue-total-slots 24 \
  --cpu-queue-max-running-jobs 1 \
  --cpu-queue-job-slots 24
```

If `--force-workers` is not explicitly set, the CLI uses the allocated queue
slots as the force-worker count. If `--force-workers` is explicitly set, it is
treated as a conservative upper bound and will not exceed the allocated slots.

## Notes

The CPU queue is a local CLI coordination layer. It reserves file-locked CPU
slots before a workflow starts, then releases the lease when the workflow exits.
It is not a scientific model parameter and does not change phonopy, phono3py,
calculator forces, or thermal-conductivity formulas.
