# CLI Reference

This page describes the local command-line surface implemented in
`src/phonoflow/cli.py`. Use the runtime help for the exact option parser:

```bash
phonoflow --help
phonoflow --help-all
phonoflow single --help
phonoflow compare-models --help
```

`--help-all` prints the current `WorkflowConfig` fields, defaults, and short
purposes. The full parameter table is maintained in
[Configuration Reference](configuration.md).

## Command Overview

| Command | Purpose |
| --- | --- |
| `phonoflow version` | Print the package version. |
| `phonoflow doctor --verbose` | Check runtime packages and optional backend availability. |
| `phonoflow init-config --out config.yaml` | Write a complete editable YAML configuration template. |
| `phonoflow single` | Run one single-structure workflow from CLI options and/or a YAML config. |
| `phonoflow run` | Direct production-friendly alias for a single workflow with automatic defaults. |
| `phonoflow compare-models` | Run one structure through multiple model/backend definitions. |
| `phonoflow read-result --path result.json` | Print a concise summary of a previous result. |
| `phonoflow batch` | Batch workflow skeleton for local CLI experiments. |

## Runtime Checks

Run these after installation or after changing optional backends:

```bash
phonoflow version
phonoflow doctor --verbose
phonoflow --help-all
```

`doctor` checks core packages plus optional Calorine, Phono3py, HiPhive, and
DeepMD availability. It does not run a heavy scientific calculation.

## Config Template

```bash
phonoflow init-config --out config.yaml
phonoflow single --config config.yaml --dry-run --print-config
```

The generated YAML includes every `WorkflowConfig` field. CLI options override
YAML values when both are provided.

## Single-Structure Harmonic Runs

NEP/NEP89 through Calorine:

```bash
phonoflow run \
  --input-path examples/Si.vasp \
  --model-path /path/to/nep-model.txt \
  --backend calorine \
  --outdir work/si_nep_phonon \
  --supercell-dim auto \
  --mesh auto \
  --overwrite
```

The same workflow through a YAML file:

```bash
phonoflow single \
  --config config.yaml \
  --input-path examples/Si.vasp \
  --model-path /path/to/nep-model.txt \
  --backend calorine \
  --outdir work/si_nep_phonon \
  --overwrite
```

Useful control flags:

- `--dry-run`: resolve settings and write metadata without force/phonon work.
- `--print-config`: print resolved settings.
- `--overwrite`: intentionally reuse an existing output directory.
- `--resume`: reuse a complete successful result where supported.
- `--no-relax`: skip relaxation.
- `--relax-cell` / `--no-relax-cell`: choose cell+position or position-only
  relaxation.

## Harmonic Defaults

- `--supercell-dim auto` uses `--target-supercell-length 15.0`, multiplier
  bounds `1..6`, and `--max-supercell-atoms 200`.
- `--mesh auto` resolves the harmonic/DOS mesh. Bulk structures use `21 21 21`;
  detected 2D slabs use a denser in-plane mesh such as `51 51 1`.
- `--kappa-mesh` is independent and is not affected by `--mesh`.
- Cubic/equal active lattice lengths keep equal auto-supercell multipliers when
  the atom cap permits it.

## Thermal Conductivity

Finite-displacement FC3 with RTA:

```bash
phonoflow single \
  --input-path examples/Si.vasp \
  --model-path /path/to/nep-model.txt \
  --backend calorine \
  --outdir work/si_nep_rta \
  --compute-kappa \
  --fc3-method finite-displacement \
  --fc3-supercell-dim auto \
  --kappa-mesh auto \
  --method rta \
  --temperatures 300 \
  --force-workers 24 \
  --force-parallel-backend process \
  --force-chunk-size 1 \
  --overwrite
```

LBTE uses the same FC2/FC3 force constants but a more expensive thermal solver:

```bash
phonoflow single \
  --input-path examples/Si.vasp \
  --model-path /path/to/nep-model.txt \
  --backend calorine \
  --outdir work/si_nep_lbte \
  --compute-kappa \
  --fc3-method finite-displacement \
  --kappa-mesh 11 11 11 \
  --method lbte \
  --temperatures 300 \
  --force-workers 24 \
  --force-parallel-backend process \
  --force-chunk-size 1 \
  --overwrite
```

Thermal defaults:

- `--fc3-supercell-dim auto` uses `--fc3-target-supercell-length 10.0` and
  `--max-fc3-supercell-atoms 200`.
- `--kappa-mesh auto` resolves to `11 11 11`.
- `--max-fc3-displacements` is only a smoke-test cap and should not be used for
  converged production calculations.
- `--method rta` is faster; `--method lbte` is usually slower and depends more
  strongly on Phono3py/BLAS/OpenMP behavior.

## HiPhive FC3 Fitting

Use HiPhive only when the optional package is installed:

```bash
phonoflow single \
  --input-path examples/Si.vasp \
  --model-path /path/to/nep-model.txt \
  --backend calorine \
  --outdir work/si_nep_hiphive \
  --compute-kappa \
  --fc3-method hiphive \
  --n-structures 200 \
  --rattle-std 0.02 \
  --cutoffs 5.0 4.0 \
  --kappa-mesh auto \
  --method rta \
  --overwrite
```

## DeepMD And DPA

DeepMD/DPA backends require a compatible local DeepMD-kit installation. DPA
aliases include `dpa31`, `dpa32`, `dpa33`, and `dpa4neo`; `dpa3` and `dpa4`
resolve to current canonical aliases.

CPU-oriented DPA example:

```bash
phonoflow single \
  --input-path examples/Si.vasp \
  --backend dpa4neo \
  --model-path /path/to/DPA4-Neo-OMat24-v20260528_rc.pt \
  --outdir work/si_dpa4_rta \
  --no-relax \
  --compute-kappa \
  --fc3-method finite-displacement \
  --kappa-mesh auto \
  --method rta \
  --deepmd-device cpu \
  --deepmd-torch-threads 1 \
  --force-workers 24 \
  --force-parallel-backend process \
  --force-chunk-size 1 \
  --overwrite
```

When many process workers are used on CPU, keep `--deepmd-torch-threads 1` to
avoid each worker creating its own large thread pool.

## Local CPU Force Scheduling

The finite-displacement force loop can run in several modes:

| Backend | Use case |
| --- | --- |
| `origin` | Preserve the original phono3py-style FC3 force loop for comparison/debugging. |
| `direct` | Run PhonoFlow's no-process path. Useful for validating numerical equivalence. |
| `serial` | Default one-process path. |
| `process` | Process-level parallelism with calculator reuse and bounded scheduling. Recommended for NEP/Calorine and CPU DPA force evaluation. |
| `worker_queue` | Prototype worker-pull queue path for fine-grained load balancing experiments. |

Recommended CPU finite-displacement settings:

```bash
--force-workers 24 \
--force-parallel-backend process \
--force-chunk-size 1
```

`force_workers` is a concurrency cap. If FC3 has 1000 displaced structures and
`force_workers=24`, PhonoFlow evaluates at most 24 force tasks concurrently and
the remaining tasks wait in the scheduler queue.

## Optional Local CPU Queue

The local CPU queue is disabled by default. Enable it when multiple CLI jobs
share one machine and you want PBS/Slurm-style slot accounting:

```bash
phonoflow single \
  --input-path examples/Si.vasp \
  --backend dummy \
  --outdir work/cpu_queue_smoke \
  --cpu-queue \
  --cpu-queue-total-slots 36 \
  --cpu-queue-max-running-jobs 2 \
  --cpu-queue-job-slots 18 \
  --force-parallel-backend process \
  --overwrite
```

If `--force-workers` is omitted, an acquired queue lease can set the effective
force-worker budget. If `--force-workers` is explicitly set, it remains an upper
bound and will not exceed the allocated queue slots.

## Compare Models

```bash
phonoflow compare-models \
  --input-path examples/Si.vasp \
  --outdir work/compare \
  --model-label nep_a --backend calorine --model-path /path/to/nep-a.txt \
  --model-label nep_b --backend calorine --model-path /path/to/nep-b.txt \
  --mesh auto \
  --overwrite
```

Thermal compare-model run with shared CPU settings:

```bash
phonoflow compare-models \
  --input-path examples/Si.vasp \
  --outdir work/compare_kappa \
  --model-label nep --backend calorine --model-path /path/to/nep.txt \
  --model-label dpa4 --backend dpa4neo --model-path /path/to/dpa4.pt \
  --compute-kappa \
  --fc3-method finite-displacement \
  --kappa-mesh auto \
  --method rta \
  --deepmd-device cpu \
  --deepmd-torch-threads 1 \
  --force-workers 24 \
  --force-parallel-backend process \
  --force-chunk-size 1 \
  --overwrite
```

## Read Existing Results

```bash
phonoflow read-result --path work/si_nep_rta/result.json
```

This prints a concise summary from an existing `result.json` without rerunning
the calculation.
