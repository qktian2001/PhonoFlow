# DeepMD/DPA backend

Date: 2026-06-17

PhonoFlow now supports DeepMD/DPA-family model aliases through the existing
phonopy/phono3py workflow.

## Backend aliases

- `deepmd`
- `dpa`
- `dpa3`
- `dpa4`
- `dpa31`
- `dpa32`
- `dpa33`
- `dpa4neo`

All aliases resolve internally to `DeepMDBackend`, while result metadata keeps
the user-requested alias.

## Recommended DPA smoke commands

```bash
python -m phonoflow single \
  --input-path examples/Si.vasp \
  --backend dpa3 \
  --outdir results/dpa_deepmd_audit_YYYYMMDD_HHMMSS/dpa3_simple \
  --compute-kappa \
  --overwrite

python -m phonoflow single \
  --input-path examples/Si.vasp \
  --backend dpa4 \
  --outdir results/dpa_deepmd_audit_YYYYMMDD_HHMMSS/dpa4_simple \
  --compute-kappa \
  --overwrite
```

The current aliases auto-resolve the bundled models:

- `dpa31` -> `models/DPA-3.1-3M.pt`
- `dpa32` / `dpa3` -> `models/DPA-3.2-5M.pt`
- `dpa33` -> `models/DPA-3.3-1M.pt`
- `dpa4neo` / `dpa4` -> `models/DPA4-Neo-OMat24-v20260528_rc.pt`

The generic `dpa` alias requires `--model-path`.

DPA4-Neo is the current lightweight, fast DPA4 option. The old DPA4-Pro model
is not the default and is not exposed as a Web preset.

The expanded official-like command is:

```bash
python -m phonoflow single \
  --input-path examples/Si.vasp \
  --model-path models/DPA4-Neo-OMat24-v20260528_rc.pt \
  --outdir results/Si_dpa4_official_like \
  --backend dpa4neo \
  --supercell-dim 2 2 2 \
  --fc3-supercell-dim 2 2 2 \
  --displacement 0.03 \
  --fc3-displacement 0.03 \
  --compute-kappa \
  --fc3-method finite-displacement \
  --method rta \
  --temperatures 300 \
  --kappa-mesh 11 11 11 \
  --primitive-matrix P \
  --phono3py-symprec 1e-5 \
  --phono3py-cutoff-frequency 0.0001 \
  --phono3py-plusminus auto \
  --phono3py-diagonal \
  --phono3py-symmetry \
  --phono3py-mesh-symmetry \
  --no-isotope \
  --boundary-mfp 0 \
  --cutoff-pair-distance 0 \
  --deepmd-deterministic \
  --deepmd-reuse-calculator \
  --save-force-audit \
  --phono3py-symmetrize-fc2 \
  --overwrite
```

## Reproducibility notes

- `--deepmd-reuse-calculator` is recommended and enabled by default in config.
- All bundled DPA aliases keep NEP89's formal geometry defaults unless there is a
  deliberate DPA-specific reason to diverge: auto FC2 supercell inference,
  `target_supercell_length=15`, `fc3_target_supercell_length=10`, auto FC3
  supercell inference, auto kappa mesh,
  default FC2/FC3 displacements, inherited phono3py symprec, and no default
  phono3py cutoff-frequency override.
- Smoke-test commands may pass explicit small settings such as
  `--supercell-dim 2 2 2`, `--fc3-supercell-dim 2 2 2`, and
  `--kappa-mesh 11 11 11`. Those are validation-command choices, not formal
  defaults.
- The intentional DPA differences are `deepmd_device=cpu`,
  `deepmd_deterministic=True`, calculator reuse, force audit output, and
  `phono3py_symmetrize_fc2=True`.
- DPA defaults skip relaxation. If `--relax` is explicitly requested, PhonoFlow
  uses NEP89/Calorine relaxation by default; DPA relaxation requires
  `--relax-backend dpa` or `--allow-dpa-relax`.
- `--deepmd-deterministic` pins common thread settings and attempts torch
  deterministic mode. It is an audit aid, not a guarantee of bit-identical DPA4
  forces.
- `--save-force-audit` writes per-displacement force stats, force hashes, and
  raw NPZ arrays for FC2/FC3.
- `--phono3py-symmetrize-fc2` applies ASR to phono3py's FC2 only. It remains off by
  default for NEP89/Calorine, but is enabled by default for DPA aliases.

### Engineering controls are not DPA scientific parameters

- **DeepMD deterministic mode** is a PhonoFlow reproducibility policy. It sets
  single-thread environment variables and best-effort deterministic runtime
  settings around DeepMD/PyTorch inference. It is not an official DPA model
  parameter and does not alter the potential.
- **Reuse DeepMD calculator** is a PhonoFlow/ASE integration performance
  policy. One initialized `deepmd.calculator.DP` instance is cached and reused
  across finite-displacement force evaluations. It is not a DPA scientific
  parameter.
- **Save force audit** is a PhonoFlow diagnostic feature. It writes force
  statistics, hashes, structure hashes, and compressed raw arrays for
  traceability. It is not a DeepMD-kit, DPA, or ASE scientific input.

These controls remain available as CLI overrides and are shown only for
DeepMD/DPA Web selections. They are never passed to NEP89 children. NEP89 does
not create a DeepMD calculator, and compare JSON marks these fields as
`not_applicable` for NEP89.

### FC3 displacement controls

`--fc3-displacement` is the physical atomic displacement amplitude used for
each finite-difference structure (normally 0.03 Å). In contrast,
`--max-fc3-displacements` truncates the number of displaced structures that are
evaluated. The latter is a smoke-test workload cap and must not be interpreted
as a displacement amplitude or a converged production setting.

## Compare models

```bash
python -m phonoflow compare-models \
  --input-path examples/Si.vasp \
  --outdir results/dpa_deepmd_audit_YYYYMMDD_HHMMSS/compare_models \
  --model nep89 \
  --model DPA-3.2-5M.pt \
  --model DPA4-Neo-OMat24-v20260528_rc.pt \
  --compute-kappa \
  --overwrite
```

The comparison command preserves per-model subdirectories and writes
`comparison_summary.csv`, `comparison_summary.md`, `comparison_summary.json`,
`comparison_result.json`, `comparison_kappa_bar.png`,
`comparison_thermal_conductivity.png`, `comparison_phonon_band.png`, and
`comparison_dos.png`. The primary kappa bar chart groups `kxx`, `kyy`, `kzz`,
and `kavg` at 300 K. Every bar has a value label, model axes use short names,
and full model names remain in JSON metadata. The chart adds no success-count
or bar-count prose inside the plot. `comparison_kappa.png` is kept as a legacy
alias. The command also writes a run-folder report set: `report_en.md`,
`report_zh.md`, `summary.json`, `commands.log`, `validation.log`,
`environment.json`, and artifact indexes.

## Expected outputs

The harmonic route writes `force_constants.hdf5`, `FORCE_CONSTANTS_2ND`,
`phonopy.yaml`, `band.yaml`, band/DOS/group-velocity CSV and PNG files.

When `--compute-kappa` is enabled, the thermal route writes `fc2.hdf5`,
`fc3.hdf5`, `phono3py_params.yaml`, `kappa-m*.hdf5`,
`thermal_conductivity.csv`, and `thermal_conductivity.png`.

With `--save-force-audit`, the workflow writes:

- `fd_fc2_forces_stats.csv`
- `fd_fc2_force_hashes.csv`
- `fd_fc2_forces_raw.npz`
- `fd_fc3_forces_stats.csv`
- `fd_fc3_force_hashes.csv`
- `fd_fc3_forces_raw.npz`

Every run also writes `structure_provenance.json`, embedded in `result.json`.

## Interpreting repeatability

DPA4 force and FC hashes may differ across repeated runs because the SeZM force
kernel can be non-bit-deterministic. That does not automatically invalidate the
kappa result. The repeatability criterion should compare force/FC hashes,
Gamma acoustic residuals, gamma statistics, and `kxx/kyy/kzz/kavg`. If hashes
differ but kappa stays stable with FC2 force-constant symmetrization enabled, report both facts.

## Comparing with NEP89

Use the same input or relaxed structure, same supercell dimensions, same
displacements, same phono3py settings, and same kappa mesh. Do not mix a
relaxed structure from one backend with force constants from another backend
without recording structure provenance and treating the result as a separate
cross-backend experiment.
