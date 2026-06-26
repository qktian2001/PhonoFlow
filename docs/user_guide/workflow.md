# Workflow

## v0.1 Skeleton Workflow

The v0.1 single workflow validates project plumbing:

1. Parse `WorkflowConfig`.
2. Read the input structure with ASE.
3. Create the output directory.
4. Save `resolved_config.yaml`.
5. Select `dummy`, `calorine`, or `gpumd`.
6. Check backend availability.
7. Write `input_structure.vasp`.
8. Run dummy relaxation or copy the input structure.
9. Write `relaxed.vasp`.
10. Use placeholder frequencies.
11. Analyze dynamic stability.
12. Write `stability_report.json` and `stability_report.txt`.

Batch mode still follows this skeleton path in v0.2-p9.

## v0.2-p9 Auto Real Single Workflow

The recommended real Calorine + Phonopy single workflow can now be started with:

```bash
python -m phonoflow run --input-path POSCAR --model-path nep.txt
```

The auto defaults resolve:

- backend: `auto` resolves to Calorine and never falls back to dummy.
- outdir: `results/<structure>_<backend>`.
- relaxation: `--relax --relax-cell --fmax 1e-5 --max-steps 2000` by default.
- FC2/harmonic supercell: inferred from cell lengths with a target length of 15 Å.
- FC3/thermal supercell: inferred independently with a target length of 10 Å.
- supercell caps: `min_supercell_dim=1`, `max_supercell_dim=6`, and `max_supercell_atoms=1000`.
- primitive matrix: explicit `P` by default.
- force-constant method: `finite-displacement`.
- DOS: enabled with mesh `[20, 20, 20]` when `mesh=auto`.
- ASR and force-constant symmetrization: requested by default.
- FC2 text export: `FORCE_CONSTANTS_2ND` is enabled by default.
- dry-run: `--dry-run` prints and writes resolved settings without running relaxation or Phonopy.
- overwrite policy: existing output directories with `result.json` are timestamped unless `--overwrite` is used.

The real workflow is:

```text
POSCAR/CIF/vasp + nep.txt/nep89.txt
-> ASE read
-> resolved settings table / JSON / YAML / run command
-> Calorine CPUNEP calculator
-> ASE atom + cell relaxation for bulk-oriented defaults
-> relaxed.vasp
-> initial/final spglib space-group diagnostics
-> Phonopy finite displacement
-> Calorine CPUNEP forces on displaced supercells
-> ASR / force-constant symmetrization when available
-> force_constants.hdf5
-> FORCE_CONSTANTS_2ND
-> phonopy.yaml
-> band.yaml
-> phonon_band.png
-> phonon_band.dat / phonon_band.csv / phonon_band_long.csv
-> phonon_band_segments.json / phonon_band_metadata.json
-> phonon_dos.dat / phonon_dos.png
-> phonon_group_velocity.csv / phonon_group_velocity.png
-> result.json
-> spacegroup_report.json / spacegroup_report.txt
-> summary.txt
```

Automatic choices are recorded in `resolved_settings.json`,
`resolved_settings.yaml`, and `run_command.txt`. `result.json` also stores
`settings_summary`, input/model SHA256 hashes, software versions, elapsed time,
and the resolved backend, supercell, mesh, primitive matrix, FC2 export, and
force-constant method fields. Auto supercell diagnostics include
`target_supercell_length`, `supercell_lengths_resolved`, `n_atoms_supercell`,
and warnings when the atom cap forces a smaller supercell. For structures with a
detected vacuum-like direction, the automatic supercell multiplier along that
direction is kept at 1.

Space-group diagnostics are written to `spacegroup_report.json` and
`spacegroup_report.txt`. The workflow compares the input structure and relaxed
structure using phonopy/spglib. `--phonopy-symprec` defaults to `1e-5`; `--angle-tolerance -1.0`
uses the spglib default. If symmetry changes after relaxation, inspect
`relax_cell`, `fmax`, `symprec`, input structure quality, and potential
compatibility.

Band plotting and export use the same explicit high-symmetry path. The path is
generated with `seekpath.get_explicit_k_path`; `explicit_kpoints_rel` are passed
to Phonopy, `explicit_kpoints_linearcoord` drive the x axis, and
`explicit_kpoints_labels` drive high-symmetry ticks. Each path segment is drawn
with independent SeekPath defaults `bandpath_symprec=1e-5` and
`bandpath_with_time_reversal=false`. These settings are separate from
`phonopy_symprec=1e-5`, which is reserved for phonopy harmonic symmetry and space-group diagnostics.
separately, and discontinuous boundaries are labeled with compound ticks such as
`U|K` rather than being connected by an artificial line.

The workflow records numbered progress steps in `run.log` and relaxation
details in `relax.log`. If relaxation does not reach `fmax`, the phonon
calculation continues and the result records a warning.

The default relaxation mode is intended for bulk crystals. For 2D materials,
slabs, surfaces, interfaces, or structures with vacuum, use `--no-relax-cell`
to keep the cell fixed and relax atomic positions only. PhonoFlow records a
heuristic `structure_type`, `vacuum_like_directions`, and relaxation warning
when it detects likely vacuum directions while `relax_cell=True`.

v0.2-p9 uses Calorine for force evaluation and Phonopy finite displacement for
second-order harmonic force constants. `FORCE_CONSTANTS_2ND` is Phonopy text
FC2 with the common ShengBTE FC2 filename. ShengBTE thermal conductivity still
requires third-order force constants, so this export alone is not a complete
ShengBTE calculation.

Thermal conductivity is optional and disabled by default. When
`--compute-kappa` is enabled, PhonoFlow uses the relaxed structure and
Calorine forces to drive a phono3py-based FC3 workflow. The finite-displacement
route can produce `fc3.hdf5`, `kappa-m*.hdf5`,
`thermal_conductivity.csv`, and `thermal_conductivity.png`. A single
temperature is shown as a bar chart over `kxx`, `kyy`, `kzz`, and `trace/3`;
multiple temperatures use grouped component bars and a highlighted `trace/3`
line. Lifetime files are written when phono3py provides either a direct
`lifetime` dataset or a `gamma` dataset that can be converted with phono3py's
documented `tau = 1 / (4*pi*gamma)` relation; gamma is in THz, so lifetime is
reported in ps. The experimental `--fc3-method hiphive` entry point reports a
clear unavailable reason if the fit cannot be completed and never fabricates
FC3, kappa, or lifetime data.

The thermal options are intentionally explicit:

```bash
python -m phonoflow run \
  --input-path examples/Si.vasp \
  --model-path nep89_potential/nep89_20250409.txt \
  --compute-kappa \
  --fc3-method finite-displacement \
  --method rta \
  --temperatures 300 \
  --kappa-mesh 5 5 5 \
  --fc3-supercell-dim 2 2 2
```

## DPA / DeepMD Workflow

PhonoFlow supports `--backend dpa31`, `dpa32`, `dpa33`, and `dpa4neo` for the
four current bundled DeepMD/DPA models. They resolve internally to DeepMD while
preserving `backend_requested`, `backend_resolved`, `backend_alias`,
`dpa_model_name`, `model_path`, and `model_file_hash` in `result.json`.

```bash
python -m phonoflow single --input-path examples/Si.vasp --backend dpa31 --outdir results/Si_dpa31 --overwrite
python -m phonoflow single --input-path examples/Si.vasp --backend dpa32 --outdir results/Si_dpa32 --overwrite
python -m phonoflow single --input-path examples/Si.vasp --backend dpa33 --outdir results/Si_dpa33 --overwrite
python -m phonoflow single --input-path examples/Si.vasp --backend dpa4neo --outdir results/Si_dpa4neo --overwrite
```

Compatibility aliases `dpa3` and `dpa4` resolve to `dpa32` and `dpa4neo`.
The generic `--backend dpa` requires an explicit `--model-path`. The old
DPA4-Pro-MPtrj model is not a default or silent fallback.

NEP89 and DPA share the formal geometry defaults: automatic FC2 inference with
`target_supercell_length=15`, automatic FC3 inference with
`fc3_target_supercell_length=10`, automatic kappa mesh, default displacement
amplitudes, explicit phono3py symprec, and no default cutoff-frequency
override. Explicit small settings are smoke parameters, not formal defaults.

The intentional DPA differences are `deepmd_device=cpu`, deterministic DeepMD
settings, calculator reuse, force audit outputs, and phono3py FC2 force-constant symmetrization enabled.
NEP89/Calorine keeps phono3py FC2 force-constant symmetrization disabled by default.

DPA single runs skip relaxation by default. If `--relax` is explicitly enabled,
PhonoFlow uses NEP89/Calorine relaxation unless `--relax-backend dpa` or
`--allow-dpa-relax` is specified.

Use repeatable `--model` options for any one to three independent workflows:

```bash
python -m phonoflow compare-models \
  --input-path examples/Si.vasp \
  --outdir results/model_compare \
  --model nep89 \
  --model DPA-3.1-3M.pt \
  --model DPA4-Neo-OMat24-v20260528_rc.pt \
  --compute-kappa \
  --overwrite
```

`--method lbte` and `--wigner true` are passed to phono3py when requested and
may depend on the installed phono3py version and problem size. Smoke-test
settings must not be treated as converged literature values; FC3 supercell,
kappa mesh, cutoff, displacement, temperature, and HiPhive fit settings require
their own convergence checks. Gruneisen parameters are not implemented.

DPA/DeepMD validation and audit outputs should live under one timestamped run
folder, for example `results/dpa_deepmd_audit_YYYYMMDD_HHMMSS/`. Each run
folder contains `report_en.md`, `report_zh.md`, `summary.json`,
`commands.log`, `validation.log`, `environment.json`, and artifact indexes.
Compare-models additionally writes `comparison_result.json`,
`comparison_summary.csv`, `comparison_summary.md`,
`comparison_kappa_bar.png`, `comparison_thermal_conductivity.png`,
`comparison_phonon_band.png`, and `comparison_dos.png`. The primary kappa bar
chart has four 300 K bars per successful model (`kxx`, `kyy`, `kzz`, and
`kavg`), giving 12 bars for any complete three-model run.
Web jobs keep their existing per-job `webapp/runs/<job_id>/` layout.

## DeepMD/DPA4 Workflow

DeepMD/DPA models use the same workflow spine as Calorine:

```text
input structure
-> deepmd.calculator.DP ASE calculator
-> phonopy FC2 / band / DOS / group velocity
-> phono3py FC3 / RTA kappa
-> result.json / summary.txt / audit files
```

Accepted backend aliases are `deepmd`, `dpa`, `dpa31`, `dpa32`, `dpa33`,
`dpa4neo`, plus compatibility aliases `dpa3` and `dpa4`. They resolve
internally to `deepmd`, while `result.json` keeps the requested/canonical alias
and exact model filename.

DPA4 reproducibility differs from NEP89. The SeZM force kernel can show
run-to-run force hash drift around floating-point noise levels. Therefore DPA4
validation should enable:

```bash
--deepmd-deterministic --deepmd-reuse-calculator --save-force-audit --phono3py-symmetrize-fc2
```

`--deepmd-deterministic` is best-effort thread pinning. It does not promise
bit-identical forces. The kappa-stabilizing option is `--phono3py-symmetrize-fc2`,
which applies ASR to phono3py's FC2 before RTA. It is enabled by default for
DPA aliases and remains off by default for NEP89, so legacy NEP89 behavior is
unchanged.

kALDo is not part of the main DPA4 workflow. It was evaluated in the reference
project and is recorded as a possible future cross-check, not a default
dependency or thermal engine.

## Web Studio Previews

The Web interface can inspect a structure before the calculation starts. The
preview API reads uploaded POSCAR/CONTCAR, VASP, CIF, or XYZ files, or pasted
POSCAR text, with ASE. For POSCAR text, it first sanitizes the declared
coordinate block so VASPKIT-style velocity blocks are not misread as atoms. It
returns formula, reduced formula, elements, atom count, cell lengths, cell
angles, Cartesian positions, scaled positions, and a covalent-radius inferred
bond list for visualization. These preview calls are temporary UI helpers and
do not write into the formal `results/` output directories.

The k-path preview API uses SeekPath to derive primitive and reciprocal lattice
information, high-symmetry point coordinates, path segments, labels, and a
display string. The display string is built directly from SeekPath path
segments as `A — B | C — D`, not from point ordering. The browser renders a
reciprocal-basis/k-path 3D preview and the path text. If SeekPath cannot
classify the structure, the API returns `available=false` with a reason; this
does not block normal job submission.

The local Web assistant is a top-of-page rule-based intent parser rather than
an external LLM. It recognizes Chinese and English keywords for phonon
dispersion, DOS, group velocity, lifetime, thermal conductivity, NEP89, custom
NEP models, quick-preview/no-relax mode, and vacuum/interface mode. The parsed
plan is converted to a fixed allowlist of Web runner parameters and then
submitted through the same job creation and runner path as the standard form.

## Si Validation Example

Use the Linux filesystem working copy for real validation:

```bash
cd ~/PhonoFlow
python -m pip install -e ".[dev,calorine]"
python -m phonoflow doctor --verbose
python -m phonoflow run \
  --input-path examples/Si.vasp \
  --model-path nep89_potential/nep89_20250409.txt \
  --dry-run
python -m phonoflow single \
  --input-path examples/Si.vasp \
  --model-path nep89_potential/nep89_20250409.txt \
  --outdir results/Si_calorine_validation \
  --backend calorine \
  --relax \
  --target-supercell-length 20 \
  --phonopy-symprec 1e-5 \
  --displacement 0.01 \
  --fmax 1e-5 \
  --max-steps 2000 \
  --band auto \
  --export-fc2-text
python scripts/validate_output.py results/Si_calorine_validation
python -m phonoflow read-result results/Si_calorine_validation
```

Expected files are `resolved_config.yaml`, `resolved_settings.json`,
`resolved_settings.yaml`, `run_command.txt`, `input_structure.vasp`,
`relaxed.vasp`, `relax.log`, `force_constants.hdf5`, `FORCE_CONSTANTS_2ND`,
`phonopy.yaml`, `band.yaml`, `phonon_band.png`, `phonon_group_velocity.png`,
`result.json`, `stability_report.json`, `stability_report.txt`, and `run.log`.

The validation succeeds when all files are present and non-empty, `result.json`
has `success=true`, force constants can be opened by h5py, `band.yaml` can be
read as YAML, and `phonon_band.png` is non-empty.

Common failure modes:

- Calorine is not installed in the active Python environment.
- Calorine imports, but `CPUNEP` is unavailable in the installed version.
- The NEP/NEP89 potential path does not exist or is incompatible with CPUNEP.
- Phonopy API failure prevents `band.yaml` from being written.
- The plot is empty because no frequencies were produced.
- Small negative frequencies may be numerical noise under the configured
  imaginary-frequency threshold.
- Large imaginary frequencies suggest a real instability or an input/potential
  mismatch.
