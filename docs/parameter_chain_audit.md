# Parameter Chain Audit

This public audit summarizes the active CLI parameter chain without historical
private references.

## Active Symmetrization Fields

- `phono3py_symmetrize_fc2`: controls official Phono3py FC2 force-constant
  symmetrization in the finite-displacement thermal route.
- `phono3py_symmetrize_fc3`: controls official Phono3py FC3 force-constant
  symmetrization in the finite-displacement thermal route.
- HiPhive FC3 fitting records that Phono3py FC2/FC3 symmetrization was requested
  but does not call those Phono3py hooks in that route.

## Deprecated Compatibility

The old `phono3py_fc2_asr` input alias is deprecated. It is accepted only for
backward-compatible config loading and is mapped to `phono3py_symmetrize_fc2`.
New CLI commands and docs should use `phono3py_symmetrize_fc2`.
