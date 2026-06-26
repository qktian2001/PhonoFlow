# Testing

PhonoFlow 1.0 includes tests for CLI parsing, config validation, backend
selection, output policy, band-path behavior, force-constant handling, plotting,
reporting, thermal options, and public smoke workflows.

## Commands

Compile all Python files:

```bash
python -m compileall src tests scripts
```

Run the suite:

```bash
PYTHONPATH=src python -m pytest tests -q
```

Run one test file:

```bash
PYTHONPATH=src python -m pytest tests/test_config.py -q
```

## Public Fixtures

The public repository keeps only one small structure fixture:

- `examples/Si.vasp`

Private model weights are not stored in the repository. Public tests use dummy
or placeholder model paths where possible.

## Optional Backends

Some behavior depends on optional packages such as Calorine, Phono3py, HiPhive,
or DeepMD-kit. Tests that require unavailable optional runtimes may skip. Run
`phonoflow doctor --verbose` to inspect the local environment before debugging
backend-specific failures.

## Expected Warning

SeekPath can emit `EdgeCaseWarning` for structures close to a symmetry boundary.
That warning is a numerical classification warning from SeekPath, not a test
failure.
