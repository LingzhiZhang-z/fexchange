# 00-conventions/00-01-SOFTWARE_ENGINEERING

This file defines the software engineering conventions for implementing the
standards in `./standards/en/`.
It is normative for code structure, dependencies, and testing.

## 1) Language and Runtime (MUST)
MUST:
- Implementation language: Python >= 3.10.
- No Cython or C extensions are required for the reference implementation.
- Type hints are recommended but not mandatory.

Code form:
```text
python_version >= "3.10"
```

## 2) Package Structure (MUST)
MUST:
- Use a single top-level package: `fexchange`.
- Subpackage layout must mirror the standard hierarchy:

```text
fexchange/
  __init__.py
  core/
    __init__.py
    fock_basis.py          # 01-00: Fock/Slater foundations
    fermion_ops.py         # 01-02: operator implementation
    orbitals.py            # 01-00/02-04: orbital mapping, angular momentum
    states.py              # 01-01: state-vector objects (BasisDet, StateVec, StateSet)
  models/
    __init__.py
    hint.py                # 02-01: H_int construction
    hsoc.py                # 02-02: H_soc construction
    hcef.py                # 02-03: H_cef + Stevens operators
    angular.py             # 02-04: L/S/J operator builders
    kramers.py             # 02-05: Kramers doublet + g-tensor
    non_kramers.py         # 02-06: non-Kramers doublet
  representations/
    __init__.py
    lsms.py                # 03-00: LSMS representation
    lsjm.py                # 03-01: LSJM representation
  sopt/
    __init__.py
    precompute.py           # 04-01: L0 + L1
    contraction.py          # 04-02: L2 + L3 + L4
    spin12.py               # 04-03: spin-1/2 mapping
  io/
    __init__.py
    disk.py                 # 05-00: disk I/O, path tokens, cache
    wannier90.py            # 05-02+05-03: Wannier90 parsing and input contract
    run_input.py            # 00-05: TOML run-input loader
  utils/
    __init__.py
    numerics.py             # 00-02: tolerance table, dtype policy
    errors.py               # 00-03: error codes, coded exceptions
    checks.py               # runtime checks (Hermiticity, orthonormality)
    parallel.py             # 00-06: MPI wrapper / serial fallback
  cli.py                    # entry point
```

Validation:
- All public module names must be importable as `from fexchange.<sub> import <mod>`.

## 3) Dependencies (MUST)
MUST:
- Core numerical: `numpy >= 1.24`, `scipy >= 1.10`.
- File I/O: `tomli` (or stdlib `tomllib` on Python >= 3.11).
- 3j/CG symbols: `sympy >= 1.12` (for `sympy.physics.wigner`; sole 3j/CG implementation).
- Parallel: `mpi4py >= 3.1` (optional; serial fallback is mandatory).
- Testing: `pytest >= 7.0`.
- No other hard runtime dependencies.

Code form:
```toml
[project]
name = "fexchange"
requires-python = ">=3.10"
dependencies = [
  "numpy>=1.24",
  "scipy>=1.10",
  "sympy>=1.12",
]

[project.optional-dependencies]
mpi    = ["mpi4py>=3.1"]
dev    = ["pytest>=7.0"]
```

Validation:
- `pip install .` must succeed with core dependencies only.
- MPI is runtime-optional: import failures must fall back to serial mode.

## 4) Entry Point and CLI (MUST)
MUST:
- Provide one CLI command: `fexchange run <run_input.toml>`.
- This command reads the TOML file per `./standards/en/00-conventions/00-05-RUN_INPUT_SINGLE_FILE.md`,
  executes the specified level window, and writes outputs per `./standards/en/05-io/05-00-IO.md`.

Code form:
```text
fexchange run ./run_input.toml
```

Validation:
- Exit code `0` on success, nonzero on failure.
- On failure, emit JSON error payload to stderr per `./standards/en/00-conventions/00-03-ERROR_CODES_AND_FAILURE_PAYLOAD.md`.

## 5) Testing Framework (MUST)
MUST:
- Use `pytest` as the test runner.
- Test directory layout:

```text
tests/
  conftest.py              # shared fixtures (small Fock bases, known LS terms)
  test_fock_basis.py       # 01-00: bitstring, sign, dimension
  test_fermion_ops.py      # 01-02: anti-commutation, Hermitian consistency
  test_hint.py             # 02-01: Coulomb operator, known term energies
  test_hsoc.py             # 02-02: SOC Hermiticity, Landé cross-check
  test_hcef.py             # 02-03: Stevens operator Hermiticity, Oh/C3v checks
  test_lsms.py             # 03-00: orthonormality, term count, H_int diagonal
  test_lsjm.py             # 03-01: CG cross-check, SOC diagonal
  test_sopt_l0.py          # 04-01: X/Y sign consistency
  test_sopt_l1.py          # 04-01: A/B vertex dimensions
  test_sopt_l2_l4.py       # 04-02: zero-hop check, Hermiticity of Heff
  test_spin12.py           # 04-03: reconstruction residual
  test_io.py               # 05-00: path token generation, cache round-trip
  test_wannier90.py        # 05-03: parsing smoke test
```

- Minimum required test categories:
  1. Anti-commutation identity checks on random determinants.
  2. Hermitian consistency (`H == H.conj().T`) for all constructed operators.
  3. Orthonormality of LSMS/LSJM state sets.
  4. Zero-hopping sanity: `t=0 => Heff=0`.
  5. Reconstruction residual for spin-1/2 mapping.

Code form:
```text
pytest tests/ -v --tb=short
```

## 6) Module Interface Pattern (MUST)
MUST:
- Each computational module must expose a single public function or class with:
  - explicit typed inputs (numpy arrays + metadata dicts),
  - explicit typed outputs (numpy arrays + metadata dicts),
  - runtime checks at entry and exit.
- Internal helpers must be prefixed with `_` (private by convention).

Code form:
```text
def build_lsms(n_ele, F2, F4, F6, basis_id, ...) -> LsmsResult:
    _validate_inputs(...)
    ...  # computation
    _validate_outputs(result)
    return result
```

Validation:
- Public functions without input validation are contract violations.

## 7) Logging and Reproducibility (MUST)
MUST:
- Use Python `logging` module for runtime messages.
- Every stage must emit a structured summary including:
  `level`, `key`, `elapsed_s`, `numerics_meta`, `parallel_meta`.
- Random seeds (if any) must be logged; prefer deterministic algorithms with no randomness.

Code form:
```text
import logging
logger = logging.getLogger("fexchange")
```
