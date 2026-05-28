# 00-meta/00-01-SOFTWARE_ENGINEERING

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
- Domain subpackage layout should mirror the standards hierarchy where a
  dedicated standards directory exists.
- `pipeline/` is a documented exception: it is governed by cross-cutting
  standards plus the module maps in `AGENTS.md` and `CLAUDE.md`, not by its own
  standards directory.
- The package layout is:

```text
fexchange/
  __init__.py
  core/
    __init__.py
    fock.py                 # 01-00: Fock/Slater foundations
    fermion.py              # 01-02: operator implementation
    states.py               # 01-01: state-vector conventions
    stevens.py              # 01-03/01-04/01-05: Stevens and tensor operators
    space_j.py              # 01-06: J-space angular momentum operators
    space_ls.py             # 01-06: LS-space angular momentum operators
  hamiltonian/
    __init__.py
    hint.py                 # 02-01: H_int construction
    hsoc.py                 # 02-02: H_soc construction
    hcef.py                 # 02-03: H_cef assembly
  spectrum/
    __init__.py
    lsms.py                 # 03-00: LSMS representation
    lsjm.py                 # 03-01: LSJM representation
    energy.py               # 02-00: intermediate-state energy reconstruction
    doublet.py              # 03-02/03-03: Kramers and non-Kramers doublet basis
    ground.py               # 03-02/03-03: ground-state doublet selection
    classify.py             # 03-04: irrep classification
    tables.py               # 03-04: symmetry character tables
    multipole.py            # 01-03/01-04: multipole operator display
  sopt/
    __init__.py
    precompute.py           # 04-01: L0 + L1
    contraction.py          # 04-02: L2 + L3
    spin12.py               # 04-03: spin-1/2 mapping
  fopt/
    __init__.py
    preprocessing.py        # 04-fopt: L0 + L1 + L2 active-pair blocks
  io/
    __init__.py
    disk.py                 # 05-00: disk I/O, path tokens, cache
    matrix.py               # 05-00: matrix serialization
    wannier90.py            # 05-02+05-03: Wannier90 parsing and input contract
    run_input.py            # 05-04: TOML run-input loader
  pipeline/
    __init__.py
    artifacts.py            # artifact persistence and metadata
    keys.py                 # pipeline key generation
    resolve.py              # dependency resolution
    stages.py               # stage execution orchestration
    validation.py           # pipeline-level validation
  utils/
    __init__.py
    numerics.py             # 06-00: tolerance table, dtype policy
    errors.py               # 06-01: error codes, coded exceptions
    checks.py               # runtime checks (Hermiticity, orthonormality)
    constants.py            # physical constants (ELL, N_ORB)
  cli.py                    # entry point
```

Validation:
- All public module names must be importable as `from fexchange.<sub> import <mod>`.

## 3) Dependencies (MUST)
MUST:
- Core numerical: `numpy >= 1.24`, `scipy >= 1.10`.
- File I/O: `tomli` (or stdlib `tomllib` on Python >= 3.11).
- 3j/CG symbols: `sympy >= 1.12` (for `sympy.physics.wigner`; sole 3j/CG implementation).
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
dev    = ["pytest>=7.0"]
```

Validation:
- `pip install .` must succeed with core dependencies only.
- No parallel runtime dependency is assumed by the current reference implementation.

## 4) Entry Point and CLI (MUST)
MUST:
- Provide one CLI command: `fexchange run <run_input.toml>`.
- This command reads the TOML file per `./standards/en/05-io/05-04-RUN_INPUT.md`,
  executes the specified level window, and writes outputs per `./standards/en/05-io/05-00-IO.md`.

Code form:
```text
fexchange run ./run_input.toml
```

Validation:
- Exit code `0` on success, nonzero on failure.
- On failure, emit JSON error payload to stderr per `./standards/en/06-utils/06-01-ERROR_CODES.md`.

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
  test_energy.py           # 02-00: energy reconstruction
  test_ground_doublets.py  # 03-02/03-03: Kramers/non-Kramers doublet
  test_sopt_l0.py          # 04-01: X/Y sign consistency
  test_sopt_l1.py          # 04-01: A/B vertex dimensions
  test_sopt_l2_l4.py       # 04-02: zero-hop check, Hermiticity of Heff
  test_run_input.py        # 05-04: TOML input schema validation
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
- Each computational module may expose one or more public functions/classes.
- Every public entry must use explicit typed inputs/outputs (numpy arrays plus
  metadata dicts where needed).
- Every public entry must perform runtime checks at its own boundary.
- Internal helpers must be prefixed with `_` (private by convention).

Code form:
```text
def build_lsms(...):
    _validate_inputs(...)
    ...
    _validate_outputs(...)

def classify_irreps(...):
    _validate_inputs(...)
    ...
```

Validation:
- Public functions without boundary validation are contract violations.

## 7) Logging and Reproducibility (MUST)
MUST:
- Use Python `logging` module for runtime messages.
- Every stage must emit a structured summary including:
  `level`, `key`, `elapsed_s`, and any numerical/runtime metadata actually in use.
  `numerics_meta` is one allowed container, not a mandatory field name.
- Random seeds (if any) must be logged; prefer deterministic algorithms with no randomness.

Code form:
```text
import logging
logger = logging.getLogger("fexchange")
```
