# Scientific Computing Standards

## Authority
Code implementation MUST follow the English standards under `./standards/en/`.
Files under `./standards/zh/` are translations only.
If EN and ZH conflict, EN is authoritative.
`*_LEGACY` files are non-normative by default.

## Directory Structure

### 00-conventions/ — Project conventions, software engineering, runtime infrastructure
- `./standards/en/00-conventions/00-00-SPEC_WRITING_CONVENTION.md`
- `./standards/en/00-conventions/00-01-SOFTWARE_ENGINEERING.md`
- `./standards/en/00-conventions/00-02-RUNTIME_NUMERICS_AND_INPUT_GATES.md`
- `./standards/en/00-conventions/00-03-ERROR_CODES_AND_FAILURE_PAYLOAD.md`
- `./standards/en/00-conventions/00-04-LEGACY_ISOLATION.md`
- `./standards/en/00-conventions/00-05-RUN_INPUT_SINGLE_FILE.md`
- `./standards/en/00-conventions/00-06-MPI_PARALLEL_RUNTIME.md`
- `./standards/en/00-conventions/00-07-CONFLICT_REPORT_TEMPLATE.md`

### 01-physics/ — Core physical foundations
- `./standards/en/01-physics/01-00-FOUNDATIONS_FOCK_SLATER.md`
- `./standards/en/01-physics/01-01-STATE_VECTOR_CONVENTION.md`
- `./standards/en/01-physics/01-02-OPERATOR_IMPLEMENTATION.md`

### 02-models/ — Local Hamiltonian and model definitions
- `./standards/en/02-models/02-00-MODEL_LOCAL_HAMILTONIAN.md`
- `./standards/en/02-models/02-01-HINT_FORM.md`
- `./standards/en/02-models/02-02-HSOC_FORM.md`
- `./standards/en/02-models/02-03-HCEF_FORM.md`
- `./standards/en/02-models/02-04-ANGULAR_MOMENTUM_OPERATORS.md`
- `./standards/en/02-models/02-05-KRAMERS_DOUBLET_G_TENSOR.md`
- `./standards/en/02-models/02-06-NON_KRAMERS_DOUBLET.md`

### 03-representations/ — Basis representations (LSMS, LSJM)
- `./standards/en/03-representations/03-00-REPRESENTATION_LSMS.md`
- `./standards/en/03-representations/03-01-REPRESENTATION_LSJM.md`

### 04-sopt/ — Second-order perturbation theory
- `./standards/en/04-sopt/04-00-SOPT_FORMALISM.md`
- `./standards/en/04-sopt/04-01-PRECOMPUTE_PIPELINE.md`
- `./standards/en/04-sopt/04-02-RUNTIME_CONTRACTION.md`
- `./standards/en/04-sopt/04-03-SPIN12_MAPPING.md`
- `./standards/en/04-sopt/04-02-RUNTIME_CONTRACTION_LEGACY.md` (legacy, non-normative)

### 05-io/ — Input/output, file formats, Wannier90 integration
- `./standards/en/05-io/05-00-IO.md`
- `./standards/en/05-io/05-01-IO_LAYOUT_GUIDE.md` (non-normative)
- `./standards/en/05-io/05-02-WANNIER90_INPUT_CONTRACT.md`
- `./standards/en/05-io/05-03-WANNIER90_PARSING_RULES.md`

## Required Reading Order
1. `./standards/en/00-conventions/00-00-SPEC_WRITING_CONVENTION.md`
2. `./standards/en/00-conventions/00-01-SOFTWARE_ENGINEERING.md`
3. `./standards/en/01-physics/01-00-FOUNDATIONS_FOCK_SLATER.md`
4. `./standards/en/01-physics/01-01-STATE_VECTOR_CONVENTION.md`
5. `./standards/en/01-physics/01-02-OPERATOR_IMPLEMENTATION.md`
6. `./standards/en/00-conventions/00-02-RUNTIME_NUMERICS_AND_INPUT_GATES.md`
7. `./standards/en/00-conventions/00-03-ERROR_CODES_AND_FAILURE_PAYLOAD.md`
8. `./standards/en/00-conventions/00-04-LEGACY_ISOLATION.md`
9. `./standards/en/00-conventions/00-05-RUN_INPUT_SINGLE_FILE.md`
10. `./standards/en/00-conventions/00-06-MPI_PARALLEL_RUNTIME.md`
11. `./standards/en/05-io/05-00-IO.md`
12. `./standards/en/05-io/05-01-IO_LAYOUT_GUIDE.md` (optional/non-normative)
13. `./standards/en/05-io/05-02-WANNIER90_INPUT_CONTRACT.md`
14. `./standards/en/05-io/05-03-WANNIER90_PARSING_RULES.md`
15. `./standards/en/02-models/02-00-MODEL_LOCAL_HAMILTONIAN.md` + `02-01` through `02-06`
16. `./standards/en/03-representations/03-00-REPRESENTATION_LSMS.md` then `./standards/en/03-representations/03-01-REPRESENTATION_LSJM.md`
17. `./standards/en/04-sopt/04-00-SOPT_FORMALISM.md`
18. `./standards/en/04-sopt/04-01-PRECOMPUTE_PIPELINE.md`
19. `./standards/en/04-sopt/04-02-RUNTIME_CONTRACTION.md`
20. `./standards/en/04-sopt/04-03-SPIN12_MAPPING.md`

## Conflict Rule
If standards conflict, stop immediately and report using:
- `./standards/en/00-conventions/00-07-CONFLICT_REPORT_TEMPLATE.md`
