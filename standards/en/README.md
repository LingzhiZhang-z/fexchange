# Scientific Computing Standards

## Authority
Code implementation MUST follow the English standards under `./standards/en/`.
Files under `./standards/zh/` are translations only.
If EN and ZH conflict, EN is authoritative.

## Directory Structure

### 00-meta/ — Spec writing conventions, software engineering, conflict reporting
- `./standards/en/00-meta/00-00-SPEC_WRITING_CONVENTION.md`
- `./standards/en/00-meta/00-01-SOFTWARE_ENGINEERING.md`
- `./standards/en/00-meta/00-07-CONFLICT_REPORT_TEMPLATE.md`

### 01-core/ — Fock space, state vectors, fermion operators, Stevens and tensor operators, angular momentum
- `./standards/en/01-core/01-00-FOCK_SLATER.md`
- `./standards/en/01-core/01-01-STATE_VECTOR.md`
- `./standards/en/01-core/01-02-OPERATOR_IMPLEMENTATION.md`
- `./standards/en/01-core/01-03-STEVENS_OPERATORS.md`
- `./standards/en/01-core/01-04-SPHERICAL_TENSOR_OPERATORS.md`
- `./standards/en/01-core/01-05-STEVENS_TENSOR_CONVERSION.md`
- `./standards/en/01-core/01-06-ANGULAR_MOMENTUM.md`

### 02-hamiltonian/ — Local Hamiltonians
- `./standards/en/02-hamiltonian/02-00-LOCAL_HAMILTONIAN.md`
- `./standards/en/02-hamiltonian/02-01-HINT.md`
- `./standards/en/02-hamiltonian/02-02-HSOC.md`
- `./standards/en/02-hamiltonian/02-03-HCEF.md`

### 03-spectrum/ — LSMS/LSJM representations, doublets, irrep classification
- `./standards/en/03-spectrum/03-00-LSMS.md`
- `./standards/en/03-spectrum/03-01-LSJM.md`
- `./standards/en/03-spectrum/03-02-KRAMERS_DOUBLET.md`
- `./standards/en/03-spectrum/03-03-NON_KRAMERS_DOUBLET.md`
- `./standards/en/03-spectrum/03-04-IRREP_CLASSIFICATION.md`

### 04-sopt/ — Second-order perturbation theory
- `./standards/en/04-sopt/04-00-SOPT_FORMALISM.md`
- `./standards/en/04-sopt/04-01-PRECOMPUTE.md`
- `./standards/en/04-sopt/04-02-RUNTIME_CONTRACTION.md`
- `./standards/en/04-sopt/04-03-SPIN12_MAPPING.md`
- `./standards/en/04-sopt/04-REF-DERIVATION.md`

### 05-io/ — Disk I/O, layout, Wannier90, run-input contract
- `./standards/en/05-io/05-00-IO.md`
- `./standards/en/05-io/05-01-IO_LAYOUT.md`
- `./standards/en/05-io/05-02-WANNIER90_CONTRACT.md`
- `./standards/en/05-io/05-03-WANNIER90_PARSING.md`
- `./standards/en/05-io/05-04-RUN_INPUT.md`

### 06-utils/ — Numerics tolerances and error contracts
- `./standards/en/06-utils/06-00-RUNTIME_NUMERICS.md`
- `./standards/en/06-utils/06-01-ERROR_CODES.md`

## Mapping Note
The standards directories cover domain and cross-cutting contracts.
`fexchange/pipeline/` is still governed by these standards, but through the
module maps in `AGENTS.md` and `CLAUDE.md` rather than a dedicated `07-pipeline/`
directory.

## Required Reading Order
1. `./standards/en/00-meta/00-00-SPEC_WRITING_CONVENTION.md`
2. `./standards/en/00-meta/00-01-SOFTWARE_ENGINEERING.md`
3. `./standards/en/06-utils/06-00-RUNTIME_NUMERICS.md`
4. `./standards/en/06-utils/06-01-ERROR_CODES.md`
5. `./standards/en/01-core/01-00-FOCK_SLATER.md`
6. `./standards/en/01-core/01-01-STATE_VECTOR.md`
7. `./standards/en/01-core/01-02-OPERATOR_IMPLEMENTATION.md`
8. `./standards/en/01-core/01-03-STEVENS_OPERATORS.md`
9. `./standards/en/01-core/01-04-SPHERICAL_TENSOR_OPERATORS.md`
10. `./standards/en/01-core/01-05-STEVENS_TENSOR_CONVERSION.md`
11. `./standards/en/01-core/01-06-ANGULAR_MOMENTUM.md`
12. `./standards/en/02-hamiltonian/02-00-LOCAL_HAMILTONIAN.md`
13. `./standards/en/02-hamiltonian/02-01-HINT.md`
14. `./standards/en/02-hamiltonian/02-02-HSOC.md`
15. `./standards/en/02-hamiltonian/02-03-HCEF.md`
16. `./standards/en/03-spectrum/03-00-LSMS.md`
17. `./standards/en/03-spectrum/03-01-LSJM.md`
18. `./standards/en/03-spectrum/03-02-KRAMERS_DOUBLET.md`
19. `./standards/en/03-spectrum/03-03-NON_KRAMERS_DOUBLET.md`
20. `./standards/en/03-spectrum/03-04-IRREP_CLASSIFICATION.md`
21. `./standards/en/04-sopt/04-00-SOPT_FORMALISM.md`
22. `./standards/en/04-sopt/04-01-PRECOMPUTE.md`
23. `./standards/en/04-sopt/04-02-RUNTIME_CONTRACTION.md`
24. `./standards/en/04-sopt/04-03-SPIN12_MAPPING.md`
25. `./standards/en/04-sopt/04-REF-DERIVATION.md`
26. `./standards/en/05-io/05-00-IO.md`
27. `./standards/en/05-io/05-01-IO_LAYOUT.md`
28. `./standards/en/05-io/05-02-WANNIER90_CONTRACT.md`
29. `./standards/en/05-io/05-03-WANNIER90_PARSING.md`
30. `./standards/en/05-io/05-04-RUN_INPUT.md`

## Conflict Rule
If standards conflict, stop immediately and report using:
- `./standards/en/00-meta/00-07-CONFLICT_REPORT_TEMPLATE.md`
