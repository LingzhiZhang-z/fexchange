# 03-05-ION_ED

This file defines the optional full single-ion ED representation used by
`model.scheme = "ED"`.

## 1) Scope (MUST)
MUST:
- IONED is defined only for fixed f-electron sectors.
- IONED diagonalizes the full single-ion Hamiltonian
  `Hion = H_int + H_soc`.
- IONED is used only for adjacent intermediate sectors `f^(n-1)` and
  `f^(n+1)` in current runtime schemes.
- The main `f^n` low-energy subspace remains the LSJM SOC-lowest subspace.

Validation:
- IONED must not change `L0`.
- IONED must not replace Kramers/projector handling.

## 2) Hamiltonian and Basis (MUST)
MUST:
- The determinant basis is the canonical f-shell Fock basis from module
  `01-00`.
- `H_int` follows module `02-01`.
- `H_soc` follows module `02-02`.
- The numerical diagonalization is a Hermitian ED:

Code form:
```text
evals, evecs = eigh(H_int + H_soc)
V_fock_ed = evecs
energies = evals + offset
```

Validation:
- `Hion` must be Hermitian.
- `V_fock_ed` columns must be orthonormal.

## 3) Degenerate Subspace Canonicalization (MUST)
MUST:
- Energy-degenerate columns must be made deterministic.
- Within each degenerate energy cluster, diagonalize `J2`.
- Within each resulting `J2` cluster, diagonalize `Jz`.
- Fix vector phases using the project state-vector convention.

Code form:
```text
cluster by energy
  diagonalize J2 in cluster
  cluster by J2
    diagonalize Jz
    sort by M
```

Validation:
- Reported `J` must be compatible with `J(J+1)`.
- Reported `M` must be half-integer compatible and ordered within each
  multiplet.

## 4) Output (MUST)
MUST:
- Output columns are canonical-Fock-to-IONED transforms.
- Required arrays:
  - `V_fock_ed`
  - `energies`
- Optional diagnostic arrays:
  - `J`
  - `M`
  - `energy_group`
- Metadata must include labels, `basis_id`, `state_order_id`,
  `orbital_order_id`, `n_ele`, and the physics parameters.

Code form:
```text
IONED = {
  V_fock_ed(alpha_fock,state),
  energies(state),
  labels(state)
}
```
