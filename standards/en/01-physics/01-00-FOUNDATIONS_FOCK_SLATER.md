# 01-00-FOUNDATIONS_FOCK_SLATER

This file defines mandatory low-level Fock/Slater conventions shared by all modules.

## 1) Fock Space (MUST)
MUST:
- Single-ion f shell uses `n_orb = 14` spin-orbitals.
- Electron-number sectors are `n = 0..14`.

Math:
$$
\dim(n)=\binom{14}{n}.
$$

Code form:
```text
n_orb = 14
dim_n = comb(14, n)
```

Validation:
- All sector objects must satisfy `0 <= n <= 14`.

## 2) Orbital Index Mapping (MUST)
MUST:
- Orbital index is `p = 0..13`.
- Mapping order is fixed: for `m=-3..3`, spin order is `(-1/2, +1/2)`.

Code form:
```text
p -> (m, sigma)
m in [-3,-2,-1,0,1,2,3]
sigma order = [-1/2, +1/2] for each m
```

Validation:
- The same mapping must be used by states, operators, and hopping tensors.

## 3) Single-Electron Basis (MUST)
MUST:
- Physical basis state is $\lvert l=3,m,\sigma\rangle$.
- External interfaces are traceable to this default complex spherical-harmonic basis.

Math:
$$
\lvert p\rangle \equiv \lvert l=3,m(p),\sigma(p)\rangle.
$$

Code form:
```text
basis_one_electron = |l=3,m,sigma>
p is serialization label only
```

Validation:
- If internal basis differs, explicit transform metadata is required.

## 4) Matrix/Vector Convention (MUST)
MUST:
- Matrix elements use bra-ket convention.
- State vectors are columns.

Math:
$$
A_{ij}=\langle i\lvert A\rvert j\rangle,
\qquad
\lvert\psi_{out}\rangle = A\lvert\psi_{in}\rangle,
\qquad
\langle\psi\vert\phi\rangle=\psi^\dagger\phi.
$$

Code form:
```text
A[i,j] = <i|A|j>
psi_out = A @ psi_in
inner = psi.conj().T @ phi
```

Validation:
- Hermitian conjugate is `A_dag = (A.conj()).T`.

## 5) Bitstring Encoding (MUST)
MUST:
- Determinant is a non-negative integer `det`.
- Bit `p` corresponds to orbital `p` from Section 2.

Math:
$$
n_p(det)=((det \gg p)\ \&\ 1)\in\{0,1\},
\qquad
\sum_{p=0}^{13} n_p(det)=n.
$$

Code form:
```text
occ_p = (det >> p) & 1
n_ele = popcount(det)
det_set   = det | (1 << p)
det_clear = det & ~(1 << p)
```

Validation:
- Creation on occupied orbital and annihilation on empty orbital return zero.

## 6) Slater-Basis Identity (MUST)
MUST:
- In sector `n`, determinants are sorted by ascending integer (`lex_v1`).
- `basis_id` format is fixed.

Code form:
```text
basis_id = f"fock{n_orb}_n{n_ele}_{det_order}_v{major}"
example: fock14_n5_lex_v1
```

Validation:
- Cross-file reads must fail on `basis_id` mismatch.
- `basis_id` must not encode hopping/Kramers/truncation choices.

## 7) Fermionic Sign Convention (MUST)
MUST:
- Use parity below index for both creation and annihilation.

Math:
$$
c_p^\dagger\lvert det\rangle=
\begin{cases}
0,& n_p=1\\
(-1)^{N_{<p}(det)}\lvert det\cup\{p\}\rangle,& n_p=0
\end{cases}
$$

$$
c_p\lvert det\rangle=
\begin{cases}
(-1)^{N_{<p}(det)}\lvert det\setminus\{p\}\rangle,& n_p=1\\
0,& n_p=0
\end{cases}
$$

Code form:
```text
phase = (-1) ** occupied_count_below_p(det)
```

Validation:
- Adjacent-sector annihilation matrices are Hermitian transpose of creation matrices.

## 8) State/Energy Array Contract (MUST)
MUST:
- State matrices use column convention.
- Energies align one-to-one with column index.

Math:
$$
V_{\mathrm{fock}}\in\mathbb C^{d_{\mathrm{fock}}\times n_{\mathrm{states}}},
\qquad
V_{\mathrm{fock}}^\dagger V_{\mathrm{fock}}=I
\ \text{(for orthonormal sets)}.
$$

Code form:
```text
V_fock.shape = (dim_fock, n_states)
energy[a] <-> column a
```

Validation:
- Column count must match labels and energies.

## 9) Units and Traceability (MUST)
MUST:
- Public outputs must record `basis_id` and orbital-order metadata when needed.
- Energy unit must be explicit in metadata.

Code form:
```text
meta = {basis_id, orbital_order_id?, unit, ...}
```

Validation:
- Any implicit-unit output is invalid.
