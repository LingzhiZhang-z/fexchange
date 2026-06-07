# 03-05-ION_ED

本文件定义 `model.scheme = "ED"` 使用的可选 full single-ion ED 表示。

## 1) 适用范围（MUST）
MUST:
- IONED 只定义在固定 f-electron sectors 上。
- IONED 对 full single-ion Hamiltonian 做对角化：
  `Hion = H_int + H_soc`；当 runtime 提供 `inputs.hcef_file` 时，可包含
  optional one-body `H_cef`。
- 当前 runtime schemes 中，IONED 只用于相邻 intermediate sectors
  `f^(n-1)` 和 `f^(n+1)`。
- 主 `f^n` low-energy subspace 仍是 LSJM SOC-lowest subspace。
  当 `runtime.kramer_source = "manual"` 启用 manual Kramers input 时，
  它由 L1 binding 处理，不是 IONED main-sector replacement。

Validation:
- IONED 不得改变 `L0`。
- IONED 不得替代 Kramers/projector handling。

## 2) Hamiltonian 和 Basis（MUST）
MUST:
- Determinant basis 是模块 `01-00` 中的 canonical f-shell Fock basis。
- `H_int` 遵循模块 `02-01`。
- `H_soc` 遵循模块 `02-02`。
- Optional one-body `H_cef` 遵循模块 `02-03`，并提升到固定 Fock sector：
  `H_cef = sum_{p,q} h_cef[p,q] c_p^dag c_q`。
- 数值对角化是 Hermitian ED：

Code form:
```text
evals, evecs = eigh(H_int + H_soc + optional(H_cef))
V_fock_ed = evecs
energies = evals + offset
```

Validation:
- `Hion` 必须 Hermitian。
- `V_fock_ed` 的列必须 orthonormal。
- `h_cef` 必须 Hermitian，并使用 canonical f spin-orbital order。

## 3) 简并子空间正则化（MUST）
MUST:
- Energy-degenerate columns 必须变为 deterministic。
- 每个 degenerate energy cluster 内，对角化 `J2`。
- 每个得到的 `J2` cluster 内，对角化 `Jz`。
- 按 project state-vector convention 固定 vector phases。

Code form:
```text
cluster by energy
  diagonalize J2 in cluster
  cluster by J2
    diagonalize Jz
    sort by M
```

Validation:
- 报告的 `J` 必须与 `J(J+1)` 兼容。
- 报告的 `M` 必须 half-integer compatible，并在每个 multiplet 内有序。

## 4) 输出（MUST）
MUST:
- 输出列是 canonical-Fock-to-IONED transforms。
- 必需 arrays:
  - `V_fock_ed`
  - `energies`
- 可选 diagnostic arrays:
  - `J`
  - `M`
  - `energy_group`
- Metadata 必须包含 labels、`basis_id`、`state_order_id`、
  `orbital_order_id`、`n_ele` 和 physics parameters。

Code form:
```text
IONED = {
  V_fock_ed(alpha_fock,state),
  energies(state),
  labels(state)
}
```
