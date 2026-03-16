# 02-00-LOCAL_HAMILTONIAN

本文件定义局域单离子模型的分解与跨项契约。
参考文献（模型层只在此文件保留一次）：
- [Nature Computational Science (2024), s43246-024-00634-w](https://www.nature.com/articles/s43246-024-00634-w)

## 1) 局域哈密顿量分解（MUST）
MUST:
- 仅使用三项可加模型。
- 各项物理语义与 `02-01/02-02/02-03` 保持一致。

Math:
$$
H_{\mathrm{local}} = H_{\mathrm{int}} + H_{\mathrm{soc}} + H_{\mathrm{cef}}.
$$

Code form:
```text
H_local = H_int + H_soc + H_cef
```

Index:
- `int`：库仑多重态项。
- `soc`：原子自旋轨道耦合项。
- `cef`：晶体场项。

Validation:
- 若引入额外局域项，必须声明新的 model scheme/version。

## 2) 库仑项契约（MUST）
MUST:
- 使用 Slater-Condon 参数 `F0/F2/F4/F6`。
- 在绝对能量中保留 `F0` 扇区平移。

Math:
$$
H_{\mathrm{int}} = F^0\,\frac{n(n-1)}{2} + F^2\hat O_2 + F^4\hat O_4 + F^6\hat O_6.
$$

Code form:
```text
H_int = F0 * n*(n-1)/2 + F2*O2 + F4*O4 + F6*O6
```

Index:
- `n`：固定电子数扇区电子数。
- `O2/O4/O6`：分秩库仑算符（见 `./standards/en/02-hamiltonian/02-01-HINT.md`）。

Validation:
- 同一 `n` 扇区内，`F0` 对所有态贡献相同平移。

## 3) SOC 项契约（MUST）
MUST:
- 使用单一 SOC 强度参数 `zeta`。
- 算符形式以 `./standards/en/02-hamiltonian/02-02-HSOC.md` 为准。

Math:
$$
H_{\mathrm{soc}} = \zeta\sum_i \mathbf l_i\cdot\mathbf s_i.
$$

Code form:
```text
H_soc = zeta * sum_i (l_i · s_i)
```

Index:
- `zeta`：SOC 强度。

Validation:
- SOC 矩阵必须厄米。

## 4) CEF 项契约（MUST）
MUST:
- 使用 Stevens 算符展开。
- 仅允许 `./standards/en/02-hamiltonian/02-03-HCEF.md` 中定义的对称分支。

Math:
$$
H_{\mathrm{cef}} = \sum_{k,q} B_k^q O_k^q.
$$

Code form:
```text
H_cef = sum_{k,q} B[k,q] * O[k,q]
```

Index:
- `B_k^q`：CEF 参数（与其它参数同一能量单位）。
- `O_k^q`：Stevens 算符。

Validation:
- `B_k^q`、`F^k`、`zeta` 必须共享同一单位体系。

## 5) 层级与求解顺序（MUST）
MUST:
- 使用 RS 风格层级
  $|H_{\mathrm{int}}| > |H_{\mathrm{soc}}| \gg |H_{\mathrm{cef}}|$。
- 严格按 `Hint -> Hsoc -> Hcef` 顺序计算。

Code form:
```text
step1: diagonalize(H_int) -> LSMS
step2: diagonalize(H_soc, within fixed LS blocks) -> LSJM
step3: apply(H_cef, on selected SOC manifold)
```

Validation:
- 若 step2 跨 `LS` 分块混合，必须声明新 scheme/version。

## 6) 能量参考规则（MUST）
MUST:
- 保持可加的绝对能量记账。
- 若阶段输出相对能量，必须在元数据里写明参考零点。

Math:
$$
E_{\mathrm{total}} = E_{\mathrm{int}} + E_{\mathrm{soc}} + E_{\mathrm{cef}}.
$$

Code form:
```text
E_total = E_int + E_soc + E_cef
```

Validation:
- 跨文件比较必须使用一致的绝对能量约定。

## 7) SOPT 接口说明（MUST）
MUST:
- SOPT 中间态能量取自 `H_int + H_soc`。
- Kramer 态能量零点与有效 `U/J` 约定属于 SOPT 规范（`04-00/04-01/04-02`），不在本文件定义。
