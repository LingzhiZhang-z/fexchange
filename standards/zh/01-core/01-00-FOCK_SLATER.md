# 01-00-FOCK_SLATER

本文件定义所有模块共享的 Fock/Slater 底层强制约定。

## 1) Fock 空间（MUST）
MUST:
- 单离子 f 壳层固定 `n_orb = 14` 个自旋轨道。
- 电子数扇区固定 `n = 0..14`。

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
- 所有扇区对象必须满足 `0 <= n <= 14`。

## 2) 轨道索引映射（MUST）
MUST:
- 轨道索引 `p = 0..13`。
- 固定映射顺序：`m=-3..3`，每个 `m` 的自旋顺序为 `(-1/2, +1/2)`。

Code form:
```text
p -> (m, sigma)
m in [-3,-2,-1,0,1,2,3]
sigma order = [-1/2, +1/2] for each m
```

Validation:
- 态、算符、hopping 张量必须使用同一映射。

## 3) 单电子基底（MUST）
MUST:
- 物理基底态为 $\lvert l=3,m,\sigma\rangle$。
- 对外接口必须可追溯到该默认复球谐基底。

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
- 若内部基底不同，必须提供显式变换元数据。

## 4) 矩阵与向量约定（MUST）
MUST:
- 矩阵元采用 bra-ket 约定。
- 态向量采用列向量约定。

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
- 厄米共轭使用 `A_dag = (A.conj()).T`。

## 5) Bitstring 编码（MUST）
MUST:
- 行列式编码为非负整数 `det`。
- bit `p` 对应第 2 节轨道 `p`。

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
- 对已占据轨道做产生、对空轨道做湮灭都必须返回零。

## 6) Slater 基底身份（MUST）
MUST:
- 扇区 `n` 内行列式按整数升序（`lex_v1`）。
- `basis_id` 格式固定。

Code form:
```text
basis_id = f"fock{n_orb}_n{n_ele}_{det_order}_v{major}"
example: fock14_n5_lex_v1
```

Validation:
- 跨文件读取时 `basis_id` 不一致必须失败。
- `basis_id` 不得编码 hopping/Kramers/截断信息。

## 7) 费米子符号约定（MUST）
MUST:
- 产生与湮灭都使用“索引下方占据数奇偶”规则。

Math:
$$
c_p^\dagger\lvert det\rangle=
\begin{cases}
0,& n_p=1\\
(-1)^{N_{<p}(det)}\lvert det\cup\{p\}\rangle,& n_p=0
\end{cases}
$$

Math:
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
- 相邻扇区湮灭矩阵必须是产生矩阵的厄米共轭。

## 8) 态/能量数组契约（MUST）
MUST:
- 态矩阵按列存储。
- 能量数组与列索引一一对应。

Math:
$$
V_{\mathrm{fock}}\in\mathbb C^{d_{\mathrm{fock}}\times n_{\mathrm{states}}},
\qquad
V_{\mathrm{fock}}^\dagger V_{\mathrm{fock}}=I
\ \text{（正交归一态集）}.
$$

Code form:
```text
V_fock.shape = (dim_fock, n_states)
energy[a] <-> column a
```

Validation:
- 列数必须与标签和能量数组长度一致。

## 9) 单位与可追溯性（MUST）
MUST:
- 对外输出必须记录 `basis_id`，必要时记录轨道顺序元数据。
- 能量单位必须显式写入元数据。

Code form:
```text
meta = {basis_id, orbital_order_id?, unit, ...}
```

Validation:
- 依赖隐含单位的输出视为无效。
