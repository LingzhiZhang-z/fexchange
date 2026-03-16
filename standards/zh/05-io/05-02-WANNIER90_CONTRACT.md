# 05-02-WANNIER90_CONTRACT

本文件定义当 hopping/CEF/Kramers 数据来自 DFT+Wannier90 时的输入契约。

## 1) 适用范围与模式（MUST）
MUST:
- 只允许两类来源：
  - `source_mode = literature_params`
  - `source_mode = wannier90`
- 本文件约束 `source_mode = wannier90`。
- 确定性文件解析与原子/轨道/自旋映射规则由
  `./standards/en/05-io/05-03-WANNIER90_PARSING.md` 定义。

Code form:
```text
if source_mode != "wannier90": skip this file
```

Validation:
- 同一工件中禁止混用来源模式（除非显式标注）。

## 2) Wannier90 必需元数据（MUST）
MUST:
- 必须记录：
  - `soc_mode in {"with_soc","without_soc"}`
  - `orbital_basis = real_harmonic_default_w90`
  - `orbital_order_id`
  - `energy_unit`（默认 `eV`）
  - `spin_completion_rule`（当 `soc_mode="without_soc"` 且需构造含自旋张量时必填）
- 必须记录原子索引绑定：
  - `f_site_i`, `f_site_j`
  - `f_site_i_cell`, `f_site_j_cell`（整数三元组）
  - `ligand_indices`
  - `ligand_cells`（与 `ligand_indices` 对齐的整数三元组列表）
  - `all_wannier_atom_indices`

Code form:
```text
required = {
  soc_mode, orbital_basis, orbital_order_id, energy_unit,
  f_site_i, f_site_j, f_site_i_cell, f_site_j_cell,
  ligand_indices, ligand_cells, all_wannier_atom_indices
}
if soc_mode == "without_soc" and spinful_required:
  require spin_completion_rule
require len(f_site_i_cell) == 3 and len(f_site_j_cell) == 3
require len(ligand_cells) == len(ligand_indices)
for cell in ligand_cells: require len(cell) == 3
```

Validation:
- 任一必需字段缺失都必须硬失败。

## 2.1) 晶胞平移契约（MUST）
MUST:
- site-image hopping 的索引必须使用相对晶胞位移，不能依赖绝对 `000` 选胞。
- 设整数晶胞向量为：
  `T_i = f_site_i_cell`，`T_j = f_site_j_cell`，
  对配体 `o` 有 `T_o = ligand_cells[o]`。
- 定义相对位移：

Math:
$$
\mathbf R_{ij} = T_j - T_i,\qquad
\mathbf R_{io} = T_o - T_i,\qquad
\mathbf R_{jo} = T_o - T_j.
$$

- 直接 $f$-$f$ hopping 必须在 `wannier90_hr.dat` 的 $\mathbf R_{ij}$ 处取样。
- 配体介导项必须在 $\mathbf R_{io}$ 取样 $i\to o$，在 $\mathbf R_{jo}$ 取样 $j\to o$。
- onsite 项使用所选局域 site image 下的 $\mathbf R=(0,0,0)$。
- 若某个 $\mathbf R$ 处条目缺失，实现可使用厄米补全：

Math:
$$
H_{mn}(\mathbf R)=H_{nm}^{\ast}(-\mathbf R).
$$

- 若直接读取与厄米补全都失败，必须硬失败。
- 必须满足共同晶胞平移不变性：
  若对全部选中 site 同时加同一个整数向量，结果应不变。

Code form:
```text
R_ij = f_site_j_cell - f_site_i_cell
R_io[o] = ligand_cells[o] - f_site_i_cell
R_jo[o] = ligand_cells[o] - f_site_j_cell

H_mn_R = fetch_hr(m, n, R)
if missing(H_mn_R):
  H_mn_R = conj(fetch_hr(n, m, -R))
if still_missing: fail(FXE-W90-002)
```

Validation:
- metadata 必须记录 `{f_site_i_cell, f_site_j_cell, ligand_cells, R_ij}`。
- 缺失晶胞绑定或三元组长度非法必须硬失败。

## 3) 配体 p 介导的有效 f-f hopping（MUST）
MUST:
- 必须区分 `without_soc` 与 `with_soc` 两套公式。
- 有效 hopping 由两部分构成：
  - 直接 `f-f` hopping 项 `t^{(0)}`
  - 配体介导的二阶修正项
- 必须先在 Wannier 原始基完成配体约化；real/complex/cubic 基变换只能在
  得到约化后的 `h/t` 之后执行。
- 本节 hopping 矩阵元的采样必须遵循第 2.1 节相对晶胞位移规则。
- 物理精确形式的分母应按通道逐一取值；全局平均分母是第 4 节定义的可选近似。
- 统一指标约定：
  - `i, i^{\prime}`：两个目标 `f` 轨道 site；
  - `u,v`：site `i, i^{\prime}` 上的 `f` 轨道局域基指标；
  - `o`：配体 site 指标；
  - `p`：配体 site `o` 上的轨道（通道）指标。
- 本文件符号作用域是局部的：
  - 这里的 `p` 只表示配体轨道指标；
  - 不应与 `04-00/04-01/04-02` 中的 SOPT 记号 `p,p',q,q'` 混淆，后者表示约化后
    的 site 绑定 `f` 轨道指标。

### 3.1) `without_soc` 公式（MUST）
MUST:
- 在 `without_soc` 下，不存在自旋翻转项。
- 在该模式下，`u/v/p` 只含轨道指标（不含自旋）。
- hopping 写为 `t_{i u, o p,\sigma}`，自旋在哈密顿量中显式求和。
- 有效 hopping 记为 `\tilde t = t^{(0)} + \delta t`。

Math:
$$
\tilde t^{\mathrm{nsoc}}_{i u,\,i^{\prime} v,\sigma}
=
t^{(0)}_{i u,\,i^{\prime} v,\sigma}
+
\sum_{o,p}
\frac{
 t_{i u,\,o p,\sigma}\,
 t_{i^{\prime} v,\,o p,\sigma}^{*}
}{
\Delta_{p-uv}
}.
$$

Math:
$$
H_{\mathrm{hop},ii^{\prime}}^{(\mu),\mathrm{nsoc}}
=
\sum_{u,v,\sigma}
\left(
 \tilde t^{\mathrm{nsoc}}_{i u,\,i^{\prime} v,\sigma}\,
 c^\dagger_{i u \sigma}c_{i^{\prime} v \sigma}
 + \mathrm{h.c.}
\right).
$$

Code form:
```text
for i, i_prime in f_site_pairs:
  for u in f_basis[i]:
    for v in f_basis[i_prime]:
      for sigma in spins:
        t_tilde_nsoc[i,u,i_prime,v,sigma] = t0_nsoc[i,u,i_prime,v,sigma]
        for o in ligand_sites:
          for p in ligand_orbitals[o]:
            t_tilde_nsoc[i,u,i_prime,v,sigma] += (
              t_nsoc[i,u,o,p,sigma] * conj(t_nsoc[i_prime,v,o,p,sigma]) / Delta_puv[p,u,v]
            )
        H_hop_nsoc += t_tilde_nsoc[i,u,i_prime,v,sigma] * cdag(i,u,sigma) * c(i_prime,v,sigma) + h.c.
```

### 3.2) `with_soc` 公式（MUST）
MUST:
- 在 `with_soc` 下，`u/v/p` 都是“轨道+自旋”复合指标。
- 不再单独写显式 `\sigma` 求和。
- hopping 写为 `t_{i u, o p}`。
- 有效 hopping 记为 `\tilde t = t^{(0)} + \delta t`。

Math:
$$
\tilde t^{\mathrm{soc}}_{i u,\,i^{\prime} v}
=
t^{(0)}_{i u,\,i^{\prime} v}
+
\sum_{o,p}
\frac{
t_{i u,\,o p}\,
t_{i^{\prime} v,\,o p}^{*}
}{
\Delta_{p-uv}
}.
$$

Math:
$$
H_{\mathrm{hop},ii^{\prime}}^{(\mu),\mathrm{soc}}
=
\sum_{u,v}
\left(
\tilde t^{\mathrm{soc}}_{i u,\,i^{\prime} v}\,
c^\dagger_{i u}c_{i^{\prime} v}
+ \mathrm{h.c.}
\right).
$$

Code form:
```text
for i, i_prime in f_site_pairs:
  for u in f_basis_soc[i]:
    for v in f_basis_soc[i_prime]:
      t_tilde_soc[i,u,i_prime,v] = t0_soc[i,u,i_prime,v]
      for o in ligand_sites:
        for p in ligand_basis_soc[o]:
          t_tilde_soc[i,u,i_prime,v] += (
            t_soc[i,u,o,p] * conj(t_soc[i_prime,v,o,p]) / Delta_puv[p,u,v]
          )
      H_hop_soc += t_tilde_soc[i,u,i_prime,v] * cdag(i,u) * c(i_prime,v) + h.c.
```

### 3.3) `without_soc` 的自旋补全规则（MUST）
MUST:
- 当 Wannier90 输入为 `without_soc` 且下游需要含自旋张量时：
  - 上自旋块使用 Wannier90 原始输出；
  - 下自旋块使用上自旋块的复共轭；
  - 自旋翻转块置零。

Math:
$$
h^{\uparrow\uparrow}=h^{\mathrm{w90}},\qquad
h^{\downarrow\downarrow}=\left(h^{\mathrm{w90}}\right)^*,\qquad
h^{\uparrow\downarrow}=h^{\downarrow\uparrow}=0.
$$

Code form:
```text
if soc_mode == "without_soc" and spinful_required:
  h_upup = h_w90
  h_dndn = conj(h_w90)
  h_updn = 0
  h_dnup = 0
```

Validation:
- 结果 hopping 矩阵必须在 `eps_herm` 阈值内厄米。

## 4) `\Delta_{p-uv}` 规则（MUST）
MUST:
- 允许两种输入模式：
  - `delta_mode = manual`
  - `delta_mode = from_onsite`
- `manual` 模式契约：
  - `delta_manual_kind in {"channelwise","global_mean"}`。
  - 若 `delta_manual_kind = "global_mean"`：
    提供标量 `delta_manual_value`，单位为 `energy_unit`。
  - 若 `delta_manual_kind = "channelwise"`：
    提供 `delta_manual_file`（NPZ），包含
    `Delta_puv[p,u,v]`，shape 为 `(n_p,n_u,n_v)`，单位为 `energy_unit`。
  - `delta_reduction` 必须与 manual kind 一致：
    `global_mean <-> global_mean`，`channelwise <-> channelwise`。
- 分母记号采用：
  - 项目/文献统一记号：`\Delta_{p-uv}`
- 在第 3 节公式中，分母写为 `\Delta_{p-uv}`，分子按 `o,p` 求和：
  - `\sum_{o,p} t_{i u,o p} t_{i^{\prime} v,o p}^{*} / \Delta_{p-uv}`。
- `from_onsite` 时，先构造含配体 site 的调和平均分母：

Math:
$$
\Delta_{u}^{(o,p)}=\epsilon_{i u}-\epsilon_{o p},\quad
\Delta_{v}^{(o,p)}=\epsilon_{i^{\prime} v}-\epsilon_{o p},
$$

Math:
$$
\Delta_{p-uv}^{(o)}
=
\frac{2\Delta_u^{(o,p)}\Delta_v^{(o,p)}}{\Delta_u^{(o,p)}+\Delta_v^{(o,p)}}.
$$

Code form:
```text
for i, i_prime in f_site_pairs:
  for o in ligand_sites:
    for p in ligand_orbitals[o]:
      for u in f_basis[i]:
        for v in f_basis[i_prime]:
          du = eps_f[i,u] - eps_lig[o,p]
          dv = eps_f[i_prime,v] - eps_lig[o,p]
          Delta_puv_o[o,p,u,v] = 2 * du * dv / (du + dv)
```

对 `with_soc`，同一公式仍成立，但 `u/v/p` 为复合轨道-自旋指标。

然后对 `o` 做约简，得到第 3 节中使用的分母：

Math:
$$
\Delta_{p-uv}
=
\mathrm{reduce}_{o}\!\left[\Delta_{p-uv}^{(o)}\right].
$$

Code form:
```text
for p,u,v in channel_indices:
  Delta_puv[p,u,v] = reduce_over_o(Delta_puv_o[:,p,u,v], mode=delta_reduction)
```

- 通道约简策略：
  - `delta_reduction = channelwise`：在 `p,u,v` 通道上保留 `\Delta_{p-uv}`。
  - `delta_reduction = global_mean`：将全部通道分母替换为同一个平均值（近似）。

若使用 `global_mean`：

Math:
$$
\bar\Delta_{p-uv}
=
\frac{1}{N_{\Delta}}
\sum_{u,v,p}\Delta_{p-uv}.
$$

Code form (global mean):
```text
delta_bar_puv = mean(Delta_puv[p,u,v] for p,u,v in channel_indices)
```

Code form (mode selection):
```text
if delta_mode == "manual" and delta_manual_kind == "global_mean":
  require delta_reduction == "global_mean"
  Delta_puv[p,u,v] = delta_manual_value
elif delta_mode == "manual" and delta_manual_kind == "channelwise":
  require delta_reduction == "channelwise"
  Delta_puv = load_npz(delta_manual_file)["Delta_puv"]    # shape (n_p,n_u,n_v)
elif delta_mode == "from_onsite" and delta_reduction == "channelwise":
  use per_channel_harmonic_deltas
elif delta_mode == "from_onsite" and delta_reduction == "global_mean":
  delta_puv = mean(per_channel_harmonic_deltas)
```

Validation:
- metadata 必须记录 `delta_mode`、`delta_reduction`、逐通道 delta、均值与标准差。
- manual 模式下，缺失/非法手动字段或 shape 不匹配必须硬失败。
- 所有分母必须是有限值并满足 `abs(Delta_puv) > eps_zero`。

## 5) 实基到复基变换（MUST）
MUST:
- Wannier90 默认 `f` 轨道按固定实球谐顺序处理。
- 基组顺序必须遵循本项目约定：
  - real 基顺序：`m = [0, 1, -1, 2, -2, 3, -3]`
  - complex 基顺序：`m = [-3, -2, -1, 0, 1, 2, 3]`
- `U_r2c` 必须由 `orbital_order_id` 唯一确定（禁止运行时猜测符号/相位）。
- 本节输入（`h_real`, `t_real`）必须是第 3/4 节得到的约化结果，而不是未约化
  的 Wannier 原始子块。

Math:
$$
U_{\mathrm{r2c}}:
\text{ 实球谐基到复球谐基的确定性映射 }.
$$

Math:
$$
h_{\mathrm{complex}} = U_{\mathrm{r2c}}^{T} h_{\mathrm{real}} U_{\mathrm{r2c}}^{*},\qquad
t_{\mathrm{complex}} = U_{\mathrm{r2c}}^{T} t_{\mathrm{real}} U_{\mathrm{r2c}}^{*}.
$$

Code form:
```text
U_r2c = build_U_r2c(orbital_order_id, spinor_flag)
U_c2r = inv(U_r2c)

require is_unitary(U_r2c, eps_unitary)

h_complex = U_r2c.T @ h_real @ U_r2c.conj()
t_complex = U_r2c.T @ t_real @ U_r2c.conj()
```

Code form（执行顺序）:
```text
# 1) 先在 Wannier 原始基做约化（第 3/4 节）
h_real, t_real = reduce_ligand_channels(wannier_blocks, delta_policy)
# 2) 再做基变换（本节）
h_complex = U_r2c.T @ h_real @ U_r2c.conj()
t_complex = U_r2c.T @ t_real @ U_r2c.conj()
```

Code form（显式模板，非 spinor）:
```text
tmp = np.array([
                 [0, 0, 0, 1, 0, 0, 0],
  (1/sqrt(2))  * [0, 0, 1, 0,-1, 0, 0],
  (1j/sqrt(2)) * [0, 0, 1, 0, 1, 0, 0],
  (1/sqrt(2))  * [0, 1, 0, 0, 0, 1, 0],
  (1j/sqrt(2)) * [0, 1, 0, 0, 0,-1, 0],
  (1/sqrt(2))  * [1, 0, 0, 0, 0, 0,-1],
  (1j/sqrt(2)) * [1, 0, 0, 0, 0, 0, 1],
], dtype=complex)
U_r2c = tmp
if SPINOR:
  U_r2c = kron(tmp, I2)
```

MUST（complex <-> cubic）:
- 必须定义唯一确定的 complex->cubic 变换 `U_c2cub`。
- cubic 基顺序必须遵循本项目约定：
  `[\xi, \eta, \zeta, A, \alpha, \beta, \gamma]`。
- `U_cub2c = inv(U_c2cub)`。

Math:
$$
h_{\mathrm{cubic}} = U_{\mathrm{c2cub}}^{T} h_{\mathrm{complex}} U_{\mathrm{c2cub}}^{*},\qquad
t_{\mathrm{cubic}} = U_{\mathrm{c2cub}}^{T} t_{\mathrm{complex}} U_{\mathrm{c2cub}}^{*}.
$$

Code form:
```text
U_c2cub = build_U_c2cub(orbital_order_id, spinor_flag)
U_cub2c = inv(U_c2cub)
require is_unitary(U_c2cub, eps_unitary)

h_cubic = U_c2cub.T @ h_complex @ U_c2cub.conj()
t_cubic = U_c2cub.T @ t_complex @ U_c2cub.conj()
```

Code form（显式模板，非 spinor）:
```text
sq3 = sqrt(3); sq5 = sqrt(5)
tmp_c2cub = np.array([
  (1/4)        * [-sq3, 0, -sq5, 0,  sq5, 0,  sq3],
  (1j/4)       * [-sq3, 0,  sq5, 0,  sq5, 0, -sq3],
  (1/sqrt(2))  * [   0, 1,    0, 0,    0, 1,    0],
  (1j/sqrt(2)) * [   0, 1,    0, 0,    0,-1,    0],
  (1/4)        * [ sq5, 0, -sq3, 0,  sq3, 0, -sq5],
  (1j/4)       * [-sq5, 0, -sq3, 0, -sq3, 0, -sq5],
                 [   0, 0,    0, 1,    0, 0,    0],
], dtype=complex)
U_c2cub = tmp_c2cub
if SPINOR:
  U_c2cub = kron(tmp_c2cub, I2)
```

Code form（派生变换）:
```text
U_r2cub = U_r2c @ U_c2cub
U_cub2r = inv(U_r2cub)
```

Validation:
- `U_r2c` 的版本/id 必须写入 metadata。
- `U_c2cub` 的版本/id 必须写入 metadata。
- 变换前后必须满足迹不变与厄米容差检查。

## 6) CEF-from-Wannier 拟合契约（MUST）
MUST:
- CEF 参数提取按顺序执行：
1. 从 Wannier90 构造 onsite 的 `f` 子块。
2. 执行 real->complex 变换。
3. 投影到 LSJM 的 SOC 低能目标子空间。
4. 按 `./standards/en/02-hamiltonian/02-03-HCEF.md` 做 Stevens 拟合。

拟合质量必须检验：

Math:
$$
r_{\mathrm{fit}}=
\frac{\|H_{\mathrm{proj}}-H_{\mathrm{Stevens}}\|_F}
{\max(\|H_{\mathrm{proj}}\|_F,\varepsilon_{\mathrm{zero}})}.
$$

Code form:
```text
require r_fit <= eps_map
```

Validation:
- 拟合失败必须硬失败，禁止静默回退。

## 7) C3v 轴约定（MUST）
MUST:
- 本项目 `C3v` 约定：
  - `c` 轴为 `z`
  - `a` 轴为 `y`
- 输出中的轴标签必须遵循该约定。

Code form:
```text
if symmetry == "C3v":
  axis_c = "z"
  axis_a = "y"
```

Validation:
- metadata 必须包含显式轴映射。
