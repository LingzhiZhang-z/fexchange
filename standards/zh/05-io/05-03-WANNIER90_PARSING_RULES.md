# 05-03-WANNIER90_PARSING_RULES

本文件定义 Wannier90 输入的确定性解析与映射规则。
本文件是 `./standards/en/05-io/05-02-WANNIER90_INPUT_CONTRACT.md` 的解析配套规范。

## 1) 作用范围（MUST）
MUST:
- 仅在 `hopping_source = "wannier90"` 时生效。
- 覆盖文件解析、原子/轨道/自旋映射、单位归一与顺序校验。
- 不定义 SOPT 收缩公式本身。

Code form:
```text
if hopping_source != "wannier90": skip this file
```

Validation:
- 该模式下本文件所有闸门均为强制项。

## 2) 必需输入文件（MUST）
MUST:
- 必需文件：
  - `wannier90_hr.dat`
  - `wannier.win`
- 可存在辅助文件，但不得替代必需文件。

Code form:
```text
required_w90_files = {hr_path, win_path}
```

Validation:
- 文件缺失使用 `FXE-W90-001`。

### 2.1) `wannier90_hr.dat` 文件格式（MUST）
文件布局：
```text
第 1 行:    注释字符串（自由格式，解析器忽略）
第 2 行:    num_wann                        （整数）
第 3 行:    nrpts                           （整数，R 矢量数）
接下来 ceil(nrpts/15) 行:
           简并度权重，每行最多 15 个整数
数据块（nrpts * num_wann^2 行，每行格式）:
           R1  R2  R3  i  j  Re(H_ij(R))  Im(H_ij(R))
```

每个矩阵元的物理定义：

Math:
$$
H_{ij}(\mathbf R)
=
\langle \mathbf 0,\,i \rvert \hat H \lvert \mathbf R,\,j \rangle,
$$

其中 $\lvert \mathbf R,\,j\rangle$ 为中心在格矢 $\mathbf R$ 处的第 $j$ 个
Wannier 函数，$\lvert \mathbf 0,\,i\rangle$ 为原胞中第 $i$ 个 Wannier 函数。
当 $\mathbf R=\mathbf 0$ 且 $i=j$ 时，$H_{ii}(\mathbf 0)$ 为在位能
$\epsilon_i=\langle \mathbf 0,\,i\rvert\hat H\lvert\mathbf 0,\,i\rangle$。

解析规则：
- `R = (R1, R2, R3)` 为整数格矢指标。
- 在位块对应 `R = (0,0,0)`。
- `i, j` 为 **1 起始** 的 Wannier 函数指标。
- 哈密顿量值：`H_ij(R) = Re + i*Im`，能量单位由 `energy_unit` 声明。
- 每个 `H_ij(R)` 必须除以对应 R 矢量的简并度权重。
- 数据块排列：外层循环 R 矢量，内层循环 `(i,j)` 其中 `j` 为快指标。

Code form:
```text
for r_idx in range(nrpts):
  for j in range(1, num_wann+1):
    for i in range(1, num_wann+1):
      read R1, R2, R3, i, j, re_h, im_h
      H[R][(i-1, j-1)] = complex(re_h, im_h) / weight[r_idx]
```

### 2.2) `wannier.win` 最少必需字段（MUST）
解析器至少须提取：
- `num_wann`：Wannier 函数数目。
- `begin projections` / `end projections`：原子位点与轨道类型。
- `begin atoms_frac` / `end atoms_frac`（或 `atoms_cart`）：原子位置。
- `begin unit_cell_cart` / `end unit_cell_cart`：晶格矢量（R 矢量解释所需）。

`.win` 文件使用 Fortran 风格 key-value 格式（`keyword = value`），
块节由 `begin`/`end` 标签界定。

Code form:
```text
num_wann     = parse_int(win, "num_wann")
projections  = parse_block(win, "projections")
atoms        = parse_block(win, "atoms_frac") or parse_block(win, "atoms_cart")
unit_cell    = parse_block(win, "unit_cell_cart")
```

Validation:
- 缺少 `num_wann` 或投影块为 `FXE-W90-001`。
- `.win` 中 `num_wann` 须与 `_hr.dat` 中一致；不一致为 `FXE-W90-003`。

## 3) 原子 site 选择规则（MUST）
MUST:
- `f_site_i != f_site_j`。
- `f_site_i_cell` 与 `f_site_j_cell` 必须是整数三元组。
- `ligand_indices` 可以为空（仅保留直接 `f-f` 项，不做配体介导修正）。
- 若非空，`ligand_indices` 不得包含 `f_site_i/f_site_j`。
- `ligand_cells` 必须提供，且与 `ligand_indices` 一一对应（每项均为整数三元组）。
- `all_wannier_atom_indices` 必须覆盖所有被选 site。
- 任一 site 列表中禁止重复原子索引。

Code form:
```text
require f_site_i != f_site_j
require disjoint({f_site_i,f_site_j}, set(ligand_indices))
require subset({f_site_i,f_site_j} ∪ ligand_indices, all_wannier_atom_indices)
require len(f_site_i_cell) == 3 and len(f_site_j_cell) == 3
require len(ligand_cells) == len(ligand_indices)
for cell in ligand_cells: require len(cell) == 3

R_ij = f_site_j_cell - f_site_i_cell
for each ligand o:
  R_io[o] = ligand_cells[o] - f_site_i_cell
  R_jo[o] = ligand_cells[o] - f_site_j_cell
```

Validation:
- 违规使用 `FXE-W90-002`。

Code form:
```text
if len(ligand_indices) == 0:
  use_direct_ff_only = true
  ligand_correction = 0
```

Validation:
- 晶胞三元组绑定非法或配体长度不一致使用 `FXE-W90-002`。
- 派生相对位移向量必须写入 metadata。

### 3.1) Relative-R 取样与厄米补全规则（MUST）
MUST:
- 所有 hopping 读取都必须使用第 3 节派生出的相对格矢
  （`R_ij`, `R_io`, `R_jo`），不论任何选中 site 是否位于 `000`。
- 对缺失直读项，解析器/加载器必须支持厄米补全：

Math:
$$
H_{mn}(\mathbf R)=H_{nm}^{\ast}(-\mathbf R).
$$

- 若直读与厄米补全都不可用，必须硬失败。

Code form:
```text
def fetch_H(m, n, R):
  if exists(H[R][m,n]): return H[R][m,n]
  if exists(H[-R][n,m]): return conj(H[-R][n,m])
  fail(FXE-W90-002)
```

Validation:
- 必需 `R` 条目缺失且无法厄米补全时，使用 `FXE-W90-002`。
- 建议执行共同平移不变性检查：
  全部选中晶胞同时加同一向量后，派生张量应不变。

## 4) 轨道映射规则（MUST）
MUST:
- `f` 通道必须显式给出轨道顺序 id（`orbital_order_id`）。
- 默认实球谐 `f` 顺序：
  `m = [0, 1, -1, 2, -2, 3, -3]`。
- 解析器必须输出显式索引映射：
  - `map_f_i[u] -> (atom, orbital, spin?)`
  - `map_f_j[v] -> (atom, orbital, spin?)`
  - `map_lig[o,p] -> (atom, orbital, spin?)`
- 禁止模糊轨道标签映射。

Code form:
```text
build map_f_i, map_f_j, map_lig with explicit deterministic ordering
```

Validation:
- 顺序/映射不匹配使用 `FXE-W90-003`。

## 5) 自旋映射规则（MUST）
MUST:
- `soc_mode = "with_soc"`：
  - 局域指标使用轨道+自旋复合指标。
  - 自旋翻转项直接来自 Wannier 输入。
- `soc_mode = "without_soc"`：
  - 局域指标仅含轨道。
  - 含自旋扩展采用：
    - up 块 = Wannier 原始块
    - down 块 = up 块复共轭
    - 自旋翻转块 = 0

Code form:
```text
if soc_mode == "with_soc":
  use composite (orbital,spin) indices directly
else:
  h_upup = h_w90
  h_dndn = conj(h_w90)
  h_updn = 0
  h_dnup = 0
```

Validation:
- 自旋模式与输入不一致使用 `FXE-W90-002`。

## 6) 单位归一规则（MUST）
MUST:
- 内部统一单位为 `eV`。
- 输入单位必须显式声明（`energy_unit`）。
- 若输入单位不是 `eV`，必须在任何物理步骤前做确定性换算。

Code form:
```text
energy_scale = unit_to_eV(energy_unit)
H_eV = energy_scale * H_input
```

Validation:
- 单位缺失或未知使用 `FXE-W90-003`。

## 7) 厄米性与顺序校验（MUST）
MUST:
- 解析后的 onsite/hopping 块在适用情形下必须通过厄米性阈值检查。
- metadata 必须输出顺序 id：
  - `atom_order_id`
  - `orbital_order_id`
  - `spin_order_id`
  - `ligand_order_id`

Code form:
```text
check_hermitian(H, eps_herm)
emit order_ids in meta
```

Validation:
- 校验失败使用 `FXE-NUM-001` 或 `FXE-W90-003`。

## 8) 解析输出契约（MUST）
MUST:
- 解析输出载荷必须包含：
  - 归一到 `eV` 的 onsite/hopping 子块
  - 显式映射表（`map_f_i`, `map_f_j`, `map_lig`）
  - 晶胞绑定与派生相对位移向量（`f_site_i_cell`, `f_site_j_cell`,
    `ligand_cells`, `R_ij`, `R_io`, `R_jo`）
  - 顺序 id
  - 源文件 hash

Code form:
```text
w90_parsed = {
  H_blocks_eV, map_f_i, map_f_j, map_lig,
  f_site_i_cell, f_site_j_cell, ligand_cells, R_ij, R_io, R_jo,
  order_ids, file_hashes
}
```

Validation:
- 缺失映射表使用 `FXE-W90-002`。
