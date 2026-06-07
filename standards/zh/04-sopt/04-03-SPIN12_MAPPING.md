# 04-03-SPIN12_MAPPING

本文件定义从 `04-02-RUNTIME_CONTRACTION` 输出到赝自旋-$\tfrac{1}{2}$模型的后处理映射。
模块读取 $\mathrm{Heff}_{cd,ab}^{(\mu)}$ 并投影为自旋耦合参数。
磁盘 I/O 布局与格式由 `./standards/en/05-io/05-00-IO.md` 统一定义。
写作形式遵循 `./standards/en/00-meta/00-00-SPEC_WRITING_CONVENTION.md`。

## 0) 适用范围（MUST）
- 输入来自 `./standards/en/04-sopt/04-02-RUNTIME_CONTRACTION.md` 的 SOPT final-$L3$
  输出，或来自 `./standards/en/04-fopt/04-00-FOPT_FORMALISM.md` 的 FOPT `L3`
  total/process projected `h_eff_4` 输出。
- 本模块是后处理映射，不改变 `L0..L3`。
- 仅在每个 site 的低能空间为二维（Kramers 赝自旋-$\tfrac{1}{2}$）时适用。

## 1) 输入契约（MUST）
对每个键/通道 $\mu$：
- 矩阵元
  $\left(\mathrm{Heff}^{(\mu)}\right)_{cd,ab}
  =\langle c,d \rvert \mathrm{Heff}^{(\mu)} \lvert a,b \rangle$，
- 每个 site 固定的 Kramers 基顺序（`+,-` 或等价顺序），
- 上游映射提供的基/规范（gauge）元数据。

若局域维度不是 `2 x 2`，本模块必须直接失败。

## 2) 算符基与投影规则（MUST）
定义
$\sigma^0=I_2,\sigma^x,\sigma^y,\sigma^z$，以及
$S^\alpha=\frac{1}{2}\sigma^\alpha$（$\alpha=x,y,z$）。

对单个 $\mu$ 通道，将 $\mathrm{Heff}^{(\mu)}$ 重排为
$\lvert a,b\rangle$ 基上的 $4\times4$ 矩阵并展开：

Math:
$$
\mathrm{Heff}^{(\mu)}
=
\sum_{\eta,\nu\in\{0,x,y,z\}}
C_{\eta\nu}^{(\mu)}\,
\sigma_i^\eta \otimes \sigma_j^\nu,
$$

Math:
$$
C_{\eta\nu}^{(\mu)}
=
\frac{1}{4}\,\mathrm{Tr}
\!\left[
\left(\sigma_i^\eta \otimes \sigma_j^\nu\right)
\mathrm{Heff}^{(\mu)}
\right].
$$

## 3) 自旋-$\tfrac{1}{2}$ 交换模型形式（MUST）
主输出模型必须为

Math:
$$
H_{\mathrm{spin}}^{(\mu)}
\equiv
\sum_{\alpha,\beta\in\{x,y,z\}}
J_{\alpha\beta}^{(\mu)} S_i^\alpha S_j^\beta.
$$

系数映射：

Math:
$$
J_{\alpha\beta}^{(\mu)} = 4C_{\alpha\beta}^{(\mu)}.
$$

等价迹公式：

Math:
$$
J_{\alpha\beta}^{(\mu)}
=
\mathrm{Tr}\!\left[
\left(\sigma_i^\alpha\otimes \sigma_j^\beta\right)\mathrm{Heff}^{(\mu)}
\right].
$$

### 3.1 原始交换系数命名（MUST）
主交换输出是完整矩阵 $J_{\alpha\beta}^{(\mu)}$。
分量命名为：

Math:
$$
J^{(\mu)}=
\begin{pmatrix}
J_x^{(\mu)} & J_{xy}^{(\mu)} & J_{xz}^{(\mu)}\\
J_{yx}^{(\mu)} & J_y^{(\mu)} & J_{yz}^{(\mu)}\\
J_{zx}^{(\mu)} & J_{zy}^{(\mu)} & J_z^{(\mu)}
\end{pmatrix},
$$

其中
$J_x\equiv J_{xx}$，$J_y\equiv J_{yy}$，$J_z\equiv J_{zz}$。
对角元保持原值，不做额外重映射。

### 3.2 派生分解（当前 Runtime 范围外）
当前 runtime 只导出原始 exchange matrix $J_{\alpha\beta}^{(\mu)}$ 和第 5 节的
mapping residual。
任何进一步分解（`J_iso`, `K`, `D`, `Gamma`, local fields, constants）均不属于
当前 runtime output contract。

## 4) 规范/基约定（MUST）
- 耦合参数依赖局域 Kramers 规范选择。
- 为可复现与可比对，映射输出必须记录 gauge/basis id。
- 若投影前做了局域基旋转，必须显式记录该旋转。

## 5) 校验（MUST）
对每个 $\mu$：
- 输入 $\mathrm{Heff}^{(\mu)}$ 必须满足厄米性。
- 由导出的 $J$ 重构
  $\widetilde H^{(\mu)}$，并检查

Math:
$$
r_\mu
=
\frac{
\left\|\widetilde H^{(\mu)}-\mathrm{Heff}^{(\mu)}\right\|_F
}{
\left\|\mathrm{Heff}^{(\mu)}\right\|_F
}
$$

- 标量常数项 $C_{00}^{(\mu)} I\otimes I$ 不导出，只可保留用于 residual check。
- Local-field terms $C_{0\alpha}^{(\mu)}$、$C_{\alpha0}^{(\mu)}$ 和其他
  non-exchange leakage 不导出为 exchange。它们必须通过 failed
  `mapping_residual` check 保持可见；实现不得把它们折进
  $J_{\alpha\beta}^{(\mu)}$。
- 要求 `mapping_residual <= eps_map`。这表示 projected Hamiltonian 除 scalar
  shift 外是 exchange-only。
- 导出实耦合中的虚部泄漏必须低于容差。

## 6) 运行时 I/O（汇总）
Code form:
```text
inputs_33  = {Heff_mu_abcd, labels_abcd, kramer_basis_id}
outputs_33 = {J_mu[3,3], mapping_residual}
```
