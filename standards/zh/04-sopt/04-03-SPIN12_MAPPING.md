# 04-03-SPIN12_MAPPING

本文件定义从 `04-02-RUNTIME_CONTRACTION` 输出到赝自旋-$\tfrac{1}{2}$模型的后处理映射。
模块读取 $\mathrm{Heff}_{cd,ab}^{(\mu)}$ 并投影为自旋耦合参数。
磁盘 I/O 布局与格式由 `./standards/en/05-io/05-00-IO.md` 统一定义。
写作形式遵循 `./standards/en/00-conventions/00-00-SPEC_WRITING_CONVENTION.md`。

## 0) 适用范围（MUST）
- 输入来自 `./standards/en/04-sopt/04-02-RUNTIME_CONTRACTION.md` 的 $L4$ 输出。
- 本模块是后处理映射，不改变 `L0..L4` 的定义。
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

### 3.2 派生分解（MUST，由 $J$ 计算）
除原始 $J$ 外，04-03 模块必须同时输出以下由原始 $J$ 计算得到的派生量：
- 各向同性项：
  $J_{\mathrm{iso}}=\frac{1}{3}(J_x+J_y+J_z)$，
- 默认键向 Kitaev 参数：
  默认按 $z$-bond，定义
  $K^{(z\text{-bond})}=J_z-J_{\mathrm{iso}}$，
- DM 向量：
  $D_x=\frac{1}{2}(J_{yz}-J_{zy})$，
  $D_y=\frac{1}{2}(J_{zx}-J_{xz})$，
  $D_z=\frac{1}{2}(J_{xy}-J_{yx})$，
- 对称各向异性：
  $\Gamma=\frac{1}{2}(J+J^\mathsf{T})-J_{\mathrm{diag}}$，其中
  $J_{\mathrm{diag}}=\mathrm{diag}(J_x,J_y,J_z)$。

说明：
- 本模块中的 `K_mu` 表示 Kitaev 型交换参数，不是 $L2$ 的核张量 `K`。

### 3.3 非交换项（MUST 定义，模型拟合可选）
恒等项和单站点项不属于交换主模型
$\sum_{\alpha\beta}J_{\alpha\beta}S_i^\alpha S_j^\beta$。
其定义为

Math:
$$
\mathrm{const}^{(\mu)} = C_{00}^{(\mu)},\quad
h_{i,\alpha}^{(\mu)} = 2C_{\alpha 0}^{(\mu)},\quad
h_{j,\alpha}^{(\mu)} = 2C_{0\alpha}^{(\mu)}.
$$

## 4) 规范/基约定（MUST）
- 耦合参数依赖局域 Kramers 规范选择。
- 为可复现与可比对，映射输出必须记录 gauge/basis id。
- 若投影前做了局域基旋转，必须显式记录该旋转。

## 5) 校验（MUST）
对每个 $\mu$：
- 输入 $\mathrm{Heff}^{(\mu)}$ 必须满足厄米性。
- 由输出 $(J,J_{\mathrm{iso}},K,\mathbf D,\Gamma,\mathrm{const},\mathbf h_i,\mathbf h_j)$ 重构
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
\le \varepsilon_{\mathrm{map}}.
$$

- 导出实耦合中的虚部泄漏必须低于容差。
- 若导出默认 $z$-bond 的 `K`，必须检查
  $K = J_z - J_{\mathrm{iso}}$。

## 6) 运行时 I/O（汇总）
Code form:
```text
inputs_33  = {Heff_mu_abcd, labels_abcd, kramer_basis_id}
outputs_33 = {J_mu[3,3], J_iso_mu, K_mu, D_mu[3], Gamma_mu[3,3], const_mu, h_i_mu[3], h_j_mu[3], residual_mu}
```
