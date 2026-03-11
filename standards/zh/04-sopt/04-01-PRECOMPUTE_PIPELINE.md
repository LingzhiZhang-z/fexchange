# 04-01-PRECOMPUTE_PIPELINE

本文件定义 SOPT 预计算中的 $L0$ 与 $L1$。
约定与 `./standards/en/04-sopt/04-00-SOPT_FORMALISM.md` 一致：f 体系取 $E_0=0$，分母处理下放到 `./standards/en/04-sopt/04-02-RUNTIME_CONTRACTION.md`。
磁盘 I/O 布局与格式由 `./standards/en/05-io/05-00-IO.md` 统一定义。
写作形式遵循 `./standards/en/00-conventions/00-00-SPEC_WRITING_CONVENTION.md`。
执行顺序由 `./standards/en/04-sopt/04-00-SOPT_FORMALISM.md` 定义。
命名规范统一继承 `./standards/en/04-sopt/04-00-SOPT_FORMALISM.md` 的第 0.1 节。

## 0) 变量分类（子模块级，MUST）
本文件覆盖 $L0/L1$，采用三类变量语义：
- 输入变量：来自外部接口或上游层输出。
- 中间变量：仅在当前层内部计算使用，不作为该层对外输出。
- 输出变量：该层对下游层提供的接口变量。

分层定义：
- $L0$：输入 `{}`；中间 `{sign/workspace}`；输出 `{X, Y}`。
- $L1$：输入 `{X, Y, U_np1, U_n_soc0, U_nm1}`；中间 `{workspace}`；输出 `{A, B}`。

约束：
- 运行时 hopping/Kramer 对象不属于本模块输入。
- 分母组装与路线求和由 `./standards/en/04-sopt/04-02-RUNTIME_CONTRACTION.md` 处理。

## 0.1) 与外部 Hopping/Kramer 输入的边界（MUST）
MUST:
- `L0/L1` 不得消费外部 hopping 或 Kramer 输入。
- hopping/Kramer 的 schema 仅在 `04-02` 模块执行。
- `L1` 输出必须携带下游运行时绑定所需的确定性轴元数据：
  `j` 轴 LSJM 子空间顺序 id，以及 `kappa/p/q` 轨道顺序身份。

Code form:
```text
L0_L1_inputs_exclude = {t_mu, W, kramer_labels, labels_abcd, E_u}
L1_meta_required = {j_order_id, orbital_order_id, vertex_axis_order_id}
```

Validation:
- 若 `L0/L1` 运行时收到 hopping/Kramer 载荷，在契约模式下必须忽略或拒绝。
- 缺少下游绑定元数据的 `L1` 工件视为无效。

## 1) Level 0: Fock 基原始跃迁元
MUST:
- 本层只在标准 Fock 基上构造跃迁元，不依赖外部态文件，且不含 site 标签 $i/j$。
- 本层必须固定 Fock 基排序与费米子符号规则（继承 `01-physics` 规范）。
- 本层统一采用“低 $\to$ 高”跃迁存储，仅保存 $f^\dagger$ 方向。

Math:
$$
X^{\kappa,n}_{\alpha\beta}
\equiv
\langle \alpha^{n+1} \rvert f_{\kappa}^{\dagger} \lvert \beta^{n} \rangle.
$$

Math:
$$
Y^{\kappa,n-1}_{\beta \gamma}
\equiv
\langle \beta^{n} \rvert f_{\kappa}^{\dagger} \lvert \gamma^{n-1} \rangle.
$$

Math:
$$
\langle \beta^{n} \rvert f_{\kappa} \lvert \alpha^{n+1} \rangle
=
\left(X^{\kappa,n}_{\alpha\beta}\right)^{\ast},
\qquad
\langle \gamma^{n-1} \rvert f_{\kappa} \lvert \beta^{n} \rangle
=
\left(Y^{\kappa,n-1}_{\beta \gamma}\right)^{\ast}.
$$

Code form:
```text
build X_n[kappa] and Y_nm1[kappa] with f_dag only
recover reverse direction by conjugation
```

Index:
- $\alpha,\beta,\gamma$：本层实际使用的 Fock 基态标签。
- $\kappa$：通用单站点局域轨道指标（与 site 无关）。

Validation:
- 产生/湮灭后的目标扇区必须正确（$n \to n+1$ 或 $n \to n-1$）。
- 费米子符号必须与 bit 排序规则一致。
- 任意 site 必须复用同一套 $X/Y$；不得在 $L0$ 引入 site 特异版本。
- 禁止把共轭反向矩阵作为独立缓存重复落盘。

Output（MUST）:
- 本层必须独立输出 $X^{\kappa,n}$ 与 $Y^{\kappa,n-1}$（或等价可恢复表示）。
- 本层输出是 $L1$ 的直接输入。

## 2) Level 1: 局域跃迁顶点（LSJM）
MUST:
- 本层只定义不带 site 标签的通用单站点顶点。
- 本层对中间态扇区（$f^{n+1}$ 与 $f^{n-1}$）做局域基变换投影到 LSJM 语义。
- 本层把 $f^n$ 腿从 Fock 基投影到 SOC 下最低能 LSJM 子空间（即 $f^n$ 的 LSJM 基态多重态）。
- 本层不引入 Kramers 标签 $a,b,c,d$。
- $U^{(m)}$ 定义为 $f^m$ 扇区 Fock 基到 LSJM 基的列变换矩阵，规范来源：`./standards/en/03-representations/03-01-REPRESENTATION_LSJM.md`。
- $U^{n,\mathrm{soc0}}$ 定义为 $f^n$ 扇区 Fock 基到 SOC 最低能 LSJM 子空间的列变换矩阵。
- 中间态指标只使用 $u,v,r,s$。

Math:
$$
A^{\kappa,n}_{u j}
=
\sum_{\alpha,\beta}
\left(U^{n+1}_{\alpha u}\right)^{\ast}
X^{\kappa,n}_{\alpha\beta}
\left(U^{n,\mathrm{soc0}}_{\beta j}\right),
$$

Math:
$$
B^{\kappa,n-1}_{j v}
=
\sum_{\beta,\gamma}
\left(U^{n,\mathrm{soc0}}_{\beta j}\right)^{\ast}
Y^{\kappa,n-1}_{\beta\gamma}
\left(U^{n-1}_{\gamma v}\right).
$$

Math:
$$
\langle u^{n+1} \rvert f_{\kappa}^{\dagger} \lvert j^{n,\mathrm{soc0}}\rangle
=
A^{\kappa,n}_{u j},
\qquad
\langle j^{n,\mathrm{soc0}} \rvert f_{\kappa} \lvert u^{n+1}\rangle
=
\left(A^{\kappa,n}_{u j}\right)^{\ast}.
$$

Math:
$$
\langle j^{n,\mathrm{soc0}} \rvert f_{\kappa}^{\dagger} \lvert v^{n-1}\rangle
=
B^{\kappa,n-1}_{j v},
\qquad
\langle v^{n-1} \rvert f_{\kappa} \lvert j^{n,\mathrm{soc0}}\rangle
=
\left(B^{\kappa,n-1}_{j v}\right)^{\ast}.
$$

Code form:
```text
build generic {A_kappa_n[u,j], B_kappa_nm1[j,v]} without site labels
rotate (n+1)/(n-1) legs with U_np1, U_nm1
project n-leg to SOC-low subspace with U_n_soc0
recover reverse direction by complex conjugation
```

Validation:
- 顶点张量维度必须与扇区维度一致。
- 算符方向（$\dagger$ / 非 $\dagger$）必须与指标语义一致。
- 本层必须使用 $U^{n,\mathrm{soc0}}$，且其列空间必须只覆盖 SOC 最低能 LSJM 子空间。
- 本层禁止出现 Kramers 指标 $a,b,c,d$ 与投影矩阵 $W$。

Output（MUST）:
- 本层必须独立输出顶点张量 $A^{\kappa,n}_{u j}$ 与 $B^{\kappa,n-1}_{j v}$。
- 本层输出是 04-02 模块中 $L2$ 的直接输入。
