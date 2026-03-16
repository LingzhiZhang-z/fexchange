# 01-06-ANGULAR_MOMENTUM

本文件定义表示层（`03-00/03-01`）使用的角动量算符约定。

## 1) 算符集合（MUST）
- 轨道角动量算符：$L_x,L_y,L_z,L_\pm$。
- 自旋角动量算符：$S_x,S_y,S_z,S_\pm$。
- 总角动量算符：$J_x,J_y,J_z,J_\pm$。

定义：
Math:
$$
L_\pm = L_x \pm iL_y,\qquad
S_\pm = S_x \pm iS_y,\qquad
J = L+S,\quad J_\pm=L_\pm+S_\pm,\quad J_z=L_z+S_z.
$$

## 2) f 壳层二次量子化形式（MUST）
取 $\ell=3$，$m=-3,\dots,3$，$\sigma=\pm \frac{1}{2}$：
Math:
$$
L_z=\sum_{m,\sigma} m\,c^\dagger_{m\sigma}c_{m\sigma},
\qquad
S_z=\sum_{m,\sigma}\sigma\,c^\dagger_{m\sigma}c_{m\sigma}.
$$

Math:
$$
L_+ = \sum_{m=-\ell}^{\ell-1}\sum_\sigma
\sqrt{\ell(\ell+1)-m(m+1)}\;
c^\dagger_{m+1,\sigma}c_{m,\sigma},
\qquad
L_- = L_+^\dagger.
$$

Math:
$$
S_+ = \sum_m c^\dagger_{m,+1/2}c_{m,-1/2},
\qquad
S_- = S_+^\dagger.
$$

## 3) LS 基中的升降作用（MUST）
Math:
$$
L_\pm\lvert L,M_L\rangle
=
\sqrt{L(L+1)-M_L(M_L\pm1)}
\lvert L,M_L\pm1\rangle,
$$

Math:
$$
S_\pm\lvert S,M_S\rangle
=
\sqrt{S(S+1)-M_S(M_S\pm1)}
\lvert S,M_S\pm1\rangle.
$$

因此：
Math:
$$
J_\pm\lvert J,M\rangle
=
\sqrt{J(J+1)-M(M\pm1)}
\lvert J,M\pm1\rangle.
$$

## 4) 对 03-00/03-01 的约束
- `./standards/en/03-spectrum/03-00-LSMS.md` 必须引用本文件的 $L_\pm,S_\pm,L_z,S_z$ 定义。
- `./standards/en/03-spectrum/03-01-LSJM.md` 必须引用本文件的 $J_\pm,J_z$ 一致性定义。
- 若实现修改上述算符定义，必须声明新的 scheme/version。

## 5) 实现契约（MUST）
实现顺序固定如下：
1. 在固定轨道顺序下构造 $L_x,L_y,L_z,S_x,S_y,S_z$ 的单体矩阵。
2. 按定义构造升降算符：
   $L_\pm=L_x\pm iL_y$，$S_\pm=S_x\pm iS_y$，$J_\pm=L_\pm+S_\pm$。
3. 通过二次量子化把单体矩阵提升为多体算符。
4. 按 `01-02-OPERATOR_IMPLEMENTATION` 的一体算符接口导出。

Code form:
```text
build one_body {Lx,Ly,Lz,Sx,Sy,Sz}
Lpm = Lx ± i*Ly
Spm = Sx ± i*Sy
Jpm = Lpm + Spm
lift to many-body via second quantization
```

Validation:
- $L_- = L_+^\dagger$，$S_- = S_+^\dagger$，$J_- = J_+^\dagger$。
- 对易关系在容差内满足角动量代数。
