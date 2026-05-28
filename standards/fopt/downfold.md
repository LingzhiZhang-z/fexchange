# 二阶 downfold：从 p 中间态到 f2 -> f1 hopping

本文只推导 directed process

$$
f_1^n f_2^n p^6
\rightarrow
f_1^{n+1} f_2^{n-1} p^6,
$$

也就是一个 electron 从 $f_2$ 有效跳到 $f_1$。

全局 Fock block 顺序固定为

$$
f_1<f_2<p.
$$

这个顺序只用于确定 fermion operator 的嵌入。downfold 完成后，
投影回 $p^6$，剩下的有效算符只作用在 $f_1\oplus f_2$ 上。

---

## 0. hopping operator

p-to-f hopping 定义为

$$
A_i
=
\sum_{\gamma\mu}
t_i^{\gamma\mu}
f_{i\gamma}^{\dagger}p_\mu,
\qquad i=1,2.
$$

反向 hopping 是

$$
B_i=A_i^\dagger
=
\sum_{\gamma\mu}
\left(t_i^{\gamma\mu}\right)^*
p_\mu^\dagger f_{i\gamma}.
$$

其中 $\gamma=1,\ldots,14$ 是 f spin-orbital index，
$\mu=1,\ldots,6$ 是 p spin-orbital index。

Code form:

```text
A_i = sum(gamma, mu) t_i[gamma, mu] * f_i_dag[gamma] * p[mu]
B_i = A_i.dagger()
```

---

## 1. charge path

初态是

$$
|I\rangle
=
|c\rangle_{f_1^n}
\otimes
|d\rangle_{f_2^n}
\otimes
|\Omega\rangle_{p^6}.
$$

末态是

$$
|F\rangle
=
|a\rangle_{f_1^{n+1}}
\otimes
|b\rangle_{f_2^{n-1}}
\otimes
|\Omega\rangle_{p^6}.
$$

因为初态 p shell 已满，第一步不能是 $f_2\to p$，否则会产生
$p^7$。非零路径必须先产生一个 p hole：

$$
f_1^n f_2^n p^6
\xrightarrow{A_1}
f_1^{n+1} f_2^n p^5
\xrightarrow{B_2}
f_1^{n+1} f_2^{n-1} p^6.
$$

Code form:

```text
path = B_2 * R_p5 * A_1
```

---

## 2. 四个费米子算符

令 $|\eta\rangle$ 表示 $p^5$ 中间态。directed 二阶项是

$$
H_{12}^{(2)}
=
P_{p^6}\,B_2\,R_{p^5}\,A_1\,P_{p^6},
$$

其中

$$
R_{p^5}
=
\sum_\eta
\frac{|\eta\rangle\langle\eta|}{E_0-E_\eta}.
$$

代入 $A_1$ 和 $B_2$ 后，核心算符串就是

$$
p_\nu^\dagger f_{2\beta}\,
|\eta\rangle\langle\eta|\,
f_{1\alpha}^\dagger p_\mu.
$$

也就是说，二阶 downfold 只是在这个四算符串中把 p 部分折叠掉。

Code form:

```text
operator_string = p_dag[nu] * f2[beta] * |eta><eta| * f1_dag[alpha] * p[mu]
```

---

## 3. 折叠 p 算符

p 部分给出一个只含 p index 的传播子：

$$
C_{\nu\mu}
=
\sum_\eta
\frac{
\langle\Omega|p_\nu^\dagger|\eta\rangle
\langle\eta|p_\mu|\Omega\rangle
}{
E_0-E_\eta
}.
$$

于是四算符串变成

$$
\sum_{\alpha\beta\mu\nu}
\left(t_2^{\beta\nu}\right)^*
t_1^{\alpha\mu}
C_{\nu\mu}
f_{2\beta}f_{1\alpha}^\dagger.
$$

这一步就是 downfold 的核心：$p_\nu^\dagger|\eta\rangle\langle\eta|p_\mu$
被替换成 $C_{\nu\mu}$，剩下的只是一对 f operator。

Code form:

```text
C_signed[nu, mu] = sum_eta <Omega|p_dag[nu]|eta> * <eta|p[mu]|Omega> / (E0 - E_eta)
```

---

## 4. 匹配到 f-only hopping

有效 f-only hopping 写成

$$
H_{12}^{(2)}
=
\sum_{\alpha\beta}
\tau_{12}^{\alpha\beta}
f_{1\alpha}^\dagger f_{2\beta}.
$$

上一节得到的是 $f_{2\beta}f_{1\alpha}^\dagger$。因为两个不同 f site
的 fermion operator 反交换，

$$
f_{2\beta}f_{1\alpha}^\dagger
=
-
f_{1\alpha}^\dagger f_{2\beta}.
$$

同时 $C_{\nu\mu}$ 用的是 signed denominator $E_0-E_\eta$。若定义
正的 p propagator

$$
G_{\mu\nu}
=
-
C_{\nu\mu}
=
\sum_\eta
\frac{
\langle\eta|p_\mu|\Omega\rangle
\langle\Omega|p_\nu^\dagger|\eta\rangle
}{
E_\eta-E_0
},
$$

则反交换带来的负号和 denominator 的负号已经合并进 $G$，最终

$$
\tau_{12}^{\alpha\beta}
=
\sum_{\mu\nu}
t_1^{\alpha\mu}
G_{\mu\nu}
\left(t_2^{\beta\nu}\right)^*.
$$

这就是矩阵形式

$$
\tau_{12}=T_1GT_2^\dagger.
$$

因此不要再额外加一个人为的 fermion sign。只要 $A_1$、$B_2$ 和
$f_1^\dagger f_2$ 都按同一个全局 Fock 顺序构造，符号已经由上面的
operator matching 固定。

Code form:

```text
G_pos[mu, nu] = -C_signed[nu, mu]
tau12 = t_f1_p @ G_pos @ t_f2_p.conj().T
```

---

## 5. p5 basis 下的 G 传播子

直接定义 hole basis：

$$
|\mu\rangle
\equiv
p_\mu|\Omega\rangle.
$$

其中 $|\Omega\rangle$ 是 $p^6$ closed shell。于是

$$
|p^5,\eta\rangle
=
\sum_\mu
Q_{\mu\eta}|\mu\rangle.
$$

定义 overlap matrix

$$
Q_{\mu\eta}
=
\langle\mu|\eta\rangle.
$$

计算 $Q$ 不需要重新构造完整的 $p^5$ determinant Hamiltonian。因为
hole basis 已经定义为 $|\mu\rangle=p_\mu|\Omega\rangle$，所以只要把
电子的一体 SOC 矩阵换成 hole 形式。

令电子 p-shell SOC 矩阵为

$$
h^{\mathrm{soc}}_{\mu\nu}
=
\langle\mu|h_{\mathrm{SOC}}|\nu\rangle.
$$

在 hole basis 中，对应的 SOC Hamiltonian 是

$$
H^{(5)}_{\mathrm{hole}}
=
-\left(h^{\mathrm{soc}}\right)^T.
$$

这里的负号来自“少一个电子”，转置来自 hole basis
$|\mu\rangle=p_\mu|\Omega\rangle$ 的指标顺序。然后直接对角化

$$
H^{(5)}_{\mathrm{hole}}Q
=
QC,
\qquad
C_{\eta\eta'}
=
c_\eta^{(5)}\delta_{\eta\eta'}.
$$

NSOC 时可以直接取

$$
Q_{\mu\eta}
=
\delta_{\mu\eta}.
$$

SOC 时，$Q$ 是上面这个 $6\times6$ Hamiltonian 的本征矢矩阵。在
hole basis

$$
\left(
|-1,\downarrow\rangle,\ |-1,\uparrow\rangle,\
|0,\downarrow\rangle,\ |0,\uparrow\rangle,\
|+1,\downarrow\rangle,\ |+1,\uparrow\rangle
\right)
$$

中，可以选取下面这个解析规范：

$$
c_\eta^{(5)}
=
\left(
-\frac12,\ -\frac12,\ -\frac12,\ -\frac12,\ 1,\ 1
\right),
$$

$$
Q
=
\begin{pmatrix}
1&0&0&0&0&0\\
0&\sqrt{1/3}&0&0&\sqrt{2/3}&0\\
0&\sqrt{2/3}&0&0&-\sqrt{1/3}&0\\
0&0&\sqrt{2/3}&0&0&-\sqrt{1/3}\\
0&0&\sqrt{1/3}&0&0&\sqrt{2/3}\\
0&0&0&1&0&0
\end{pmatrix}.
$$

前四列对应 $c_\eta^{(5)}=-1/2$，后两列对应
$c_\eta^{(5)}=1$。由于这两个子空间内部有简并，$Q$ 在各自简并
子空间内还可以再做酉变换；上式只是一个固定的 Clebsch-Gordan
规范选择。

这个表示下

$$
\langle p^5,\eta|p_\mu|\Omega\rangle
=
\langle\eta|\mu\rangle,
$$

并且

$$
\langle\eta|\mu\rangle
=
Q_{\mu\eta}^*.
$$

在 $p^5$ eigenbasis 下，正能量分母的传播子是

$$
G^{(5,+)}_{\eta\eta'}
=
\frac{\delta_{\eta\eta'}}{E_{p^5,\eta}},
\qquad
E_{p^5,\eta}
=
\Delta+\lambda_p c_\eta^{(5)}.
$$

代码 L3 里的 signed resolvent 是

$$
R^{(5)}_{\eta\eta'}
=
-
\frac{\delta_{\eta\eta'}}{E_{p^5,\eta}}.
$$

如果把传播子写回 hole-orbital index，则

$$
G^p_{\mu\nu}
=
\sum_\eta
\frac{
\langle\eta|\mu\rangle
\langle\nu|\eta\rangle
}{
E_{p^5,\eta}
}
=
\sum_\eta
\frac{
Q_{\mu\eta}^*
Q_{\nu\eta}
}{
E_{p^5,\eta}
}.
$$

Code form:

```text
ket_mu = p[mu] |Omega>
bra_eta_p_mu_Omega = <eta|mu>
Q[mu, eta] = <mu|eta>
NSOC: Q = I6
SOC:  c_eta, Q = eigh(-h_soc.T)
SOC analytic c_eta = [-1/2, -1/2, -1/2, -1/2, 1, 1]
G5_pos[eta, eta_prime] = delta(eta, eta_prime) / E_p5[eta]
R5_code[eta, eta_prime] = -delta(eta, eta_prime) / E_p5[eta]
G_p[mu, nu] = sum_eta conj(Q[mu, eta]) * Q[nu, eta] / E_p5[eta]
```
