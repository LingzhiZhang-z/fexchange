# 01-05-STEVENS_TENSOR_CONVERSION

本文件定义 Stevens 算符与球张量算符之间的转换契约。

## 1) 范围（MUST）
必须在 `k <= 6` 范围内提供以下双向映射：
- Stevens tesseral 算符 \(O_k^{q,c}\)、\(O_k^{q,s}\)
- 球张量算符 \(T_k^q\)

## 2) 转换公式（MUST）
当 `q=0`：

Math:
$$
O_k^0 = \alpha_{k0}(J)\, T_k^0.
$$

当 `q>0`：

Math:
$$
O_k^{q,c} = \frac{\alpha_{kq}(J)}{\sqrt2}\left[(-1)^q T_k^q + T_k^{-q}\right],
$$

Math:
$$
O_k^{q,s} = \frac{\alpha_{kq}(J)}{i\sqrt2}\left[(-1)^q T_k^q - T_k^{-q}\right].
$$

说明：在 `01-04` 的约化矩阵元为 1 约定下，转换系数通常依赖 `J`，必须数值计算。

## 3) 系数数值策略（MUST）
必须通过矩阵比较在运行时求系数：

Code form:
```text
alpha = <B, O> / <B, B>
```
其中：
- `O` 为 Stevens 矩阵；
- `B` 为对应的张量基组合；
- `<A,B> = Tr(A† B)`。

文献静态表可选，运行时计算为规范要求。

## 4) 逆映射（MUST）
必须在同一系数约定下，提供从张量分量回到 Stevens tesseral 通道的逆映射。

## 5) API 契约（MUST）

Code form:
```text
stevens_to_tensor_coefficient(J, k, q, mode='cos') -> complex
convert_stevens_to_tensors(J, k, q, mode='cos') -> dict[int, complex]
convert_tensor_to_stevens(J, k, q, mode='cos') -> dict[str, complex]
```

## 6) 验证（MUST）
必须在整数与半整数流形上做 round-trip 数值验证。
推荐覆盖：`J = 2, 5/2, 4, 7/2`。

验证判据：

Math:
$$
\|O - O_{\text{reconstructed}}\|_F < \varepsilon,
\qquad
\|T - T_{\text{reconstructed}}\|_F < \varepsilon.
$$
