# 05-01-IO_LAYOUT_GUIDE

本文件是**非规范性**说明。
用于提供推荐目录布局、示例路径与可选工程策略。
规范性约束以 `./standards/en/05-io/05-00-IO.md` 为准。

## 1) 用途
- 帮助实现者快速搭建符合契约的目录树。
- 给出典型运行参数下的具体路径样例。
- 提供可选的存储/性能实践建议。

## 2) 推荐目录树（示例）
Code form:
```text
./outputs/
  fock/
    meta_n{n-1}.json
    meta_n{n}.json
    meta_n{n+1}.json
    n{n}{n+1}.npz
    n{n-1}{n}.npz

  core/
    n-{n}_r42-{r42}_r62-{r62}_scheme-{scheme}/
      LMSM/
      LSJM/
      L1/
      hopping/
        {hopping_name}/
          L2/
          input_hopping.npz   (可选快照)
          U-{U}_Jh-{Jh}_z-{zeta}/
              L3/
              kramer/
                {kramer_name}/
                  input_kramer.npz (可选快照)
                  L4/
                  spin12/           (可选，模块 04-03)
```

说明：
- `LMSM` 是模块 03-00 LSMS 输出目录的路径别名。
- 元数据 `level="L0"` 对应磁盘子树 `./outputs/fock/`。

## 3) 具体示例（`n=6`）
Code form:
```text
./outputs/fock/meta_n5.json
./outputs/fock/meta_n6.json
./outputs/fock/meta_n7.json
./outputs/fock/n67.npz
./outputs/fock/n56.npz

./outputs/core/n-6_r42-1.000000000000_r62-2.000000000000_scheme-RS/LMSM/V.npz
./outputs/core/n-6_r42-1.000000000000_r62-2.000000000000_scheme-RS/LMSM/E_terms.npz
./outputs/core/n-6_r42-1.000000000000_r62-2.000000000000_scheme-RS/LSJM/V.npz
./outputs/core/n-6_r42-1.000000000000_r62-2.000000000000_scheme-RS/LSJM/E_terms.npz
./outputs/core/n-6_r42-1.000000000000_r62-2.000000000000_scheme-RS/hopping/wannier/L2/data.npz
./outputs/core/n-6_r42-1.000000000000_r62-2.000000000000_scheme-RS/hopping/wannier/U-3.000000000000_Jh-4.000000000000_z-5.000000000000/L3/data.npz
./outputs/core/.../hopping/wannier/U-3.000000000000_Jh-4.000000000000_z-5.000000000000/kramer/k1/L4/data.npz
```

## 4) 可选快照文件（仅审计）
可选保存：
- `hopping/{hopping_name}/input_hopping.npz`
- `kramer/{kramer_name}/input_kramer.npz`

这些文件仅用于复现/审计与一致性检查，
不能作为计算源输入。

## 5) 可选存储策略
以下为建议，不是强制：
- Dense 载荷：
  - 键：`value`, `shape`
- Sparse COO 载荷：
  - 键：`indices`, `value_re`, `value_im`, `shape`
- Factorized 载荷：
  - 键：若干因子数组 + `meta.json` 精确重构说明

建议默认：
Code form:
```text
write_npz = np.savez_compressed
default_dtype_real = float64
default_dtype_complex = complex128
```

## 6) 建议索引字段
`./outputs/index.jsonl` 推荐每行至少包含：
- `key,module,level,path,created_at`
- `n,r42,r62,U,Jh,zeta`
- `hopping_name,kramer_name`
- `content_hash`

## 7) 冲突处理
若本文件样例与 `./standards/en/05-io/05-00-IO.md` 冲突，
以 `./standards/en/05-io/05-00-IO.md` 为准。
