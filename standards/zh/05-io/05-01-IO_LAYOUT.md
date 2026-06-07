# 05-01-IO_LAYOUT

本文件是**非规范性**说明。
它提供推荐目录布局、示例路径和可选工程策略。
规范性约束在 `./standards/en/05-io/05-00-IO.md`。

## 1) 用途
- 帮助实现者快速搭建符合契约的目录树。
- 给出典型运行的具体路径样例。
- 提供可选的存储/性能实践建议。

## 2) 推荐目录树（示例）
Code form:
```text
./outputs/
  core/
    L0/
      f_create_{n}_to_{n+1}.npz
      f_create_{n-1}_to_{n}.npz
      p_create_4_to_5.npz
      p_create_5_to_6.npz
    LMSM/
      n-{n}_r42-{r42:.8f}_r62-{r62:.8f}_scheme-{scheme}/
    LSJM/
      n-{n}_r42-{r42:.8f}_r62-{r62:.8f}_scheme-{scheme}/
    ligand/
      soc/n-{N}/
      nsoc/n-{N}/
    L1/
      F/n_ele-{n}_r42-{r42:.8f}_r62-{r62:.8f}_scheme-{scheme}/
      P/n-4_to_5_{soc|nsoc}/
      P/n-5_to_6_{soc|nsoc}/

  <output_run>/<run_name>/  # output_run 是可选 base override
    source.txt
    IONED/
      n-{n-1}/states.npz
      n-{n+1}/states.npz
    L1/
      F/data.npz
    L2/data.npz
    L3/data.npz
```

说明：
- `LMSM` 是模块 03-00 LSMS 输出目录的路径别名。
- `level="L0"` 对应磁盘子树 `./outputs/core/L0/`。

## 3) 具体示例（`n=6`）
Code form:
```text
./outputs/core/L0/f_create_6_to_7.npz
./outputs/core/L0/f_create_5_to_6.npz
./outputs/core/LMSM/n-6_r42-1.00000000_r62-2.00000000_scheme-RS/LMSM.npz
./outputs/core/LSJM/n-6_r42-1.00000000_r62-2.00000000_scheme-RS/LSJM.npz
./outputs/core/L1/F/n_ele-6_r42-1.00000000_r62-2.00000000_scheme-RS/data.npz
./outputs/demo_run/IONED/n-5/states.npz
./outputs/demo_run/IONED/n-7/states.npz
./outputs/demo_run/L1/F/data.npz
./outputs/demo_run/L2/data.npz
./outputs/demo_run/L3/data.npz
```

## 4) 可选快照文件（仅审计）
可选保存：
- `hopping/{hopping_name}/input_hopping.npz`
- `kramer/input_kramer.npz`

这些文件仅用于复现/审计和一致性检查。
它们不是计算源输入。

## 5) 可选存储策略
以下只是建议：
- Dense 载荷：
  - keys: `value`, `shape`
- Sparse COO 载荷：
  - keys: `indices`, `value_re`, `value_im`, `shape`
- Factorized 载荷：
  - keys: named factors + `meta.json` 中的精确重构说明

建议默认：
Code form:
```text
write_npz = np.savez_compressed
default_dtype_real = float64
default_dtype_complex = complex128
```

## 6) 建议索引字段
推荐 `./outputs/index.jsonl` 每行字段：
- `key,module,level,path,created_at`
- `n,r42,r62,U,Jh,zeta`
- `hopping_name,kramer_file`
- `content_hash`

## 7) 冲突处理
若本文件样例与 `./standards/en/05-io/05-00-IO.md` 冲突，
以 `./standards/en/05-io/05-00-IO.md` 为准。
