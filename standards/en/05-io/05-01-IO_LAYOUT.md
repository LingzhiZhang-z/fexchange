# 05-01-IO_LAYOUT

This file is **non-normative**.
It provides recommended directory layouts, examples, and optional engineering strategies.
Normative requirements are in `./standards/en/05-io/05-00-IO.md`.

## 1) Purpose
- Help implementers quickly build compliant directory trees.
- Show concrete path examples for typical runs.
- Provide optional storage/performance recipes.

## 2) Recommended Directory Tree (Example)
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

  <run_name>/
    source.txt
    IONED/
      n-{n-1}/states.npz
      n-{n+1}/states.npz
    L1/
      F/data.npz
    L2/data.npz
    L3/data.npz
```

Note:
- `LMSM` is a path alias of module 03-00 LSMS output directory.
- `level="L0"` maps to disk subtree `./outputs/core/L0/`.

## 3) Concrete Example (`n=6`)
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

## 4) Optional Snapshot Files (Audit-Only)
You may store:
- `hopping/{hopping_name}/input_hopping.npz`
- `kramer/{kramer_name}/input_kramer.npz`

These are only for reproducibility/audit and consistency checks.
They are not computational source inputs.

## 5) Optional Storage Recipes
These are recommendations only:
- Dense payload:
  - keys: `value`, `shape`
- Sparse COO payload:
  - keys: `indices`, `value_re`, `value_im`, `shape`
- Factorized payload:
  - keys: named factors + exact reconstruction note in `meta.json`

Suggested defaults:
Code form:
```text
write_npz = np.savez_compressed
default_dtype_real = float64
default_dtype_complex = complex128
```

## 6) Suggested Index Record Fields
Recommended line fields for `./outputs/index.jsonl`:
- `key,module,level,path,created_at`
- `n,r42,r62,U,Jh,zeta`
- `hopping_name,kramer_name`
- `content_hash`

## 7) Conflict Handling
If any example in this file conflicts with `./standards/en/05-io/05-00-IO.md`,
`./standards/en/05-io/05-00-IO.md` is authoritative.
