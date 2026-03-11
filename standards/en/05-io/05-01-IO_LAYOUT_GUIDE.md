# 05-01-IO_LAYOUT_GUIDE

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
          input_hopping.npz   (optional snapshot)
          U-{U}_Jh-{Jh}_z-{zeta}/
              L3/
              kramer/
                {kramer_name}/
                  input_kramer.npz (optional snapshot)
                  L4/
                  spin12/           (optional, module 04-03)
```

Note:
- `LMSM` is a path alias of module 03-00 LSMS output directory.
- `level="L0"` maps to disk subtree `./outputs/fock/`.

## 3) Concrete Example (`n=6`)
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
