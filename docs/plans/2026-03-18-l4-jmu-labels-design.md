# L4 J_mu And Label Defaults Design

## Status

Approved design for:
- exposing user-facing `hopping_label` / `projection_label` while normalizing to internal canonical tokens,
- supporting `RE`-based default `F2/F4/F6/zeta` values with explicit input override,
- exporting `J_mu` from `L4` when the projected local space is `2 x 2`,
- writing one shallow single-point text result under `./outputs/`.

## Decisions

1. Level system stays unchanged.
   - Keep `LMSM/LSJM/L0/L1/L2/L3/L4`.
   - Do not add a new `spin12` runtime level.

2. `J_mu` belongs to `L4`.
   - `L4/data.npz` always contains `Heff_mu_abcd` and `h_mu_abcd`.
   - If the projected local space is `2 x 2`, `L4/data.npz` additionally contains `J_mu` and `mapping_residual`.
   - If not `2 x 2`, `L4` still succeeds and simply omits `J_mu`.
   - Do not export `J_iso/K/D/Gamma/const/h_i/h_j`.

3. User-facing labels are normalized once at input load.
   - Input accepts `hopping_label` and `projection_label`.
   - Loader maps them to internal canonical fields `hopping_name` and `kramer_name`.
   - Core runtime, keys, paths, metadata, and cache indexing use only the canonical names.

4. `RE` is a default-parameter template name, not a physics consistency gate.
   - `RE="auto"` means no template defaults.
   - Otherwise `RE` loads default `F2/F4/F6/zeta` from the SQPerturbation table.
   - Explicit `physics.F2/F4/F6` and `sopt.zeta` override template values field-by-field.
   - `U/Jh/zeta` input remains single-point only.

5. Shallow result export is per single point.
   - Write one text file under `./outputs/`:
     `{RE}_{n_ele}_{hopping_label}_{projection_label}_{U}_{Jh}_{zeta}.txt`
   - Use fixed `%.6f` formatting for `U/Jh/zeta` in the filename.
   - File content is one line with columns:
     `U Jh Jh/U zeta Jxx Jxy Jxz Jyx Jyy Jyz Jzx Jzy Jzz error`
   - `error` is `mapping_residual`.

## Implementation Notes

- The current codebase still uses `[inputs]` for file-based hopping/projector sources.
- Existing standards already reserve canonical cache tokens via `hopping_name` / `kramer_name`.
- The implementation should therefore:
  - extend the loader instead of introducing a second internal naming path,
  - fix `disk.py`, `keys.py`, and pipeline validation/path usage to include canonical labels,
  - keep the CLI and runtime level orchestration minimal.

## Verification Scope

- Targeted unit tests for run-input normalization and L4 J_mu export.
- One or a few single-point validation runs for `YbOCl` and `YbOBr`.
