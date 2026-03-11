# 00-00-SPEC_WRITING_CONVENTION

This file defines AI-readable writing conventions for all standards files.

## 1) Scope (MUST)
- This convention applies to all files under `./standards/en/` and `./standards/zh/`.
- Physics/model content stays in domain standards defined in later sections; this file only defines writing format.

## 2) Per-Rule Structure (MUST)
Each nontrivial rule should be written in the following order:
1. `MUST` bullets (hard constraints).
2. `Math:` block with LaTeX (`$$ ... $$`).
3. `Code form:` single-line ASCII expression.
4. `Index:` explicit symbol/index meaning.
5. `Validation:` minimal checks.

## 3) Formula Style (MUST)
- Prefer display math blocks (`$$ ... $$`) over long inline math.
- For long formulas, split into multiple display blocks or use aligned layouts.
- Do not rely on implicit symbols from distant sections; redefine key symbols locally.

## 4) Index Convention (MUST)
- Index names must be globally consistent across files.
- If a section introduces local index rules, it must state them before formulas.
- If an index is reused with different meaning, rename it instead of overloading.

## 5) Code Mapping (MUST)
- Every core formula must include one `Code form` line for direct implementation mapping.
- `Code form` should be parseable plain text and avoid notation-only shorthand.

## 6) Cross-File References
- Cross-file references should use relative paths under `./standards/en/...`.
- Domain files may add one-line references to this convention file instead of repeating formatting policy.
