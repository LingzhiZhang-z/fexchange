# L4 J_mu And Label Defaults Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Add canonical label normalization, `RE` default parameters, optional `J_mu` export inside `L4`, and shallow single-point txt export without changing the existing runtime level system.

**Architecture:** Keep the runtime pipeline structure unchanged and integrate the new behavior at existing boundaries. Normalize user input in `run_input.py`, propagate canonical names through keys and paths, compute `J_mu` as optional `L4` post-processing, and emit a shallow txt summary after a successful single-point run.

**Tech Stack:** Python 3, NumPy, pytest, existing `fexchange` CLI/pipeline/artifact stack.

---

### Task 1: Document and pin the changed runtime contract

**Files:**
- Create: `docs/plans/2026-03-18-l4-jmu-labels-design.md`
- Create: `docs/plans/2026-03-18-l4-jmu-labels-plan-gpt-5.md`

**Step 1: Save the approved design**

Write the approved decisions for:
- `J_mu` inside `L4`
- `hopping_label/projection_label -> hopping_name/kramer_name`
- `RE` defaults with explicit override
- shallow per-point txt export

**Step 2: Save the implementation plan**

Write a minimal, task-oriented plan tied to exact files and tests.

### Task 2: Update input normalization and canonical naming

**Files:**
- Modify: `fexchange/io/run_input.py`
- Test: `tests/test_run_input.py`

**Step 1: Write/adjust failing tests**

Cover:
- accepting `sources` instead of rejecting it,
- accepting `hopping_label` / `projection_label`,
- normalizing them to canonical names,
- applying `RE` defaults,
- allowing explicit `F2/F4/F6/zeta` overrides.

**Step 2: Run targeted tests to confirm failure**

Run:
```bash
pytest tests/test_run_input.py -q
```

**Step 3: Implement loader normalization**

Add canonical-field normalization in `load_run_input()` so downstream code sees one internal naming scheme only.

**Step 4: Re-run targeted tests**

Run:
```bash
pytest tests/test_run_input.py -q
```

### Task 3: Fix cache keys and disk paths to include canonical labels

**Files:**
- Modify: `fexchange/io/disk.py`
- Modify: `fexchange/pipeline/keys.py`
- Modify: `fexchange/pipeline/validation.py`
- Modify: `fexchange/pipeline/artifacts.py`
- Modify: `fexchange/pipeline/stages.py`

**Step 1: Add failing tests or extend existing ones where practical**

Cover:
- `L2` path includes hopping token,
- `L4` path includes hopping and kramer tokens,
- keys include canonical names.

**Step 2: Implement path/key propagation**

Use normalized canonical names only:
- `hopping_name`
- `kramer_name`

**Step 3: Re-run focused tests**

Run:
```bash
pytest tests/test_run_input.py tests/test_sopt_l2_l4.py -q
```

### Task 4: Shrink spin-1/2 mapping output and integrate it into L4

**Files:**
- Modify: `fexchange/sopt/spin12.py`
- Modify: `fexchange/pipeline/artifacts.py`
- Modify: `fexchange/pipeline/stages.py`
- Test: `tests/test_spin12.py`
- Test: `tests/test_sopt_l2_l4.py`

**Step 1: Write/adjust failing tests**

Cover:
- `spin12_map()` returns only `J_mu` and `mapping_residual`,
- `L4/data.npz` writes those fields only when `n_k == 2`,
- non-`2 x 2` projection keeps `L4` valid without `J_mu`.

**Step 2: Run targeted failing tests**

Run:
```bash
pytest tests/test_spin12.py tests/test_sopt_l2_l4.py -q
```

**Step 3: Implement minimal mapping integration**

Reuse the existing mapping logic but remove unused derived outputs.

**Step 4: Re-run targeted tests**

Run:
```bash
pytest tests/test_spin12.py tests/test_sopt_l2_l4.py -q
```

### Task 5: Emit the shallow single-point txt result

**Files:**
- Modify: `fexchange/cli.py`
- Modify: `fexchange/io/disk.py` or `fexchange/pipeline/artifacts.py` (wherever the result writer fits best)
- Add/Modify tests near the chosen seam

**Step 1: Write/adjust failing tests**

Cover:
- per-point txt filename format,
- one-line column order,
- omission when `J_mu` is unavailable.

**Step 2: Run focused tests to confirm failure**

Run the relevant pytest subset.

**Step 3: Implement txt export**

Use fixed `%.6f` formatting in filenames and write one line:
`U Jh Jh/U zeta Jxx Jxy Jxz Jyx Jyy Jyz Jzx Jzy Jzz error`

**Step 4: Re-run focused tests**

Run the relevant pytest subset.

### Task 6: Run targeted verification and requested single-point checks

**Files:**
- Reuse existing validation inputs and generated outputs only

**Step 1: Run the unit-test subset**

Run:
```bash
pytest tests/test_run_input.py tests/test_spin12.py tests/test_sopt_l2_l4.py -q
```

**Step 2: Run requested single-point validations**

Pick a few points from:
- `data/test/YbOCl`
- `data/test/YbOBr`

Verify:
- pipeline success,
- `L4/data.npz` contains `J_mu` when expected,
- shallow txt file is emitted.

**Step 3: Summarize residual risk**

Call out any remaining mismatch between current code and standards that was intentionally left out of scope.
