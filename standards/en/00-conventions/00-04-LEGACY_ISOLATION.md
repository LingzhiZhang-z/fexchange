# 00-04-LEGACY_ISOLATION

This file defines mandatory isolation rules for legacy standards/artifacts.
Goal: prevent accidental reads of frozen `*_LEGACY` content.

## 1) Scope (MUST)
MUST:
- Apply to all standard readers, validators, and runtime loaders.
- Apply to both files and directories marked as legacy.

Code form:
```text
legacy_isolation_applies = true
```

Validation:
- Any default path resolver must enforce this file.

## 2) Legacy Identification (MUST)
MUST:
- A path is legacy if one condition is true:
  - basename contains `_LEGACY`
  - basename ends with `.legacy`
  - path is inside `legacy/` subtree
- Legacy identification is purely lexical and deterministic.

Code form:
```text
is_legacy(path) = contains("_LEGACY") or endswith(".legacy") or in_subtree("legacy/")
```

Validation:
- Path tagged as legacy by rule must never be treated as normative by default.

## 3) Default Read Policy (MUST)
MUST:
- Default mode forbids reading legacy specs/artifacts.
- Discovery lists must hide legacy entries by default.
- `README`/required-reading flows must include normative files only.

Code form:
```text
allow_legacy = false
if is_legacy(path): raise LegacyAccessError(FXE-LEGACY-001/002)
```

Validation:
- Attempted legacy read in default mode is a hard failure.

## 4) Explicit Override Policy (MUST)
MUST:
- Legacy access is allowed only with explicit override flags:
  - `allow_legacy = true`
  - `legacy_scope` (exact allowlist paths)
  - `legacy_reason` (human-readable reason)
  - `legacy_ticket` (traceability id)
- Override is session-scoped and must not persist silently.

Code form:
```text
require allow_legacy and legacy_scope and legacy_reason and legacy_ticket
```

Validation:
- Missing any override field keeps legacy access forbidden.

## 5) Output Contamination Guard (MUST)
MUST:
- If legacy inputs are used, outputs must be marked:
  - `legacy_mode=true`
  - `legacy_inputs=[...]`
- Legacy-mode outputs must not be promoted to normal cache/index channels.

Code form:
```text
if legacy_mode:
  meta.legacy_mode = true
  block_publish_to_normative_index()
```

Validation:
- Publishing legacy-derived outputs as normative is forbidden.

## 6) Agent Reading Gate (MUST)
MUST:
- AI/agent readers must read only normative paths unless explicit legacy override is present.
- On conflict between normative and legacy text, normative wins automatically.

Code form:
```text
reader_paths = filter_not_legacy(all_paths) unless allow_legacy
```

Validation:
- Any prompt context containing legacy text without override is a gate failure.

## 7) Failure Codes (MUST)
MUST:
- Use fixed codes from `./standards/en/00-conventions/00-03-ERROR_CODES_AND_FAILURE_PAYLOAD.md`:
  - `FXE-LEGACY-001`: legacy spec read blocked.
  - `FXE-LEGACY-002`: legacy artifact load blocked.

Validation:
- Legacy failures without these codes are invalid error reports.
