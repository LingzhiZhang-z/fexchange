# 00-03-ERROR_CODES_AND_FAILURE_PAYLOAD

This file defines fixed runtime error codes and mandatory failure payload fields.
It is normative for all modules.

## 1) Scope (MUST)
MUST:
- Apply to modules `03-00/03-01/04-00/04-01/04-02/04-03/05-00/05-02/05-03`.
- Every hard failure must emit one fixed error code.
- Raw exceptions without coded payload are forbidden at public boundaries.

Code form:
```text
if failure: raise CodedError(code, payload)
```

Validation:
- Any public-facing error without `code` is a contract violation.

## 2) Error-Code Format (MUST)
MUST:
- Error code format is:
  `FXE-{DOMAIN}-{NNN}`.
- `DOMAIN` is uppercase and fixed-width.
- `NNN` is a zero-padded decimal id.
- Codes are stable once published (no semantic reassignment).

Code form:
```text
code = f"FXE-{DOMAIN}-{id:03d}"
```

Validation:
- Unknown code format must be rejected by error serializers.

## 3) Domain Set (MUST)
MUST:
- Allowed domains:
  - `INPUT`: missing/invalid run input file content.
  - `SCHEMA`: schema/type/field validation failures.
  - `BIND`: axis/basis/order binding mismatches.
  - `IO`: file path/hash/read-write/atomic write failures.
  - `NUM`: numerical stability/tolerance/linear-algebra failures.
  - `PHYS`: physical-constraint violations.
  - `W90`: Wannier90 parsing/mapping failures.
  - `LEGACY`: forbidden legacy-spec/artifact access.
  - `RUNTIME`: uncategorized runtime failures after normalization.

Validation:
- Domain outside this set is invalid.

## 4) Mandatory Payload Schema (MUST)
MUST:
- Every failure payload must contain:
  - `code`
  - `message`
  - `module`
  - `level`
  - `stage`
  - `op`
  - `run_id`
  - `key`
  - `schema_version`
  - `standard_version`
  - `expected`
  - `actual`
  - `paths`
  - `ts_utc`
- `expected/actual/paths` can be empty objects but must exist.

Code form:
```text
payload = {
  code, message, module, level, stage, op, run_id, key,
  schema_version, standard_version,
  expected: {...}, actual: {...}, paths: {...},
  ts_utc
}
```

Validation:
- Missing required payload fields is a secondary hard failure.

## 5) Fixed Code Table (MUST)
MUST:
- The following codes and meanings are fixed:
  - `FXE-INPUT-001`: run input file missing.
  - `FXE-INPUT-002`: run input field missing.
  - `FXE-INPUT-003`: run input type/value-domain invalid.
  - `FXE-SCHEMA-001`: schema version mismatch.
  - `FXE-SCHEMA-002`: payload field/type mismatch.
  - `FXE-BIND-001`: `basis_id` mismatch.
  - `FXE-BIND-002`: `orbital_order_id` mismatch.
  - `FXE-BIND-003`: axis-length/order mismatch (including illegal extra channel axis).
  - `FXE-IO-001`: required artifact missing.
  - `FXE-IO-002`: artifact hash mismatch.
  - `FXE-IO-003`: atomic write failure.
  - `FXE-NUM-001`: Hermiticity/orthogonality check failure.
  - `FXE-NUM-002`: denominator near-zero/unstable.
  - `FXE-NUM-003`: eigensolver/SVD convergence failure.
  - `FXE-PHYS-001`: forbidden quantum-number/sector condition.
  - `FXE-W90-001`: Wannier90 file parse failure.
  - `FXE-W90-002`: atom/orbital/spin mapping failure.
  - `FXE-W90-003`: orbital order/unit validation failure.
  - `FXE-LEGACY-001`: attempted read of `*_LEGACY` spec in normal mode.
  - `FXE-LEGACY-002`: attempted load of legacy artifact in normal mode.
  - `FXE-RUNTIME-001`: normalized uncategorized runtime failure.

Validation:
- Implementations may add new codes but must not redefine existing ones.

## 6) Failure-Context Minimum per Domain (MUST)
MUST:
- `INPUT/SCHEMA`: include `field_path`, `expected_type`, `actual_type`.
- `BIND`: include `lhs_id`, `rhs_id`, `lhs_shape`, `rhs_shape`.
- `IO`: include `path`, `exists`, `hash_expected`, `hash_actual`.
- `NUM`: include `residual_name`, `residual_value`, `threshold`.
- `W90`: include `site_index`, `orbital_index`, `spin_mode`, `unit`.
- `LEGACY`: include `requested_path`, `mode_flags`.

Code form:
```text
payload.actual += domain_specific_context
```

Validation:
- Missing domain-specific context is a payload contract error.

## 7) Serialization and Persistence (MUST)
MUST:
- Emit machine-readable JSON payload.
- Persist one-line error record to index/log stream.
- `code` is the primary key for downstream triage.

Code form:
```text
write_json(error_payload)
append(index_errors.jsonl, error_payload)
```

Validation:
- Non-JSON error output is forbidden for automated pipelines.
