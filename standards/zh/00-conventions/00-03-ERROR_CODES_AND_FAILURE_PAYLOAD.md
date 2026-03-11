# 00-03-ERROR_CODES_AND_FAILURE_PAYLOAD

本文件定义固定运行时错误码与必填失败载荷字段。
本文件对所有模块具有规范约束力。

## 1) 作用范围（MUST）
MUST:
- 适用于模块 `03-00/03-01/04-00/04-01/04-02/04-03/05-00/05-02/05-03`。
- 每个硬失败必须输出一个固定错误码。
- 对外接口禁止直接抛出无错误码的原始异常。

Code form:
```text
if failure: raise CodedError(code, payload)
```

Validation:
- 对外错误若无 `code`，视为契约违规。

## 2) 错误码格式（MUST）
MUST:
- 错误码格式固定为：
  `FXE-{DOMAIN}-{NNN}`。
- `DOMAIN` 必须为大写固定域名。
- `NNN` 为三位零填充十进制编号。
- 一旦发布，错误码语义不可重映射。

Code form:
```text
code = f"FXE-{DOMAIN}-{id:03d}"
```

Validation:
- 不符合格式的错误码必须被序列化层拒绝。

## 3) 域名集合（MUST）
MUST:
- 允许的域名：
  - `INPUT`：运行输入文件缺失/非法。
  - `SCHEMA`：schema/字段/类型校验失败。
  - `BIND`：轴/基组/顺序绑定不一致。
  - `IO`：文件路径/hash/读写/原子写失败。
  - `NUM`：数值稳定性/阈值/线性代数失败。
  - `PHYS`：物理约束违规。
  - `W90`：Wannier90 解析/映射失败。
  - `LEGACY`：非法访问旧规范/旧工件。
  - `RUNTIME`：归一化后仍无法归类的运行错误。

Validation:
- 域名不在集合内则无效。

## 4) 必填载荷 Schema（MUST）
MUST:
- 每个失败载荷必须包含：
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
- `expected/actual/paths` 允许为空对象，但字段必须存在。

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
- 缺失必填字段会触发二次硬失败。

## 5) 固定错误码表（MUST）
MUST:
- 下列错误码与语义固定：
  - `FXE-INPUT-001`：运行输入文件不存在。
  - `FXE-INPUT-002`：运行输入字段缺失。
  - `FXE-INPUT-003`：运行输入类型/取值域非法。
  - `FXE-SCHEMA-001`：schema 版本不匹配。
  - `FXE-SCHEMA-002`：载荷字段/类型不匹配。
  - `FXE-BIND-001`：`basis_id` 不匹配。
  - `FXE-BIND-002`：`orbital_order_id` 不匹配。
  - `FXE-BIND-003`：轴长度/顺序不匹配（含非法额外通道轴）。
  - `FXE-IO-001`：必需工件缺失。
  - `FXE-IO-002`：工件 hash 不一致。
  - `FXE-IO-003`：原子写失败。
  - `FXE-NUM-001`：厄米性/正交性检查失败。
  - `FXE-NUM-002`：分母近零/数值不稳定。
  - `FXE-NUM-003`：特征分解/SVD 收敛失败。
  - `FXE-PHYS-001`：量子数/扇区物理约束违规。
  - `FXE-W90-001`：Wannier90 文件解析失败。
  - `FXE-W90-002`：原子/轨道/自旋映射失败。
  - `FXE-W90-003`：轨道顺序/单位校验失败。
  - `FXE-LEGACY-001`：普通模式读取 `*_LEGACY` 规范。
  - `FXE-LEGACY-002`：普通模式加载 legacy 工件。
  - `FXE-RUNTIME-001`：归一化后未分类运行错误。

Validation:
- 允许新增错误码，但禁止重定义既有错误码语义。

## 6) 各域最小失败上下文（MUST）
MUST:
- `INPUT/SCHEMA`：必须包含 `field_path`, `expected_type`, `actual_type`。
- `BIND`：必须包含 `lhs_id`, `rhs_id`, `lhs_shape`, `rhs_shape`。
- `IO`：必须包含 `path`, `exists`, `hash_expected`, `hash_actual`。
- `NUM`：必须包含 `residual_name`, `residual_value`, `threshold`。
- `W90`：必须包含 `site_index`, `orbital_index`, `spin_mode`, `unit`。
- `LEGACY`：必须包含 `requested_path`, `mode_flags`。

Code form:
```text
payload.actual += domain_specific_context
```

Validation:
- 缺失域特定上下文字段是载荷契约错误。

## 7) 序列化与持久化（MUST）
MUST:
- 必须输出机器可读 JSON 失败载荷。
- 必须向索引/日志流追加单行错误记录。
- `code` 是下游排障与统计的主键。

Code form:
```text
write_json(error_payload)
append(index_errors.jsonl, error_payload)
```

Validation:
- 自动化流水线中禁止非 JSON 错误输出。
