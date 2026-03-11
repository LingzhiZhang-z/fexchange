# 00-04-LEGACY_ISOLATION

本文件定义旧规范/旧工件的强制隔离规则。
目标：防止误读 `*_LEGACY` 冻结内容。

## 1) 作用范围（MUST）
MUST:
- 适用于所有规范读取器、校验器与运行时加载器。
- 同时约束 legacy 文件与 legacy 目录。

Code form:
```text
legacy_isolation_applies = true
```

Validation:
- 所有默认路径解析逻辑都必须执行本文件规则。

## 2) Legacy 识别规则（MUST）
MUST:
- 路径满足任一条件即视为 legacy：
  - 文件名包含 `_LEGACY`
  - 文件名以 `.legacy` 结尾
  - 位于 `legacy/` 子目录
- 识别必须是词法确定的，不依赖上下文猜测。

Code form:
```text
is_legacy(path) = contains("_LEGACY") or endswith(".legacy") or in_subtree("legacy/")
```

Validation:
- 被识别为 legacy 的路径在默认模式下不得作为规范源。

## 3) 默认读取策略（MUST）
MUST:
- 默认模式禁止读取任何 legacy 规范/工件。
- 默认发现列表必须隐藏 legacy 项。
- `README`/必读序列只能包含规范文件。

Code form:
```text
allow_legacy = false
if is_legacy(path): raise LegacyAccessError(FXE-LEGACY-001/002)
```

Validation:
- 默认模式尝试读取 legacy 必须硬失败。

## 4) 显式覆盖策略（MUST）
MUST:
- 仅在显式开启时允许 legacy 访问：
  - `allow_legacy = true`
  - `legacy_scope`（精确 allowlist 路径）
  - `legacy_reason`（人工可读原因）
  - `legacy_ticket`（可追溯 id）
- 覆盖仅对当前会话生效，不得静默持久化。

Code form:
```text
require allow_legacy and legacy_scope and legacy_reason and legacy_ticket
```

Validation:
- 任一覆盖字段缺失则仍禁止访问 legacy。

## 5) 输出污染防护（MUST）
MUST:
- 若使用 legacy 输入，输出 metadata 必须标记：
  - `legacy_mode=true`
  - `legacy_inputs=[...]`
- legacy 模式输出不得进入正常 cache/index 通道。

Code form:
```text
if legacy_mode:
  meta.legacy_mode = true
  block_publish_to_normative_index()
```

Validation:
- 将 legacy 派生输出发布为规范输出是禁止行为。

## 6) Agent 读取闸门（MUST）
MUST:
- AI/agent 默认仅可读取规范路径；除非显式启用 legacy 覆盖。
- 规范文本与 legacy 文本冲突时，规范文本自动优先。

Code form:
```text
reader_paths = filter_not_legacy(all_paths) unless allow_legacy
```

Validation:
- 未授权 legacy 文本进入 prompt 上下文，视为闸门失败。

## 7) 失败错误码（MUST）
MUST:
- 错误码固定使用 `./standards/en/00-conventions/00-03-ERROR_CODES_AND_FAILURE_PAYLOAD.md`：
  - `FXE-LEGACY-001`：阻止读取 legacy 规范。
  - `FXE-LEGACY-002`：阻止加载 legacy 工件。

Validation:
- 未使用固定错误码的 legacy 失败报告无效。
