# 科学计算规范

## 权威性
代码实现必须遵循 `./standards/en/` 下的英文规范。
`./standards/zh/` 仅为中文翻译。
若 EN 与 ZH 不一致，以 EN 为准。
默认情况下，`*_LEGACY` 文件不具备规范效力。

## 目录结构

### 00-conventions/ — 项目约定、软件工程、运行时基础设施
- `./standards/en/00-conventions/00-00-SPEC_WRITING_CONVENTION.md`
- `./standards/en/00-conventions/00-01-SOFTWARE_ENGINEERING.md`
- `./standards/en/00-conventions/00-02-RUNTIME_NUMERICS_AND_INPUT_GATES.md`
- `./standards/en/00-conventions/00-03-ERROR_CODES_AND_FAILURE_PAYLOAD.md`
- `./standards/en/00-conventions/00-04-LEGACY_ISOLATION.md`
- `./standards/en/00-conventions/00-05-RUN_INPUT_SINGLE_FILE.md`
- `./standards/en/00-conventions/00-06-MPI_PARALLEL_RUNTIME.md`
- `./standards/en/00-conventions/00-07-CONFLICT_REPORT_TEMPLATE.md`

### 01-physics/ — 核心物理基础
- `./standards/en/01-physics/01-00-FOUNDATIONS_FOCK_SLATER.md`
- `./standards/en/01-physics/01-01-STATE_VECTOR_CONVENTION.md`
- `./standards/en/01-physics/01-02-OPERATOR_IMPLEMENTATION.md`

### 02-models/ — 局域哈密顿量与模型定义
- `./standards/en/02-models/02-00-MODEL_LOCAL_HAMILTONIAN.md`
- `./standards/en/02-models/02-01-HINT_FORM.md`
- `./standards/en/02-models/02-02-HSOC_FORM.md`
- `./standards/en/02-models/02-03-HCEF_FORM.md`
- `./standards/en/02-models/02-04-ANGULAR_MOMENTUM_OPERATORS.md`
- `./standards/en/02-models/02-05-KRAMERS_DOUBLET_G_TENSOR.md`
- `./standards/en/02-models/02-06-NON_KRAMERS_DOUBLET.md`

### 03-representations/ — 基表示（LSMS、LSJM）
- `./standards/en/03-representations/03-00-REPRESENTATION_LSMS.md`
- `./standards/en/03-representations/03-01-REPRESENTATION_LSJM.md`

### 04-sopt/ — 二阶微扰论
- `./standards/en/04-sopt/04-00-SOPT_FORMALISM.md`
- `./standards/en/04-sopt/04-01-PRECOMPUTE_PIPELINE.md`
- `./standards/en/04-sopt/04-02-RUNTIME_CONTRACTION.md`
- `./standards/en/04-sopt/04-03-SPIN12_MAPPING.md`
- `./standards/en/04-sopt/04-02-RUNTIME_CONTRACTION_LEGACY.md`（legacy，非规范）

### 05-io/ — 输入/输出、文件格式、Wannier90 集成
- `./standards/en/05-io/05-00-IO.md`
- `./standards/en/05-io/05-01-IO_LAYOUT_GUIDE.md`（非规范）
- `./standards/en/05-io/05-02-WANNIER90_INPUT_CONTRACT.md`
- `./standards/en/05-io/05-03-WANNIER90_PARSING_RULES.md`

## 阅读顺序（MUST）
1. `./standards/en/00-conventions/00-00-SPEC_WRITING_CONVENTION.md`
2. `./standards/en/00-conventions/00-01-SOFTWARE_ENGINEERING.md`
3. `./standards/en/01-physics/01-00-FOUNDATIONS_FOCK_SLATER.md`
4. `./standards/en/01-physics/01-01-STATE_VECTOR_CONVENTION.md`
5. `./standards/en/01-physics/01-02-OPERATOR_IMPLEMENTATION.md`
6. `./standards/en/00-conventions/00-02-RUNTIME_NUMERICS_AND_INPUT_GATES.md`
7. `./standards/en/00-conventions/00-03-ERROR_CODES_AND_FAILURE_PAYLOAD.md`
8. `./standards/en/00-conventions/00-04-LEGACY_ISOLATION.md`
9. `./standards/en/00-conventions/00-05-RUN_INPUT_SINGLE_FILE.md`
10. `./standards/en/00-conventions/00-06-MPI_PARALLEL_RUNTIME.md`
11. `./standards/en/05-io/05-00-IO.md`
12. `./standards/en/05-io/05-01-IO_LAYOUT_GUIDE.md`（可选/非规范）
13. `./standards/en/05-io/05-02-WANNIER90_INPUT_CONTRACT.md`
14. `./standards/en/05-io/05-03-WANNIER90_PARSING_RULES.md`
15. `./standards/en/02-models/02-00-MODEL_LOCAL_HAMILTONIAN.md` + `02-01` 至 `02-06`
16. `./standards/en/03-representations/03-00-REPRESENTATION_LSMS.md` 再 `./standards/en/03-representations/03-01-REPRESENTATION_LSJM.md`
17. `./standards/en/04-sopt/04-00-SOPT_FORMALISM.md`
18. `./standards/en/04-sopt/04-01-PRECOMPUTE_PIPELINE.md`
19. `./standards/en/04-sopt/04-02-RUNTIME_CONTRACTION.md`
20. `./standards/en/04-sopt/04-03-SPIN12_MAPPING.md`

## 冲突规则（MUST）
若规范冲突，必须立即停止，并使用下列模板上报：
- `./standards/en/00-conventions/00-07-CONFLICT_REPORT_TEMPLATE.md`
