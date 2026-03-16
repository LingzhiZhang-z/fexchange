# 科学计算规范

## 权威性
代码实现必须遵循 `./standards/en/` 下的英文规范。
`./standards/zh/` 仅为中文翻译。
若 EN 与 ZH 不一致，以 EN 为准。

## 目录结构

### 00-meta/ — 写作约定、软件工程、冲突上报
- `./standards/en/00-meta/00-00-SPEC_WRITING_CONVENTION.md`
- `./standards/en/00-meta/00-01-SOFTWARE_ENGINEERING.md`
- `./standards/en/00-meta/00-07-CONFLICT_REPORT_TEMPLATE.md`

### 01-core/ — Fock 空间、态矢量、费米子算符、Stevens 与张量算符、角动量
- `./standards/en/01-core/01-00-FOCK_SLATER.md`
- `./standards/en/01-core/01-01-STATE_VECTOR.md`
- `./standards/en/01-core/01-02-OPERATOR_IMPLEMENTATION.md`
- `./standards/en/01-core/01-03-STEVENS_OPERATORS.md`
- `./standards/en/01-core/01-04-SPHERICAL_TENSOR_OPERATORS.md`
- `./standards/en/01-core/01-05-STEVENS_TENSOR_CONVERSION.md`
- `./standards/en/01-core/01-06-ANGULAR_MOMENTUM.md`

### 02-hamiltonian/ — 局域哈密顿量
- `./standards/en/02-hamiltonian/02-00-LOCAL_HAMILTONIAN.md`
- `./standards/en/02-hamiltonian/02-01-HINT.md`
- `./standards/en/02-hamiltonian/02-02-HSOC.md`
- `./standards/en/02-hamiltonian/02-03-HCEF.md`

### 03-spectrum/ — LSMS/LSJM 表示、doublet、不可约表示分类
- `./standards/en/03-spectrum/03-00-LSMS.md`
- `./standards/en/03-spectrum/03-01-LSJM.md`
- `./standards/en/03-spectrum/03-02-KRAMERS_DOUBLET.md`
- `./standards/en/03-spectrum/03-03-NON_KRAMERS_DOUBLET.md`
- `./standards/en/03-spectrum/03-04-IRREP_CLASSIFICATION.md`

### 04-sopt/ — 二阶微扰论
- `./standards/en/04-sopt/04-00-SOPT_FORMALISM.md`
- `./standards/en/04-sopt/04-01-PRECOMPUTE.md`
- `./standards/en/04-sopt/04-02-RUNTIME_CONTRACTION.md`
- `./standards/en/04-sopt/04-03-SPIN12_MAPPING.md`
- `./standards/en/04-sopt/04-REF-DERIVATION.md`

### 05-io/ — 磁盘 I/O、布局、Wannier90、运行输入契约
- `./standards/en/05-io/05-00-IO.md`
- `./standards/en/05-io/05-01-IO_LAYOUT.md`
- `./standards/en/05-io/05-02-WANNIER90_CONTRACT.md`
- `./standards/en/05-io/05-03-WANNIER90_PARSING.md`
- `./standards/en/05-io/05-04-RUN_INPUT.md`

### 06-utils/ — 数值容差与错误契约
- `./standards/en/06-utils/06-00-RUNTIME_NUMERICS.md`
- `./standards/en/06-utils/06-01-ERROR_CODES.md`

## 映射说明
规范目录覆盖的是领域与横切契约。
`fexchange/pipeline/` 仍由这些规范约束，但通过 `AGENTS.md` 与 `CLAUDE.md`
中的模块映射体现，而不是单独新增 `07-pipeline/` 目录。

## 阅读顺序（MUST）
1. `./standards/en/00-meta/00-00-SPEC_WRITING_CONVENTION.md`
2. `./standards/en/00-meta/00-01-SOFTWARE_ENGINEERING.md`
3. `./standards/en/06-utils/06-00-RUNTIME_NUMERICS.md`
4. `./standards/en/06-utils/06-01-ERROR_CODES.md`
5. `./standards/en/01-core/01-00-FOCK_SLATER.md`
6. `./standards/en/01-core/01-01-STATE_VECTOR.md`
7. `./standards/en/01-core/01-02-OPERATOR_IMPLEMENTATION.md`
8. `./standards/en/01-core/01-03-STEVENS_OPERATORS.md`
9. `./standards/en/01-core/01-04-SPHERICAL_TENSOR_OPERATORS.md`
10. `./standards/en/01-core/01-05-STEVENS_TENSOR_CONVERSION.md`
11. `./standards/en/01-core/01-06-ANGULAR_MOMENTUM.md`
12. `./standards/en/02-hamiltonian/02-00-LOCAL_HAMILTONIAN.md`
13. `./standards/en/02-hamiltonian/02-01-HINT.md`
14. `./standards/en/02-hamiltonian/02-02-HSOC.md`
15. `./standards/en/02-hamiltonian/02-03-HCEF.md`
16. `./standards/en/03-spectrum/03-00-LSMS.md`
17. `./standards/en/03-spectrum/03-01-LSJM.md`
18. `./standards/en/03-spectrum/03-02-KRAMERS_DOUBLET.md`
19. `./standards/en/03-spectrum/03-03-NON_KRAMERS_DOUBLET.md`
20. `./standards/en/03-spectrum/03-04-IRREP_CLASSIFICATION.md`
21. `./standards/en/04-sopt/04-00-SOPT_FORMALISM.md`
22. `./standards/en/04-sopt/04-01-PRECOMPUTE.md`
23. `./standards/en/04-sopt/04-02-RUNTIME_CONTRACTION.md`
24. `./standards/en/04-sopt/04-03-SPIN12_MAPPING.md`
25. `./standards/en/04-sopt/04-REF-DERIVATION.md`
26. `./standards/en/05-io/05-00-IO.md`
27. `./standards/en/05-io/05-01-IO_LAYOUT.md`
28. `./standards/en/05-io/05-02-WANNIER90_CONTRACT.md`
29. `./standards/en/05-io/05-03-WANNIER90_PARSING.md`
30. `./standards/en/05-io/05-04-RUN_INPUT.md`

## 冲突规则（MUST）
若规范冲突，必须立即停止，并使用下列模板上报：
- `./standards/en/00-meta/00-07-CONFLICT_REPORT_TEMPLATE.md`
