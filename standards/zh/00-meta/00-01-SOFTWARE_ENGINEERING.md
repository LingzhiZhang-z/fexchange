# 00-meta/00-01-SOFTWARE_ENGINEERING

本文件定义实现 `./standards/en/` 中各规范所需的软件工程约定。
对代码结构、依赖和测试具有规范效力。

## 1) 语言与运行时（MUST）
MUST:
- 实现语言：Python >= 3.10。
- 参考实现不要求 Cython 或 C 扩展。
- 建议使用类型提示但非强制。

Code form:
```text
python_version >= "3.10"
```

## 2) 包结构（MUST）
MUST:
- 使用单一顶层包：`fexchange`。
- 存在专门规范目录的领域子包应与规范层级对应。
- `pipeline/` 是显式例外：它由横切规范以及 `AGENTS.md` / `CLAUDE.md`
  中的模块映射约束，而不是单独对应一个规范目录。
- 当前包布局为：

```text
fexchange/
  __init__.py
  core/
    __init__.py
    fock.py                # 01-00: Fock/Slater 基础
    fermion.py             # 01-02: 算符实现
    states.py              # 01-01: 态矢量约定
    stevens.py             # 01-03/01-04/01-05: Stevens 与张量算符
    space_j.py             # 01-06: J 空间角动量算符
    space_ls.py            # 01-06: LS 空间角动量算符
  hamiltonian/
    __init__.py
    hint.py                # 02-01: H_int 构造
    hsoc.py                # 02-02: H_soc 构造
    hcef.py                # 02-03: H_cef 构造
  spectrum/
    __init__.py
    lsms.py                # 03-00: LSMS 表象
    lsjm.py                # 03-01: LSJM 表象
    energy.py              # 02-00: 中间态能量重构
    doublet.py             # 03-02/03-03: Kramers / 非 Kramers 双重态
    ground.py              # 03-02/03-03: 基态双重态选取
    classify.py            # 03-04: 不可约表示分类
    tables.py              # 03-04: 对称角色表
    multipole.py           # 01-03/01-04: 多极矩展示
  sopt/
    __init__.py
    precompute.py          # 04-01: L0 + L1
    contraction.py         # 04-02: L2 + L3
    spin12.py              # 04-03: 自旋-1/2 映射
  fopt/
    __init__.py
    preprocessing.py       # 04-fopt: L0 + L1 + L2 active-pair blocks
  io/
    __init__.py
    disk.py                # 05-00: 磁盘 I/O、路径令牌、缓存
    matrix.py              # 05-00: 矩阵序列化
    wannier90.py           # 05-02+05-03: Wannier90 解析与输入契约
    run_input.py           # 05-04: TOML 运行输入加载
  pipeline/
    __init__.py
    artifacts.py           # 工件持久化与元数据
    keys.py                # pipeline key 生成
    resolve.py             # 依赖解析
    stages.py              # 阶段执行编排
    validation.py          # pipeline 级校验
  sweep/
    __init__.py            # parameter sweep public exports
    expand.py              # 05-05: pure sweep-table expansion
    runner.py              # 05-05: serial/MPI sweep orchestration
  utils/
    __init__.py
    numerics.py            # 06-00: 容差表、dtype 策略
    errors.py              # 06-01: 错误码、编码异常
    checks.py              # 运行时检查（厄米性、正交归一性）
    constants.py           # 物理常数（ELL, N_ORB）
  cli.py                   # 入口点
```

Validation:
- 所有公共模块须可通过 `from fexchange.<sub> import <mod>` 导入。

## 3) 依赖（MUST）
MUST:
- 核心数值：`numpy >= 1.24`、`scipy >= 1.10`。
- 文件 I/O：`tomli`（或 Python >= 3.11 的标准库 `tomllib`）。
- 3j/CG 符号：`sympy >= 1.12`（`sympy.physics.wigner`；唯一 3j/CG 实现）。
- 测试：`pytest >= 7.0`。
- 无其它硬性运行时依赖。
- MPI sweep support 是可选依赖，仅当安装 optional `mpi` extra 时使用
  `mpi4py >= 3.1`。

Code form:
```toml
[project]
name = "fexchange"
requires-python = ">=3.10"
dependencies = [
  "numpy>=1.24",
  "scipy>=1.10",
  "sympy>=1.12",
]

[project.optional-dependencies]
mpi    = ["mpi4py>=3.1"]
dev    = ["pytest>=7.0"]
```

Validation:
- `pip install .` 仅需核心依赖即可成功。
- `pip install .[mpi]` 支持在 MPI launcher 下运行 `fexchange sweep`。

## 4) 入口点与 CLI（MUST）
MUST:
- 提供两个 CLI 命令：
  - `fexchange run <run_input.toml>`
  - `fexchange sweep <base_run_input_with_sweep.toml>`
- `fexchange run` 按 `./standards/en/05-io/05-04-RUN_INPUT.md` 读取一个 TOML 文件，
  执行指定级别窗口，按 `./standards/en/05-io/05-00-IO.md` 写入输出。
- `fexchange sweep` 按 `./standards/en/05-io/05-05-SWEEP_INPUT.md` 读取一个带
  `[sweep]` 表的 base TOML，在内存中 materialize 每个 case，并通过与
  `fexchange run` 相同的 runtime pipeline 执行每个 case。

Code form:
```text
fexchange run ./run_input.toml
fexchange sweep ./sweep_base.toml
```

Validation:
- 成功时退出码 `0`，失败时非零。
- 失败时按 `./standards/en/06-utils/06-01-ERROR_CODES.md` 向 stderr 输出 JSON 错误载荷。

## 5) 测试框架（MUST）
MUST:
- 使用 `pytest` 作为测试运行器。
- 测试目录布局：

```text
tests/
  conftest.py              # 共享 fixture（小 Fock 基、已知 LS 项）
  test_fock_basis.py       # 01-00: 位串、符号、维度
  test_fermion_ops.py      # 01-02: 反对易、厄米一致性
  test_hint.py             # 02-01: 库仑算符、已知项能量
  test_hsoc.py             # 02-02: SOC 厄米性、Landé 交叉验证
  test_hcef.py             # 02-03: Stevens 算符厄米性、Oh/C3v 检查
  test_lsms.py             # 03-00: 正交归一性、项数、H_int 对角性
  test_lsjm.py             # 03-01: CG 交叉验证、SOC 对角性
  test_energy.py           # 02-00: 能量重构
  test_ground_doublets.py  # 03-02/03-03: Kramers / 非 Kramers 双重态
  test_sopt_l0.py          # 04-01: X/Y 符号一致性
  test_sopt_l1.py          # 04-01: A/B 顶点维度
  test_sopt_l2_l3.py       # 04-02: 零跃迁检查、Heff 厄米性
  test_run_input.py        # 05-04: TOML 输入 schema 校验
  test_wannier90.py        # 05-03: 解析冒烟测试
```

- 最低必需测试类别：
  1. 随机 Slater 行列式上的反对易恒等式检查。
  2. 所有构造算符的厄米一致性（`H == H.conj().T`）。
  3. LSMS/LSJM 态集的正交归一性。
  4. 零跃迁健全性检查：`t=0 => Heff=0`。
  5. 自旋-1/2 映射重构残差。

Code form:
```text
pytest tests/ -v --tb=short
```

## 6) 模块接口模式（MUST）
MUST:
- 每个计算模块须暴露一个公共函数或类，具有：
  - 显式类型化输入（numpy 数组 + 元数据字典），
  - 显式类型化输出（numpy 数组 + 元数据字典），
  - 入口和出口处的运行时检查。
- 内部辅助函数须以 `_` 前缀（按约定为私有）。

Code form:
```text
def build_lsms(n_ele, F2, F4, F6, basis_id, ...) -> LsmsResult:
    _validate_inputs(...)
    ...  # 计算
    _validate_outputs(result)
    return result
```

Validation:
- 无输入校验的公共函数违反契约。

## 7) 日志与可复现性（MUST）
MUST:
- 使用 Python `logging` 模块输出运行时信息。
- 每个阶段须发出结构化摘要，包含：
  `level`、`key`、`elapsed_s`，以及实际使用到的数值/runtime metadata。
  `numerics_meta` 是一个允许的容器名，不是强制字段名。
- 随机种子（若有）须记录；优先使用无随机性的确定性算法。

Code form:
```text
import logging
logger = logging.getLogger("fexchange")
```
