# 00-conventions/00-01-SOFTWARE_ENGINEERING

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
- 子包布局须对应规范层级：

```text
fexchange/
  __init__.py
  core/
    __init__.py
    fock_basis.py          # 01-00: Fock/Slater 基础
    fermion_ops.py         # 01-02: 算符实现
    orbitals.py            # 01-00/02-04: 轨道映射、角动量
    states.py              # 01-01: 态矢量对象 (BasisDet, StateVec, StateSet)
  models/
    __init__.py
    hint.py                # 02-01: H_int 构造
    hsoc.py                # 02-02: H_soc 构造
    hcef.py                # 02-03: H_cef + Stevens 算符
    angular.py             # 02-04: L/S/J 算符构建
    kramers.py             # 02-05: Kramers 双重态 + g 张量
    non_kramers.py         # 02-06: 非 Kramers 双重态
  representations/
    __init__.py
    lsms.py                # 03-00: LSMS 表象
    lsjm.py                # 03-01: LSJM 表象
  sopt/
    __init__.py
    precompute.py           # 04-01: L0 + L1
    contraction.py          # 04-02: L2 + L3 + L4
    spin12.py               # 04-03: 自旋-1/2 映射
  io/
    __init__.py
    disk.py                 # 05-00: 磁盘 I/O、路径令牌、缓存
    wannier90.py            # 05-02+05-03: Wannier90 解析与输入契约
    run_input.py            # 00-05: TOML 运行输入加载
  utils/
    __init__.py
    numerics.py             # 00-02: 容差表、dtype 策略
    errors.py               # 00-03: 错误码、编码异常
    checks.py               # 运行时检查（厄米性、正交归一性）
    parallel.py             # 00-06: MPI 封装 / 串行回退
  cli.py                    # 入口点
```

Validation:
- 所有公共模块须可通过 `from fexchange.<sub> import <mod>` 导入。

## 3) 依赖（MUST）
MUST:
- 核心数值：`numpy >= 1.24`、`scipy >= 1.10`。
- 文件 I/O：`tomli`（或 Python >= 3.11 的标准库 `tomllib`）。
- 3j/CG 符号：`sympy >= 1.12`（`sympy.physics.wigner`；唯一 3j/CG 实现）。
- 并行：`mpi4py >= 3.1`（可选；串行回退为必需）。
- 测试：`pytest >= 7.0`。
- 无其它硬性运行时依赖。

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
- MPI 为运行时可选：导入失败须回退到串行模式。

## 4) 入口点与 CLI（MUST）
MUST:
- 提供一个 CLI 命令：`fexchange run <run_input.toml>`。
- 该命令按 `./standards/en/00-conventions/00-05-RUN_INPUT_SINGLE_FILE.md` 读取 TOML 文件，
  执行指定级别窗口，按 `./standards/en/05-io/05-00-IO.md` 写入输出。

Code form:
```text
fexchange run ./run_input.toml
```

Validation:
- 成功时退出码 `0`，失败时非零。
- 失败时按 `./standards/en/00-conventions/00-03-ERROR_CODES_AND_FAILURE_PAYLOAD.md` 向 stderr 输出 JSON 错误载荷。

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
  test_sopt_l0.py          # 04-01: X/Y 符号一致性
  test_sopt_l1.py          # 04-01: A/B 顶点维度
  test_sopt_l2_l4.py       # 04-02: 零跃迁检查、Heff 厄米性
  test_spin12.py           # 04-03: 重构残差
  test_io.py               # 05-00: 路径令牌生成、缓存往返
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
  `level`、`key`、`elapsed_s`、`numerics_meta`、`parallel_meta`。
- 随机种子（若有）须记录；优先使用无随机性的确定性算法。

Code form:
```text
import logging
logger = logging.getLogger("fexchange")
```
