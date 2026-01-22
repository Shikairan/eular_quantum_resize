# complexALL.py - 纯复数量子计算实现

## 概述

`complexALL.py` 是基于 `ALLeularV2.py` 的纯复数量子计算实现，用于提供高精度的基准测试和对照验证。

## 核心特性

### 1. **纯复数运算**
- 直接使用 Python 复数类型进行计算
- 无量化误差，机器精度级别准确性
- 作为其他实现的基准参考

### 2. **完整量子门集合**
- **单比特门**: X, Y, Z, S, T, H, Rx, Ry, Rz, U1, U2, U3
- **控制门**: CNOT, CZ, CY, CH, CS, CT, CRx, CRy, CRz, CU1, CU2, CU3
- **矩阵运算**: 基于 NumPy 的高效矩阵乘法

### 3. **状态历史记录**
```python
def process_sequence_complex(seq, initial_vec, verbose=False):
    """
    处理量子门序列，返回最终状态和完整历史记录

    Returns:
        (final_vec, state_history): 每个门应用后的状态快照
    """
```

## 使用方法

### 基本使用（代码块 1-10）

```python
from complexALL import process_sequence_complex, create_initial_vec_like_v2

# 生成随机序列（与 ALLeularV2.py 相同）
seq = random_sequence(50, 50, 50, shuffle=True)

# 创建初始向量（与 ALLeularV2.py 第903行相同）
initial_vec = create_initial_vec_like_v2(32)

# 处理序列，获取最终状态和历史记录
final_vec, state_history = process_sequence_complex(seq, initial_vec, verbose=True)

print(f"记录了 {len(state_history)} 步状态变化")
print(f"每步都可以访问: state_history[step]")
```

### 高级功能

```python
# 单独应用门
from complexALL import apply_gate_unified_complex

vec = create_initial_vec_like_v2(8)
apply_gate_unified_complex(vec, 'H', qubit_idx=0)  # Hadamard 门
apply_gate_unified_complex(vec, 'CX', control_idx=0, target_idx=1)  # CNOT 门
apply_gate_unified_complex(vec, 'Rx', math.pi/4, qubit_idx=2)  # 参数化门
```

## 技术实现

### 门矩阵定义

```python
# 基本门矩阵
H_MAT = np.array([[1, 1], [1, -1]], dtype=complex) / math.sqrt(2)
X_MAT = np.array([[0, 1], [1, 0]], dtype=complex)
Y_MAT = np.array([[0, -1j], [1j, 0]], dtype=complex)
Z_MAT = np.array([[1, 0], [0, -1]], dtype=complex)

# 相位门
S_MAT = np.array([[1, 0], [0, 1j]], dtype=complex)
T_MAT = np.array([[1, 0], [0, cmath.exp(1j * math.pi / 4)]], dtype=complex)
```

### 状态向量处理

```python
def apply_gate_pair(vec, gate_func, *args, qubit_idx=0):
    """
    对指定比特应用量子门
    通过状态对操作实现高效计算
    """
    n = len(vec)
    step = 1 << qubit_idx
    block_size = step << 1

    for base in range(0, n, block_size):
        for offset in range(0, step):
            idx0 = base + offset
            idx1 = base + offset + step
            vec[idx0], vec[idx1] = gate_func(vec[idx0], vec[idx1], *args)
```

## 与其他实现的对比

### 功能对比

| 特性 | complexALL.py | polarALL.py | ALLeularV5.py |
|------|---------------|-------------|---------------|
| **精度** | 机器精度 | 可配置 | int8 固定 |
| **速度** | 中等 | 最快 | 中等 |
| **内存** | 标准 | 最省 | 最省 |
| **用途** | 基准测试 | 生产使用 | 兼容性 |

### 性能特点

- **complexALL.py**: 高精度基准，适合验证算法正确性
- **polarALL.py**: 平衡性能和精度，支持多种配置
- **ALLeularV5.py**: 传统 int8 编码，内存最优

## 文件结构

```
complexALL.py          # 主实现文件
├── 量子门矩阵定义        # H_MAT, X_MAT, Y_MAT, Z_MAT 等
├── 量子门函数           # hadamard_complex_pair, x_complex_pair 等
├── 控制门处理          # apply_cy_complex, apply_cnot_complex 等
├── 核心应用函数        # apply_gate_pair, apply_gate_unified_complex
├── 序列生成功能        # random_sequence
├── 序列处理主函数      # process_sequence_complex
└── 辅助工具函数        # create_initial_vec_like_v2, compute_normalization_error
```

## 运行示例

```bash
# 基本测试
python complexALL.py

# 与其他实现对比
python test_error.py

# 功能演示
python test_complexALL_demo.py
```

## 扩展性

该框架设计为可扩展的，可以轻松添加：
- 新的量子门类型
- 不同的门分解策略
- 优化的计算内核
- 批量处理优化

## 注意事项

1. **性能**: NumPy 实现比 PyTorch 稍慢，但精度更高
2. **内存**: 使用标准 Python 复数，内存占用适中
3. **兼容性**: 与 ALLeularV2.py 完全兼容的接口
4. **扩展性**: 易于添加新的量子门和功能

## 总结

`complexALL.py` 提供了纯复数量子计算的高精度基准实现，是验证其他量化实现正确性和精度的理想工具。⚛️🔬📊