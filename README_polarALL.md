# polarALL.py - 可配置精度的极坐标量子计算

## 概述

`polarALL.py` 是基于 `ALLeularV5.py` 的极坐标量子计算实现，支持多种精度配置（int8, float16, float32, double），用于与纯复数版本进行精确对比测试。

## 核心特性

### 1. **可配置精度**
- **int8**: 传统极坐标编码（幅度和相位分别编码）
- **float16**: 半精度浮点数
- **float32**: 单精度浮点数
- **double**: 双精度浮点数（最高精度）

### 2. **统一的接口**
```python
def process_sequence_polar(initial_vec: List[complex],
                          seq: List[Tuple[str, str, List]],
                          precision: PrecisionType = 'int8',
                          verbose: bool = False) -> Tuple[torch.Tensor, List[torch.Tensor]]:
    """
    处理量子门序列，支持多种精度

    Args:
        initial_vec: 初始状态向量（复数形式）
        seq: 量子门序列
        precision: 精度类型 ('int8', 'float16', 'float32', 'double')
        verbose: 是否打印详细信息

    Returns:
        (final_vec, state_history): 最终状态向量和历史记录
    """
```

### 3. **自动精度转换**
- 输入：复数向量 → 极坐标编码
- 计算：极坐标表示进行量子运算
- 输出：可转换为复数进行对比

## 使用方法

### 基本使用

```python
from polarALL import process_sequence_polar, create_initial_vec_like_v2, random_sequence_tensor

# 创建初始向量（与 ALLeularV2.py 第903行相同）
initial_vec = create_initial_vec_like_v2(32)

# 生成序列
seq = random_sequence_tensor(50, 50, 50, shuffle=True)

# 使用 int8 精度处理
final_vec, history = process_sequence_polar(initial_vec, seq, precision='int8')

# 使用 float32 精度处理
final_vec_fp32, history_fp32 = process_sequence_polar(initial_vec, seq, precision='float32')
```

### 精度选项说明

| 精度 | 优势 | 劣势 | 适用场景 |
|------|------|------|----------|
| **int8** | 内存占用最小 | 量化误差大 | 传统对比测试 |
| **float16** | 内存占用小，GPU加速好 | 精度有限 | 大规模模拟 |
| **float32** | 平衡精度和性能 | 内存占用中等 | 一般科学计算 |
| **double** | 最高精度 | 内存和计算开销大 | 精确数值验证 |

## 与 complexALL.py 的对比

### 功能对比

| 特性 | complexALL.py | polarALL.py |
|------|---------------|-------------|
| **计算方式** | 纯复数运算 | 极坐标编码 |
| **精度** | 机器精度 | 可配置 (int8/float16/float32/double) |
| **内存效率** | 标准 | int8模式下最优 |
| **GPU加速** | 无 | PyTorch优化 |
| **用途** | 高精度基准 | 多精度对比测试 |

### 误差分析

```python
# 使用 test_error.py 进行对比
from test_error import compare_implementations

# 对比 int8 版本
result_int8 = compare_implementations(precision='int8')

# 对比 float32 版本
result_float32 = compare_implementations(precision='float32')
```

## 文件结构

- `polarALL.py`: 主要的极坐标计算实现
- `complexALL.py`: 纯复数版本（基准）
- `test_error.py`: 误差对比测试
- `demo_polarALL.py`: 使用演示
- `test_precision_comparison.py`: 精度对比测试

## 技术细节

### 编码转换流程

```
复数向量 → 极坐标编码 → 量子门计算 → 极坐标解码 → 复数向量
    ↓           ↓           ↓           ↓           ↓
[complex] → [amplitude, phase] → 门运算 → [amplitude, phase] → [complex]
```

### 精度特定的处理

- **int8**: 使用专门的编码/解码函数处理量化
- **浮点类型**: 直接使用 PyTorch 张量操作

## 运行测试

```bash
# 基本功能测试
python polarALL.py

# 精度对比测试
python test_precision_comparison.py

# 与 complexALL 对比测试
python test_error.py

# 使用演示
python demo_polarALL.py
```

## 注意事项

1. **内存使用**: int8 模式最省内存，但精度最低
2. **计算速度**: PyTorch GPU 加速在 float32/double 模式下最有效
3. **数值精度**: double 模式提供最高精度，但计算开销最大
4. **兼容性**: 与 complexALL.py 使用相同的序列格式和初始向量

## 扩展性

该框架设计为可扩展的，可以轻松添加：
- 新的精度类型
- 新的量子门
- 不同的编码方案
- 优化的计算内核

## 总结

`polarALL.py` 提供了灵活的极坐标量子计算框架，支持从低精度 int8 到高精度 double 的完整范围，是验证量子计算数值稳定性和精度权衡的理想工具。⚛️🔬📊