# PolarVector: 可配置混合精度极坐标量子向量

## 概述

`PolarVector` 是一个封装了极坐标量子状态向量操作的 Python 类，**幅度**和**相位**均可独立配置为 **int16 (高精度)** 或 **int8 (内存优化)**，默认均为 **int16**，提供灵活的量子状态表示。

## 主要特性

- **可配置混合精度**: 幅值与相位均可选 int8 或 int16，默认均为 int16
- **自动设备管理**: 支持 CPU/GPU 自动检测
- **完整的编解码**: 幅度和相位的高精度编解码
- **向量操作**: 创建、转换、验证和字符串表示
- **向后兼容**: 提供兼容性函数接口

## 技术规格

| 项目 | int16 模式 | int8 模式 |
|------|-----------|-----------|
| 幅度编码 | int16 (-32768 到 32767) | int8 (-128 到 127) |
| 相位编码 | int16 (-2048 到 2047) | int8 (-128 到 127) |
| 幅度范围 | [0.0, 1.0] (支持动态缩放) | [0.0, 1.0] (支持动态缩放) |
| 相位范围 | [-π, π] | [-π, π] |
| 内存效率 | 较高 | 最高 |
| 精度 | 高 | 中等 |
| 推荐用途 | 高精度计算 | 大规模模拟 |

## 使用方法

### 基本使用 (默认幅值与相位均为 int16)

```python
from vector import PolarVector

# 创建向量实例 (默认 amplitude_dtype=int16, phase_dtype=int16)
vector = PolarVector()  # 自动检测设备

# 创建随机量子状态
polar_vec, scale_vec = vector.create_random_state(4)  # 2-qubit 系统

# 显示状态
print(vector.to_string(polar_vec, scale_vec))
```

### 内存优化模式 (可配置幅值/相位)

```python
# 仅幅度 int8
vector_amp_int8 = PolarVector(amplitude_dtype=torch.int8)

# 仅相位 int8
vector_phase_int8 = PolarVector(phase_dtype=torch.int8)

# 幅值与相位均 int8（最小内存）
vector_full_int8 = PolarVector(amplitude_dtype=torch.int8, phase_dtype=torch.int8)

# 创建随机量子状态
polar_vec, scale_vec = vector_full_int8.create_random_state(4)
print(vector_full_int8.to_string(polar_vec, scale_vec))
```

### 从复数创建

```python
import torch

# 创建贝尔态 |00⟩ + |11⟩
bell_state = torch.tensor([
    0.707 + 0.0j,    # |00⟩
    0.0 + 0.0j,      # |01⟩
    0.0 + 0.0j,      # |10⟩
    0.707 + 0.0j     # |11⟩
], dtype=torch.complex64)

polar_vec, scale_vec = vector.create_from_complex(bell_state)
```

### 转换操作

```python
# 极坐标 -> 复数
complex_vec = vector.to_complex(polar_vec, scale_vec)

# 复数 -> 极坐标
polar_vec, scale_vec = vector.complex_to_polar_tensor(complex_vec)
```

### 向量验证

```python
is_valid = PolarVector.validate_vector(polar_vec, scale_vec)
print(f"向量有效: {is_valid}")
```

## API 参考

### 类方法

#### `PolarVector(device=None, amplitude_dtype=torch.int16, phase_dtype=torch.int16)`
构造函数，创建极坐标向量实例。

**参数:**
- `device` (torch.device, 可选): 计算设备，默认为自动检测 (CUDA优先)
- `amplitude_dtype` (torch.dtype, 可选): 幅度编码，`torch.int16` (默认) 或 `torch.int8`
- `phase_dtype` (torch.dtype, 可选): 相位编码，`torch.int16` (默认) 或 `torch.int8`

#### `create_random_state(n_amps)`
创建随机量子状态向量。

**参数:**
- `n_amps` (int): 向量长度，必须是2的幂次

**返回:**
- `Tuple[torch.Tensor, torch.Tensor]`: (极坐标向量, 缩放向量)

#### `create_from_complex(complex_vec)`
从复数向量创建极坐标向量。

**参数:**
- `complex_vec` (torch.Tensor): 复数向量

**返回:**
- `Tuple[torch.Tensor, torch.Tensor]`: (极坐标向量, 缩放向量)

#### `to_string(polar_vec, scale_vec=None, indices=None)`
将极坐标向量转换为字符串表示。

**参数:**
- `polar_vec` (torch.Tensor): 极坐标向量
- `scale_vec` (torch.Tensor, 可选): 缩放向量
- `indices` (List[int], 可选): 要显示的索引

**返回:**
- `str`: 格式化的字符串

#### `to_complex(polar_vec, scale_vec)`
将极坐标向量转换为复数向量。

**参数:**
- `polar_vec` (torch.Tensor): 极坐标向量
- `scale_vec` (torch.Tensor): 缩放向量

**返回:**
- `torch.Tensor`: 复数向量

#### `complex_to_polar_tensor(complex_vec)`
复数向量转极坐标向量。

#### `polar_to_complex_tensor(polar_vec, scale_vec)`
极坐标向量转复数向量。

#### `encode_r_tensor(r)`
编码幅度值 (float32 -> int16 + scale)。

#### `decode_r_tensor(r_encoded, scale_vec)`
解码幅度值 (int16 + scale -> float32)。

#### `encode_th_tensor(th)`
编码相位值 (float32 -> int8)。

#### `decode_th_tensor(th_encoded)`
解码相位值 (int8 -> float32)。

#### `add_phase_encoded(th_encoded, delta_th)`
在编码空间直接添加相位偏移。

#### `get_info()`
获取向量类的信息。

**返回:**
- `dict`: 包含类信息的字典

### 静态方法

#### `PolarVector.validate_vector(polar_vec, scale_vec)`
验证极坐标向量的有效性。

**参数:**
- `polar_vec` (torch.Tensor): 极坐标向量
- `scale_vec` (torch.Tensor): 缩放向量

**返回:**
- `bool`: 是否有效

## 向后兼容性函数

为了保持与现有代码的兼容性，提供以下全局函数：

```python
from vector import (
    create_random_polar_state,
    polar_vec_to_string,
    polar_to_complex_tensor,
    complex_to_polar_tensor
)
```

## 示例输出

```
向量类信息: {
    'precision': 'int16 + int16',
    'amplitude_range': '[0.0, 1.0]',
    'phase_range': '[-π, π]',
    'amplitude_dtype': 'torch.int16',
    'phase_dtype': 'torch.int16',
    'device': 'cuda',
    'th_scale': 40.74366543152521
}

状态向量: 0.707+0.000j | 0.000+0.000j | 0.000+0.000j | 0.707+0.000j
```

## 性能优势

- **内存效率**: 使用 int16+int8 相比 float32+float32 节省约 75% 内存
- **计算精度**: 幅度使用16位整数编码，保持高精度
- **相位优化**: 直接在编码空间进行相位运算，避免量化误差
- **GPU加速**: 完全支持 PyTorch GPU 计算

## 精度选择指南

### int16 模式 (默认)
- **优点**: 高精度编码，适合需要精确计算的场景
- **幅度/相位**: 每分量 2 字节
- **适用**: 高精度量子模拟、科学计算

### int8 模式
- **优点**: 内存效率高，适合大规模模拟
- **幅度/相位**: 每分量 1 字节
- **精度**: 中等精度，相位 int8 时控制门误差较大
- **适用**: 大规模量子电路模拟、内存受限环境

### 如何选择
```python
# 高精度应用（默认，与 polarALL_state_3 精度相当）
vector_high_prec = PolarVector()  # 或 PolarVector(amplitude_dtype=torch.int16, phase_dtype=torch.int16)

# 内存优化：仅幅度 int8
vector_amp_opt = PolarVector(amplitude_dtype=torch.int8)

# 最小内存（相位 int8 会增大控制门误差）
vector_full_opt = PolarVector(amplitude_dtype=torch.int8, phase_dtype=torch.int8)
```

## 文件结构

- `vector.py`: 主要的 PolarVector 类实现
- `test_vector.py`: 完整的测试套件
- `test_vector_precision.py`: 精度配置测试
- `README_vector.md`: 本文档

## 相关文件

- `polarALL_state_int16.py`: 使用 PolarVector 类的量子模拟器示例