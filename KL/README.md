# KL Watermarking Transform Analysis Toolkit

这是一个用于分析水印变换(Watermarking Transform)特性的完整工具包，专注于KL散度分析和参数优化。

## 📁 目录结构

```
KL/
├── __init__.py                 # 包初始化文件
├── KL.py                       # KL散度计算核心模块
├── wm_simple.py                # WM变换核心功能
├── plot_kl_single.py           # 单向量KL散度可视化
├── plot_kl_comparison.py       # 多向量KL散度对比
├── plot_optimal_a_distribution.py  # 最优参数a分布分析
├── plot_size_migration.py      # 向量尺寸迁移分析
├── test_plot_modules.py        # 模块测试脚本
├── README_PLOT_MODULES.md      # 绘图模块详细文档
└── README.md                   # 本文档
```

## 🎯 核心概念

### 水印变换 (WM Transform)
WM变换是一个三步过程：
1. **w(x) = exp(a × (x-1))**: 指数变换
2. **f(x) = round(255 × w(x))**: 量化函数
3. **m(x) = 1 + (1/a) × log(x/255)**: 对数变换

### KL散度分析
分析输入分布X与变换后分布Y之间的KL散度：
- **KL(X||Y)**: 衡量X到Y的信息损失
- **KL(Y||X)**: 衡量Y到X的信息损失

## 📦 核心模块

### KL.py - KL散度计算核心
**功能**: 实现KL散度的计算和相关工具函数

**主要函数**:
```python
kl_divergence_manual(p_logits, q_logits)  # 手动计算KL散度
kl_divergence_categorical(p_dist, q_dist)  # 分类分布KL散度
kl_divergence_normal(p_mean, p_std, q_mean, q_std)  # 正态分布KL散度
```

**使用示例**:
```python
from KL.KL import kl_divergence_manual
import torch

# 计算两个概率分布的KL散度
p_logits = torch.randn(10, 5)
q_logits = torch.randn(10, 5)
kl_div = kl_divergence_manual(p_logits, q_logits)
print(f"KL divergence: {kl_div}")
```

### wm_simple.py - WM变换核心
**功能**: 实现WM变换的核心数学运算，不包含绘图功能

**主要函数**:
```python
w(x, a=5.0)                    # 指数变换
m(x, a=5.0)                    # 对数变换
f(x, a=5.0)                    # 量化函数
create_normalized_vector(size, seed, distribution, **kwargs)  # 生成归一化向量
analyze_transformation(X, a=5.0)  # 完整WM变换分析
```

**支持的分布类型**:
- `'uniform'`: 均匀分布
- `'exponential'`: 指数分布 (rate参数)
- `'beta'`: Beta分布 (alpha, beta参数)
- `'gamma'`: Gamma分布 (shape, rate参数)
- `'normal'`: 正态分布 (loc, scale参数)

**使用示例**:
```python
from KL.wm_simple import create_normalized_vector, analyze_transformation

# 创建测试向量
X = create_normalized_vector(size=10, distribution='uniform', seed=42)

# 分析WM变换
results = analyze_transformation(X, a=3.5)
print(f"KL(X||Y): {results['KL_X_Y']:.4f}")
print(f"KL(Y||X): {results['KL_Y_X']:.4f}")
```

## 🎨 可视化模块

### plot_kl_single.py - 单向量KL散度图表
**功能**: 为单个向量绘制KL散度随参数a变化的曲线图

**主要函数**:
```python
plot_kl_vs_a(X, a_range=None, filename="KL_A.jpg")
```

**使用示例**:
```python
from KL.plot_kl_single import plot_kl_vs_a, create_normalized_vector

X = create_normalized_vector(size=5, seed=42)
plot_kl_vs_a(X, filename="single_kl_plot.jpg")
```

### plot_kl_comparison.py - 多向量KL散度对比
**功能**: 在同一图表中比较多个向量的KL散度变化

**主要函数**:
```python
plot_multiple_kl_vs_a(X_list, labels, a_range=None, filename="KL_A_multiple.jpg", kl_type="X_Y")
```

**参数说明**:
- `kl_type`: "X_Y" (KL(X||Y)) 或 "Y_X" (KL(Y||X))

**使用示例**:
```python
from KL.plot_kl_comparison import plot_multiple_kl_vs_a, create_normalized_vector

vectors = []
labels = []
for dist in ['uniform', 'exponential']:
    X = create_normalized_vector(size=5, distribution=dist, seed=42)
    vectors.append(X)
    labels.append(dist.upper())

plot_multiple_kl_vs_a(vectors, labels, filename="comparison.jpg")
```

### plot_optimal_a_distribution.py - 最优a值分布分析
**功能**: 分析多个随机向量在不同分布下的最优a值分布

**主要函数**:
```python
analyze_optimal_a_distribution(num_vectors=50, vector_size=256,
                              a_range=None, distributions=None, filename="optimal_a_distribution.jpg")
```

**输出**: 生成散点图显示每种分布下最优a值的分布情况

**使用示例**:
```python
from KL.plot_optimal_a_distribution import analyze_optimal_a_distribution

results = analyze_optimal_a_distribution(
    num_vectors=30,
    distributions=['uniform', 'exponential', 'beta', 'gamma', 'normal'],
    filename="optimal_a_dist.jpg"
)
```

### plot_size_migration.py - 向量尺寸迁移分析
**功能**: 分析随着向量尺寸变化，最优a值如何迁移

**主要函数**:
```python
analyze_optimal_a_vs_size_multiple_distributions(sizes=None, num_vectors_per_size=5,
                                               a_range=None, filename="optimal_a_migration.jpg")
```

**输出**: 生成折线图显示不同分布在各种尺寸下的平均最优a值变化

**使用示例**:
```python
from KL.plot_size_migration import analyze_optimal_a_vs_size_multiple_distributions

results = analyze_optimal_a_vs_size_multiple_distributions(
    sizes=[10, 50, 100, 200, 500],
    num_vectors_per_size=5,
    filename="size_migration.jpg"
)
```

## 🧪 测试与验证

### test_plot_modules.py - 模块测试脚本
**功能**: 验证所有模块是否正常工作

**运行方法**:
```bash
cd /path/to/KL
python test_plot_modules.py
# 或
python -m KL.test_plot_modules
```

**测试内容**:
- 核心模块功能验证
- 所有绘图模块导入测试
- GPU可用性检查

## ⚙️ 配置与依赖

### 系统要求
- Python 3.7+
- PyTorch 1.7+
- NumPy (随PyTorch安装)
- Matplotlib 3.0+
- SciPy (可选，用于密度估计)

### GPU支持
所有模块自动检测GPU：
- 有GPU时自动使用CUDA加速
- 无GPU时自动回退到CPU计算
- 支持多GPU环境

### 性能优化
- 张量操作自动在GPU上执行
- 内存管理优化
- 批量计算支持

## 📊 输出格式

### 图表文件
所有可视化函数生成高分辨率JPG文件：
- 分辨率: 300 DPI
- 格式: RGB色彩
- 大小: 200KB - 500KB (取决于复杂度)

### 数据格式
- 输入向量: PyTorch张量 (自动GPU迁移)
- 输出结果: Python字典
- KL散度: float类型
- 统计数据: dict格式

## 🔧 高级用法

### 自定义分布参数
```python
from KL.wm_simple import create_normalized_vector

# 自定义Beta分布参数
X = create_normalized_vector(size=100, distribution='beta',
                           alpha=2.5, beta=7.8, seed=42)

# 自定义Gamma分布参数
X = create_normalized_vector(size=100, distribution='gamma',
                           shape=3.0, rate=0.5, seed=42)
```

### 批量分析
```python
import torch
from KL.wm_simple import analyze_transformation

# 批量分析不同a值
X = create_normalized_vector(size=50, seed=42)
a_values = torch.linspace(2, 5, 30)

results = []
for a in a_values:
    result = analyze_transformation(X, a.item())
    results.append(result)

# 分析KL散度变化趋势
kl_values = [r['KL_X_Y'] for r in results]
```

### 内存优化
```python
# 对于大向量，考虑分批处理
large_X = create_normalized_vector(size=10000, seed=42)

# 分块分析
chunk_size = 1000
for i in range(0, len(large_X), chunk_size):
    chunk = large_X[i:i+chunk_size]
    result = analyze_transformation(chunk, a=3.5)
    # 处理结果...
```

## 🚨 注意事项

### 数值稳定性
- 参数a值不建议过小（< 0.1）或过大（> 10）
- 输入向量必须是概率分布（和为1，非负）
- 大向量可能需要更多GPU内存

### 性能考虑
- GPU计算比CPU快10-100倍
- 大批量分析建议使用GPU
- 监控GPU内存使用情况

### 错误处理
- 自动检测NaN/inf值
- 提供详细错误信息
- 优雅降级到CPU计算

## 📈 应用场景

1. **参数优化**: 找到最佳的WM变换参数a
2. **分布分析**: 比较不同输入分布的变换特性
3. **尺寸影响**: 理解向量尺寸对变换效果的影响
4. **鲁棒性测试**: 评估变换在不同输入下的稳定性

## 🤝 贡献

欢迎提交问题和改进建议！

## 📄 许可证

本项目采用MIT许可证。