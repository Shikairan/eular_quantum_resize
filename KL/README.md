# KL Watermarking Transform Analysis Toolkit

这是一个用于分析水印变换(Watermarking Transform)特性的完整工具包，专注于KL散度分析和参数优化。

## 📁 目录结构

```
KL/
├── __init__.py                 # 包初始化文件
├── KL.py                       # KL散度计算核心模块
├── wm_simple.py                # WM变换核心功能
├── wm_max_normalized.py        # 最大值归一化WM变换
├── plot_kl_single.py           # 单向量KL散度可视化
├── plot_kl_comparison.py       # 多向量KL散度对比
├── plot_optimal_a_distribution.py  # 最优参数a分布分析
├── plot_size_migration.py      # 向量尺寸迁移分析
├── test_plot_modules.py        # 模块测试脚本
├── test_max_normalized.py     # 最大值归一化测试脚本
├── demo_max_normalized.py    # 最大值归一化演示脚本
├── generate_optimal_a_plots.py # 最优a值分析图表生成脚本
├── test_method_comparison.py   # 方法对比功能测试脚本
├── demo_method_comparison.py  # 方法对比功能演示脚本
├── KL-3D.py                  # 3D KL散度曲面图分析
├── quick_3d_demo.py         # 快速3D分析演示脚本
├── quick_optimal_a_demo.py    # 快速最优a值演示脚本
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

### wm_max_normalized.py - 最大值归一化WM变换
**功能**: 实现基于向量最大值的归一化WM变换，提供更好的数值稳定性和不同的变换特性

**核心变化**:
- **w(x) = exp(a × (x - X_max))**: 相对于向量最大值进行指数变换
- **m(x) = X_max + (1/a) × log(x/255)**: 基于最大值的对数逆变换

**主要函数**:
```python
w_max_normalized(x, a=5.0, X_max=None)    # 最大值归一化指数变换
m_max_normalized(x, a=5.0, X_max=1.0)    # 最大值归一化对数变换
analyze_transformation_max_normalized(X, a=5.0)  # 完整最大值归一化分析
benchmark_max_normalized_vs_original(...)        # 与原始方法对比
plot_max_normalized_comparison(filename)         # 生成对比图表
analyze_optimal_a_distribution_max_normalized(...)  # 最优a值分布分析
analyze_optimal_a_vs_size_max_normalized(...)       # 最优a值vs向量大小迁移分析
```

**优势**:
- 相对于向量中的实际最大值进行变换，更符合实际分布特征
- 可能提供更好的数值稳定性和变换效果
- 自动适应不同范围的输入向量

**使用示例**:
```python
from KL.wm_max_normalized import analyze_transformation_max_normalized

# 创建测试向量
X = torch.tensor([0.1, 0.3, 0.5, 0.8, 0.9])

# 使用最大值归一化变换
results = analyze_transformation_max_normalized(X, a=3.0)
print(f"X_max: {results['X_max']:.4f}")
print(f"KL(X||Y): {results['KL_X_Y']:.4f}")

# 与原始方法对比
from KL.wm_max_normalized import benchmark_max_normalized_vs_original
results = benchmark_max_normalized_vs_original()
```

**对比分析**:
```python
# 生成两种方法的效果对比图
from KL.wm_max_normalized import plot_max_normalized_comparison
plot_max_normalized_comparison("method_comparison.jpg")
```

**最优参数分析**:
```python
# 分析不同分布下最优a值的分布
from KL.wm_max_normalized import analyze_optimal_a_distribution_max_normalized
optimal_values = analyze_optimal_a_distribution_max_normalized(
    num_vectors=50,      # 每个分布的向量数量
    vector_size=100,     # 向量长度
    distributions=['uniform', 'exponential', 'beta']
)

# 分析不同向量长度下最优a值的迁移
from KL.wm_max_normalized import analyze_optimal_a_vs_size_max_normalized
analyze_optimal_a_vs_size_max_normalized(
    vector_sizes=[10, 50, 100, 200],
    num_vectors_per_size=20,
    distributions=['uniform', 'exponential']
)
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

### test_max_normalized.py - 最大值归一化测试脚本
**功能**: 全面测试最大值归一化WM变换的功能

**运行方法**:
```bash
cd /path/to/KL
python test_max_normalized.py
```

**测试内容**:
- 基本变换功能测试（w_max_normalized, m_max_normalized）
- 与原始方法的性能对比
- 不同概率分布的兼容性测试
- 边界情况和异常处理测试

### demo_max_normalized.py - 最大值归一化演示脚本
**功能**: 交互式演示最大值归一化WM变换的各种功能和用法

**运行方法**:
```bash
cd /path/to/KL
python demo_max_normalized.py
```

**演示内容**:
- 基本使用方法：创建向量并应用变换
- 方法对比：与原始WM变换的性能对比
- 多场景应用：不同大小和分布的向量
- 参数扫描：探索不同a值对KL散度的影响

**输出示例**:
```
🔬 Max-Normalized WM Transform Demo
==================================================
1. Creating test vector...
   Input vector X: [0.168 0.003 0.109 ...]

2. Applying max-normalized WM transform (a=3.0)...
   X_max: 0.2018
   KL(X||Y): 0.000162

3. Comparing with original method...
   Results summary:
   Distribution | Size | Method | Avg KL(X||Y)
   ------------|------|--------|------------
   uniform     | 15   | max_normalized | 0.0003
   uniform     | 15   | original       | 0.0005

🔍 Optimal a Value Analysis Demo
===================================
Generating optimal a distribution plot...
Analyzing optimal a distribution (max-normalized) for 10 vectors per distribution...
...
Optimal a value statistics:
uniform      avg=3.25, min=2.75, max=3.72
exponential  avg=2.98, min=2.45, max=3.51
beta         avg=3.12, min=2.62, max=3.68
```

### generate_optimal_a_plots.py - 最优a值分析图表生成脚本
**功能**: 专门生成最大值归一化WM变换的最优a值分析图表

**运行方法**:
```bash
cd /path/to/KL
python generate_optimal_a_plots.py
```

**生成图表**:
- `optimal_a_distribution_max_normalized.jpg`: 不同分布下最优a值的分布
- `optimal_a_vs_size_max_normalized.jpg`: 不同向量长度下最优a值的迁移趋势
- `method_comparison_optimal_a.jpg`: Max-Normalized与原生方法的最优a值对比（x轴为log2尺度）

**可调参数**:
```python
# 在脚本中修改这些参数
num_vectors = 50          # 每个分布的向量数量
vector_size = 100         # 向量长度
vector_sizes = [16, 32, 64, 128, 256, 512]  # 向量长度列表（2的幂）
distributions = ['uniform', 'exponential', 'beta', 'gamma', 'normal']
```

**新增功能说明**:
**方法对比图表** (`method_comparison_optimal_a.jpg`):
- 同时显示Max-Normalized和原生WM变换在不同向量尺度上的最优a值
- x轴使用log2尺度，更好地展示尺度效应
- 每个数据点显示具体的平均最优a值
- 支持多种概率分布的对比分析

**性能提示**:
- 增加`num_vectors`可以获得更准确的统计结果，但会增加计算时间
- 减小`vector_size`可以加快计算速度，但可能影响结果准确性
- 可以注释掉不需要的分布来加快处理速度
- 方法对比功能计算量较大，建议先用测试脚本验证

### quick_optimal_a_demo.py - 快速最优a值演示脚本
**功能**: 快速演示最大值归一化WM变换的最优a值分析，使用较少的计算量

**运行方法**:
```bash
cd /path/to/KL
python quick_optimal_a_demo.py
```

**特点**:
- 使用较少的向量数量（15个）和较小的向量长度（50）
- 快速生成结果，适合演示和测试
- 生成的文件带有"quick_"前缀以区别于完整版本

**输出文件**:
- `quick_optimal_a_distribution_max_normalized.jpg`
- `quick_optimal_a_vs_size_max_normalized.jpg`

**适用场景**:
- 快速验证功能是否正常工作
- 演示最优a值分析的概念
- 在完整分析前进行参数调优

### test_method_comparison.py - 方法对比功能测试脚本
**功能**: 快速测试Max-Normalized与原生方法的最优a值对比功能

**运行方法**:
```bash
cd /path/to/KL
python test_method_comparison.py
```

**特点**:
- 使用较少的向量数量和大小，快速生成结果
- 专门用于测试新增的对比功能
- 生成`test_method_comparison.jpg`图表用于验证

**适用场景**:
- 验证方法对比功能是否正常工作
- 快速预览对比效果
- 调试参数设置

### demo_method_comparison.py - 方法对比功能演示脚本
**功能**: 演示Max-Normalized与原生方法在不同向量尺度上的最优a值对比

**运行方法**:
```bash
cd /path/to/KL
python demo_method_comparison.py
```

**特点**:
- 使用精心设计的参数以获得最佳演示效果
- x轴使用log2尺度标注（2^4, 2^5, 2^6, 2^7等）
- 每个数据点显示具体的平均最优a值
- 包含详细的说明和总结信息

**输出示例**:
```
🎯 Method Comparison Demo: Max-Normalized vs Original
============================================================
Demo Parameters:
  - Vector sizes: [16, 32, 64, 128] (powers of 2)
  - Vectors per size: 8
  - Distribution: uniform

Analyzing uniform distribution across different vector sizes...
  Vector size 2^4: 3.250 | 3.125
  Vector size 2^5: 3.188 | 3.063
  ...
```

**适用场景**:
- 演示方法对比功能的效果
- 学术报告或演示的视觉材料
- 快速了解两种方法在不同尺度下的表现差异

### KL-3D.py - 3D KL散度曲面图分析
**功能**: 生成3D可视化图表，展示Max-Normalized WM变换在不同向量长度和a值参数下的完整KL散度分布

**生成图表**:
- **`KL-3D_surface.jpg`**: 3D曲面图
  - X轴：向量长度（以2为底的指数尺度）
  - Y轴：遍历的所有a值（参数空间扫描）
  - Z轴：对应长度和a值的直接KL散度数值
- **`KL-3D_contour.jpg`**: 2D等高线图

**运行方法**:
```bash
cd /path/to/KL
python KL-3D.py
```

**特点**:
- **完整参数扫描**: 显示所有向量长度和a值组合的KL散度，而非仅最优值
- **3D参数空间可视化**: 直观展示参数空间中的KL散度分布特征
- **等高线补充**: 提供2D投影视图，便于分析KL散度的变化模式
- **Log2尺度**: X轴使用对数尺度，更好地展示向量大小的尺度效应
- **固定种子**: 使用可重现的随机种子，确保结果的一致性

**参数配置**:
```python
vector_sizes = [2**i for i in range(4, 9)]  # [16, 32, 64, 128, 256]
a_values = torch.arange(2.0, 4.5, 0.1).numpy()  # 完整的a值范围扫描
# 每个参数组合直接计算KL散度，无需平均
```

**适用场景**:
- **参数空间探索**: 全面了解KL散度在参数空间中的分布特征
- **最优参数识别**: 通过3D曲面识别全局最优参数组合
- **尺度效应分析**: 观察向量长度如何影响KL散度曲面形态
- **学术研究**: 生成用于论文展示的高质量3D参数空间可视化
- **算法调优**: 识别参数敏感区域和最优参数区间

### quick_3d_demo.py - 快速3D分析演示脚本
**功能**: 快速演示3D KL散度分析，使用较少的计算量以加快速度

**运行方法**:
```bash
cd /path/to/KL
python quick_3d_demo.py
```

**特点**:
- **快速预览**: 使用较少的向量大小、a值范围和样本数
- **完整3D视图**: 仍然包含完整的3D曲面图和统计信息
- **参数优化**: 适合快速测试和参数调优

**输出文件**:
- `quick_3d_demo.jpg`: 快速3D曲面图

**适用场景**:
- 快速验证3D分析功能
- 参数预调优
- 演示基本概念
- 在完整分析前进行快速检查

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