"""
最大值归一化水印变换(WM Transform)分析工具

此模块实现了基于向量最大值的归一化WM变换，与原始方法相比具有更好的数值稳定性和适应性。

核心变化:
- w(x) = exp(a * (x - X_max))  # 相对于向量最大值进行指数变换
- m(x) = X_max + (1/a) * log(x/255)  # 基于最大值的对数逆变换

主要功能:
- analyze_transformation_max_normalized(): 完整的最大值归一化WM变换分析
- benchmark_max_normalized_vs_original(): 与原始方法的性能对比
- plot_max_normalized_comparison(): 生成对比图表
"""

import torch
import torch.nn as nn
import math
import matplotlib.pyplot as plt
import sys
import os

# GPU设备设置
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"CUDA version: {torch.version.cuda}")
else:
    print("CUDA not available, using CPU")

# 导入相关模块
import sys
import os
# 添加父目录到路径
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, parent_dir)

try:
    from KL.KL import kl_divergence_manual
    from KL.wm_simple import create_normalized_vector
except ImportError:
    # 如果相对导入失败，尝试绝对导入
    sys.path.insert(0, os.path.dirname(__file__))
    from KL import kl_divergence_manual
    from wm_simple import create_normalized_vector


def w_max_normalized(x, a=5.0, X_max=None):
    """
    基于最大值归一化的指数变换
    w(x) = e^(a*(x-X_max))

    参数:
        x: 输入值或张量
        a: 参数，默认值为5.0
        X_max: 预计算的最大值，如果为None则自动计算
    返回:
        计算结果
    """
    if X_max is None:
        X_max = torch.max(x)
    return torch.exp(a * (x - X_max))


def m_max_normalized(x, a=5.0, X_max=1.0):
    """
    基于最大值归一化的对数变换
    m(x) = X_max + (1/a)*ln(x/255)

    参数:
        x: 输入值或张量
        a: 参数，默认值为5.0
        X_max: 对应的最大值
    返回:
        计算结果
    """
    return X_max + (1/a) * torch.log(x / 65535)


def f(x, a=5.0):
    """
    计算f(x) = round(255*w(x))

    参数:
        x: 输入值或张量
        a: 参数，默认值为5.0
    返回:
        计算结果，四舍五入到最接近的整数
    """
    return torch.round(65534 * x) + 1


def analyze_transformation_max_normalized(X, a=5.0):
    """
    使用最大值归一化的完整WM变换分析

    参数:
        X: 输入向量
        a: 参数
    返回:
        包含所有中间结果的字典
    """
    if not isinstance(X, torch.Tensor):
        X = torch.tensor(X, dtype=torch.float32, device=device)
    else:
        X = X.to(device)

    # 找出最大值
    X_max = torch.max(X)

    results = {
        'X': X,
        'X_max': X_max.item(),
        'X_sum': X.sum().item(),
        'a': a
    }

    # 计算中间步骤
    w_X = w_max_normalized(X, a, X_max)
    results['w_X'] = w_X

    f_w_X = f(w_X, a)
    results['f_w_X'] = f_w_X

    Y = m_max_normalized(f_w_X, a, X_max)
    results['Y'] = Y
    results['Y_sum'] = Y.sum().item()

    # 计算KL散度
    kl_X_Y, kl_Y_X = compare_kl_divergence(X, Y)
    results['KL_X_Y'] = kl_X_Y
    results['KL_Y_X'] = kl_Y_X

    return results


def compare_kl_divergence(X, Y):
    """
    比较X与Y的KL散度

    参数:
        X: 第一个分布
        Y: 第二个分布
    返回:
        KL(X||Y) 和 KL(Y||X)
    """
    # 确保输入是tensor且归一化
    if not isinstance(X, torch.Tensor):
        X = torch.tensor(X, dtype=torch.float32, device=device)
    else:
        X = X.to(device)
    if not isinstance(Y, torch.Tensor):
        Y = torch.tensor(Y, dtype=torch.float32, device=device)
    else:
        Y = Y.to(device)

    # 将向量转换为logits格式（KL函数需要）
    # 由于KL函数内部会使用softmax，我们需要先转换为合适的格式
    X_logits = torch.log(X + 1e-10)  # 避免log(0)
    Y_logits = torch.log(Y + 1e-10)

    kl_X_Y = kl_divergence_manual(X_logits.unsqueeze(0), Y_logits.unsqueeze(0))
    kl_Y_X = kl_divergence_manual(Y_logits.unsqueeze(0), X_logits.unsqueeze(0))

    return kl_X_Y.item(), kl_Y_X.item()


def benchmark_max_normalized_vs_original(sizes=None, distributions=None, a_values=None):
    """
    对比最大值归一化与原始方法的性能

    参数:
        sizes: 向量长度列表
        distributions: 分布类型列表
        a_values: a值列表
    返回:
        对比结果字典
    """
    if sizes is None:
        sizes = [10, 50, 100]

    if distributions is None:
        distributions = ['uniform', 'exponential', 'beta']

    if a_values is None:
        a_values = [1.0, 3.0, 5.0]

    print("Benchmarking max-normalized vs original WM transform...")
    print(f"Sizes: {sizes}")
    print(f"Distributions: {distributions}")
    print(f"A values: {a_values}")

    results = {
        'max_normalized': {},
        'original': {}
    }

    # 导入原始方法
    try:
        from wm_simple import analyze_transformation as analyze_original
    except ImportError:
        import sys
        import os
        sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        from KL.wm_simple import analyze_transformation as analyze_original

    for size in sizes:
        print(f"\n--- Size {size} ---")
        for dist in distributions:
            print(f"  Distribution: {dist}")

            # 为每种方法收集结果
            max_norm_results = []
            original_results = []

            for a in a_values:
                # 生成测试向量
                X = create_normalized_vector(size=size, seed=42, distribution=dist)

                # 最大值归一化方法
                try:
                    result_max = analyze_transformation_max_normalized(X, a)
                    max_norm_results.append({
                        'a': a,
                        'KL_X_Y': result_max['KL_X_Y'],
                        'KL_Y_X': result_max['KL_Y_X']
                    })
                except Exception as e:
                    print(f"    Max-normalized failed for a={a}: {e}")
                    max_norm_results.append({'a': a, 'error': str(e)})

                # 原始方法
                try:
                    result_orig = analyze_original(X, a)
                    original_results.append({
                        'a': a,
                        'KL_X_Y': result_orig['KL_X_Y'],
                        'KL_Y_X': result_orig['KL_Y_X']
                    })
                except Exception as e:
                    print(f"    Original failed for a={a}: {e}")
                    original_results.append({'a': a, 'error': str(e)})

            results['max_normalized'][f'{dist}_{size}'] = max_norm_results
            results['original'][f'{dist}_{size}'] = original_results

            # 打印比较结果
            print(f"    Max-normalized vs Original comparison:")
            for i, a in enumerate(a_values):
                max_kl = max_norm_results[i].get('KL_X_Y', 'ERROR')
                orig_kl = original_results[i].get('KL_X_Y', 'ERROR')

    return results


def plot_max_normalized_comparison(filename="max_normalized_comparison.jpg"):
    """
    绘制最大值归一化与原始方法的对比图表
    """
    print("Generating comparison plot...")

    # 运行基准测试
    benchmark_results = benchmark_max_normalized_vs_original(
        sizes=[20, 50],
        distributions=['uniform', 'exponential'],
        a_values=[1.0, 2.0, 3.0, 4.0, 5.0]
    )

    # 准备绘图数据
    methods = ['max_normalized', 'original']
    distributions = ['uniform', 'exponential']
    sizes = [20, 50]
    a_values = [1.0, 2.0, 3.0, 4.0, 5.0]

    # 创建子图
    fig, axes = plt.subplots(len(distributions), len(sizes),
                            figsize=(5*len(sizes), 4*len(distributions)))
    if len(distributions) == 1 and len(sizes) == 1:
        axes = [axes]
    elif len(distributions) == 1:
        axes = axes.reshape(1, -1)
    elif len(sizes) == 1:
        axes = axes.reshape(-1, 1)

    colors = ['blue', 'red']

    for i, dist in enumerate(distributions):
        for j, size in enumerate(sizes):
            ax = axes[i, j]

            for k, method in enumerate(methods):
                key = f'{dist}_{size}'
                if key in benchmark_results[method]:
                    results = benchmark_results[method][key]
                    valid_a = []
                    valid_kl = []

                    for result in results:
                        if 'KL_X_Y' in result and not (torch.isnan(torch.tensor(result['KL_X_Y'])) or torch.isinf(torch.tensor(result['KL_X_Y']))):
                            valid_a.append(result['a'])
                            valid_kl.append(result['KL_X_Y'])

                    if valid_a and valid_kl:
                        ax.plot(valid_a, valid_kl, 'o-', color=colors[k],
                               linewidth=2, markersize=6, label=method.replace('_', ' ').title())

            ax.set_xlabel('Parameter a')
            ax.set_ylabel('KL(X||Y)')
            ax.set_title(f'{dist.title()} Distribution\nSize {size}')
            ax.grid(True, alpha=0.3)
            ax.legend()

    plt.tight_layout()
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    print(f"Comparison plot saved as: {filename}")
    plt.close()


# 示例用法
if __name__ == "__main__":
    print("Max-Normalized WM Transform Analysis")
    print("=" * 50)

    # 测试基本功能
    print("Testing basic functionality...")
    X = create_normalized_vector(size=10, seed=42, distribution='uniform')
    print(f"Test vector: {X}")

    result = analyze_transformation_max_normalized(X, a=3.0)
    print(f"X_max: {result['X_max']:.4f}")
    print(f"KL(X||Y): {result['KL_X_Y']:.4f}")
    print(f"KL(Y||X): {result['KL_Y_X']:.4f}")

    # 生成对比图表
    print("\nGenerating comparison plot...")
    plot_max_normalized_comparison()

    # 运行基准测试
    print("\nRunning benchmark...")
    benchmark_results = benchmark_max_normalized_vs_original()

def analyze_optimal_a_distribution_max_normalized(num_vectors=50, vector_size=256, a_range=None,
                                               distributions=None, filename="optimal_a_distribution_max_normalized.jpg"):
    """
    统计多个随机向量在最大值归一化变换下最佳a值的分布 - 多分布对比

    参数:
        num_vectors: 每个分布的随机向量数量
        vector_size: 向量长度
        a_range: a值范围
        distributions: 分布类型列表，默认使用所有支持的分布
        filename: 保存的文件名
    """
    import random

    if distributions is None:
        distributions = ['uniform', 'exponential', 'beta', 'gamma', 'normal']

    print(f"Analyzing optimal a distribution (max-normalized) for {num_vectors} vectors per distribution...")
    print(f"Distributions: {distributions}")

    if a_range is None:
        a_range = torch.arange(2, 4.4, 0.01, device=device)  # 用户指定的a范围

    # 创建子图
    n_distributions = len(distributions)
    n_cols = min(3, n_distributions)
    n_rows = (n_distributions + n_cols - 1) // n_cols

    plt.switch_backend('Agg')
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(6*n_cols, 5*n_rows))
    if n_rows == 1 and n_cols == 1:
        axes = [axes]
    elif n_rows == 1:
        axes = axes.flatten()
    else:
        axes = axes.flatten()

    all_optimal_values = {}

    for idx, distribution in enumerate(distributions):
        print(f"\n--- Analyzing {distribution.upper()} distribution (max-normalized) ---")

        optimal_a_values = []

        for i in range(num_vectors):
            if i % 10 == 0 and i > 0:
                print(f"  Processing vector {i+1}/{num_vectors}...")

            # 使用不同的种子生成随机向量
            seed = random.randint(0, 100000)

            X = create_normalized_vector(size=vector_size, seed=seed, distribution=distribution)

            # 找到这个向量对应的最小KL散度a值
            min_kl = float('inf')
            optimal_a = None

            for a in a_range:
                try:
                    results = analyze_transformation_max_normalized(X, a.item())
                    kl_xy = results['KL_X_Y']

                    if not (torch.isnan(torch.tensor(kl_xy)) or torch.isinf(torch.tensor(kl_xy))):
                        if kl_xy < min_kl:
                            min_kl = kl_xy
                            optimal_a = a.item()
                except Exception:
                    continue

            if optimal_a is not None:
                optimal_a_values.append(optimal_a)

        all_optimal_values[distribution] = optimal_a_values

        if not optimal_a_values:
            print(f"No valid optimal a values found for {distribution}")
            continue

        print(f"Found {len(optimal_a_values)} valid optimal a values for {distribution}")

        # 在对应的子图中绘制
        ax = axes[idx]

        # 只绘制落点，不显示统计信息
        if optimal_a_values:
            # 创建一个简单的落点图 - 所有点都在一条水平线上
            y_positions = [0.5] * len(optimal_a_values)  # 所有点在y=0.5的位置
            ax.scatter(optimal_a_values, y_positions, alpha=0.7, s=20, color='red',
                      edgecolors='none', linewidth=0)

            # 设置子图属性
            ax.set_xlabel('Optimal Parameter a', fontsize=12)
            ax.set_ylabel('', fontsize=12)  # 不显示y轴标签
            ax.set_title(f'{distribution.upper()}\n(n={len(optimal_a_values)})', fontsize=14, pad=10)

            # 设置x轴范围为a_range的范围
            ax.set_xlim(a_range.min().item(), a_range.max().item())

            # 移除y轴刻度和标签
            ax.set_yticks([])
            ax.set_yticklabels([])

            # 添加网格
            ax.grid(True, alpha=0.3, axis='x')

    # 隐藏多余的子图
    for idx in range(n_distributions, len(axes)):
        axes[idx].set_visible(False)

    plt.suptitle(f'Max-Normalized WM Transform: Optimal a Distribution\n(Vector Size: {vector_size})',
                fontsize=16, y=0.95)
    plt.tight_layout()
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    print(f"Max-normalized optimal a distribution chart saved as: {filename}")

    plt.close()

    return all_optimal_values


def analyze_optimal_a_vs_size_max_normalized(vector_sizes=None, num_vectors_per_size=20,
                                           distributions=None, a_range=None,
                                           filename="optimal_a_vs_size_max_normalized.jpg"):
    """
    分析不同向量长度下最大值归一化变换的最优a值迁移过程

    参数:
        vector_sizes: 向量长度列表
        num_vectors_per_size: 每个长度使用多少个随机向量
        distributions: 分布类型列表
        a_range: a值范围
        filename: 保存的文件名
    """
    import random

    if vector_sizes is None:
        vector_sizes = [10, 50, 100, 200, 500]

    if distributions is None:
        distributions = ['uniform', 'exponential', 'beta']

    print(f"Analyzing optimal a vs vector size (max-normalized)...")
    print(f"Vector sizes: {vector_sizes}")
    print(f"Distributions: {distributions}")

    if a_range is None:
        a_range = torch.arange(2, 4.4, 0.01, device=device)

    plt.switch_backend('Agg')
    fig, ax = plt.subplots(1, 1, figsize=(12, 8))

    colors = ['red', 'blue', 'green', 'orange', 'purple']

    for idx, distribution in enumerate(distributions):
        print(f"\n--- Analyzing {distribution.upper()} distribution (max-normalized) ---")

        size_optimal_a = []

        for size in vector_sizes:
            print(f"  Vector size: {size}")

            optimal_a_values = []

            for i in range(num_vectors_per_size):
                seed = random.randint(0, 100000)
                X = create_normalized_vector(size=size, seed=seed, distribution=distribution)

                # 找到最小KL的a值
                min_kl = float('inf')
                optimal_a = None

                for a in a_range:
                    try:
                        results = analyze_transformation_max_normalized(X, a.item())
                        kl_xy = results['KL_X_Y']

                        if not (torch.isnan(torch.tensor(kl_xy)) or torch.isinf(torch.tensor(kl_xy))):
                            if kl_xy < min_kl:
                                min_kl = kl_xy
                                optimal_a = a.item()
                    except Exception:
                        continue

                if optimal_a is not None:
                    optimal_a_values.append(optimal_a)

            if optimal_a_values:
                avg_optimal_a = sum(optimal_a_values) / len(optimal_a_values)
                size_optimal_a.append(avg_optimal_a)
            else:
                size_optimal_a.append(None)

        # 绘制这个分布的曲线
        valid_sizes = [s for s, a in zip(vector_sizes, size_optimal_a) if a is not None]
        valid_a_values = [a for a in size_optimal_a if a is not None]

        if valid_sizes and valid_a_values:
            ax.plot(valid_sizes, valid_a_values, 'o-', color=colors[idx % len(colors)],
                   linewidth=2, markersize=8, label=f'{distribution.title()} (Max-Normalized)',
                   alpha=0.8)

    ax.set_xlabel('Vector Size', fontsize=14)
    ax.set_ylabel('Average Optimal Parameter a', fontsize=14)
    ax.set_title('Max-Normalized WM Transform: Optimal a vs Vector Size', fontsize=16, pad=20)
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=12)

    # 设置x轴为对数刻度，因为向量长度变化很大
    ax.set_xscale('log')

    plt.tight_layout()
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    print(f"Max-normalized optimal a vs size migration chart saved as: {filename}")

    plt.close()


# 示例用法
if __name__ == "__main__":
    print("Max-Normalized WM Transform Analysis")
    print("=" * 50)

    # 测试基本功能
    print("Testing basic functionality...")
    X = create_normalized_vector(size=10, seed=42, distribution='uniform')
    print(f'Test vector: {X}')

    result = analyze_transformation_max_normalized(X, a=3.0)
    print(f'X_max: {result["X_max"]:.4f}')
    print(f'KL(X||Y): {result["KL_X_Y"]:.4f}')
    print(f'KL(Y||X): {result["KL_Y_X"]:.4f}')

    # 生成对比图表
    print("\nGenerating comparison plot...")
    plot_max_normalized_comparison()

    # 运行基准测试
    print("\nRunning benchmark...")
    benchmark_results = benchmark_max_normalized_vs_original()

    # 生成最优a值分布图
    print("\nGenerating optimal a distribution plot...")
    optimal_values = analyze_optimal_a_distribution_max_normalized(
        num_vectors=20,  # 减少向量数量以加快测试
        vector_size=50
    )

    # 生成最优a值vs向量大小的迁移图
    print("\nGenerating optimal a vs size migration plot...")
    analyze_optimal_a_vs_size_max_normalized(
        vector_sizes=[10, 20, 50],
        num_vectors_per_size=10
    )

    print("\nMax-normalized WM transform analysis completed!")
