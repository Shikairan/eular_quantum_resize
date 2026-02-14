import torch
import torch.nn as nn
import math
import matplotlib.pyplot as plt
import sys
import os
import random
# 导入KL模块
from KL import kl_divergence_manual

# GPU设备设置
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"CUDA version: {torch.version.cuda}")
else:
    print("CUDA not available, using CPU")

def w(x, a=5.0):
    """
    计算w(x) = e^(a*(x-1))

    参数:
        x: 输入值或张量
        a: 参数，默认值为5.0
    返回:
        计算结果
    """
    return torch.exp(a * (x - 1))


def m(x, a=5.0):
    """
    计算m(x) = 1 + (1/a)*ln(x/255)

    参数:
        x: 输入值或张量
        a: 参数，默认值为5.0
    返回:
        计算结果
    """
    return 1 + (1/a) * torch.log(x / 65535.0)


def f(x, a=5.0):
    """
    计算f(x) = round(255*w(x))

    参数:
        x: 输入值或张量
        a: 参数，默认值为5.0
    返回:
        计算结果，四舍五入到最接近的整数
    """
    return torch.round(65535 * w(x, a))+1


def create_normalized_vector(size=10, seed=42, distribution='uniform', **kwargs):
    """
    创建一个非负浮点向量，向量和为整数1

    参数:
        size: 向量长度
        seed: 随机种子
        distribution: 分布类型 ('uniform', 'exponential', 'beta', 'gamma', 'normal')
        **kwargs: 分布参数
            - exponential: rate (默认1.0)
            - beta: alpha, beta (默认1.0, 1.0)
            - gamma: shape, rate (默认2.0, 1.0)
            - normal: 首先取绝对值然后归一化，loc, scale (默认0.0, 1.0)
    返回:
        归一化后的非负向量
    """
    torch.manual_seed(seed)

    if distribution == 'uniform':
        # 均匀分布
        x = torch.rand(size, device=device)
    elif distribution == 'exponential':
        # 指数分布
        rate = kwargs.get('rate', 1.0)
        x = torch.distributions.Exponential(rate).sample((size,)).to(device)
    elif distribution == 'beta':
        # Beta分布
        alpha = kwargs.get('alpha', 1.0)
        beta = kwargs.get('beta', 1.0)
        x = torch.distributions.Beta(alpha, beta).sample((size,)).to(device)
    elif distribution == 'gamma':
        # Gamma分布
        shape = kwargs.get('shape', 2.0)
        rate = kwargs.get('rate', 1.0)
        x = torch.distributions.Gamma(shape, rate).sample((size,)).to(device)
    elif distribution == 'normal':
        # 正态分布（取绝对值）
        loc = kwargs.get('loc', 0.0)
        scale = kwargs.get('scale', 1.0)
        x = torch.abs(torch.distributions.Normal(loc, scale).sample((size,)).to(device))
    else:
        raise ValueError(f"不支持的分布类型: {distribution}")

    # 归一化使其和为1
    x = x / x.sum()
    return x


def compute_Y_from_X(X, a=5.0):
    """
    计算Y = m(f(w(X)))

    参数:
        X: 输入向量
        a: 参数，默认值为5.0
    返回:
        计算结果Y
    """
    # 确保X是tensor
    if not isinstance(X, torch.Tensor):
        X = torch.tensor(X, dtype=torch.float32)

    # 计算Y = m(f(w(X)))
    w_X = w(X, a)
    f_w_X = f(w_X, a)
    Y = m(f_w_X, a)
    return Y


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
    X_logits = torch.log(X )  # 避免log(0)
    Y_logits = torch.log(Y )

    kl_X_Y = kl_divergence_manual(X_logits.unsqueeze(0), Y_logits.unsqueeze(0))
    kl_Y_X = kl_divergence_manual(Y_logits.unsqueeze(0), X_logits.unsqueeze(0))

    return kl_X_Y.item(), kl_Y_X.item()


def analyze_transformation(X, a=5.0):
    """
    完整分析X到Y的变换过程

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

    results = {
        'X': X,
        'X_sum': X.sum().item(),
        'a': a
    }

    # 计算中间步骤
    w_X = w(X, a)
    results['w_X'] = w_X

    f_w_X = f(w_X, a)
    results['f_w_X'] = f_w_X

    Y = m(f_w_X, a)
    results['Y'] = Y
    results['Y_sum'] = Y.sum().item()

    # 计算KL散度
    kl_X_Y, kl_Y_X = compare_kl_divergence(X, Y)
    results['KL_X_Y'] = kl_X_Y
    results['KL_Y_X'] = kl_Y_X

    return results


def plot_kl_vs_a(X, a_range=None, filename="KL_A.jpg"):
    """
    绘制KL散度随参数a变化的图表

    参数:
        X: 输入向量
        a_range: a值的范围，默认0.1到15.0，步长0.2
        filename: 保存的文件名
    """
    print("开始执行plot_kl_vs_a函数")
    if a_range is None:
        a_range = torch.arange(2, 4.4, 0.01, device=device)  # 用户指定的a范围
    else:
        a_range = torch.tensor(a_range, dtype=torch.float32, device=device)

    kl_X_Y_values = []
    kl_Y_X_values = []
    valid_a_values = []

    print(f"Calculating KL divergence for a from {a_range.min().item():.1f} to {a_range.max().item():.1f}...")

    for a in a_range:
        try:
            results = analyze_transformation(X, a.item())
            kl_xy = results['KL_X_Y']
            kl_yx = results['KL_Y_X']

            # 检查是否为有效数值（不是nan或inf）
            if not (torch.isnan(torch.tensor(kl_xy)) or torch.isinf(torch.tensor(kl_xy)) or
                    torch.isnan(torch.tensor(kl_yx)) or torch.isinf(torch.tensor(kl_yx))):
                kl_X_Y_values.append(kl_xy)
                kl_Y_X_values.append(kl_yx)
                valid_a_values.append(a.item())
                print(f"a={a.item():.1f}: KL(X||Y)={kl_xy:.6f}, KL(Y||X)={kl_yx:.6f}")
            else:
                print(f"Skipping a={a.item():.1f} (invalid values: KL(X||Y)={kl_xy}, KL(Y||X)={kl_yx})")
        except Exception as e:
            print(f"Skipping a={a.item():.1f} (calculation error: {e})")
            continue

    if not valid_a_values:
        print("No valid KL divergence values found, cannot create chart")
        return

    a_plot = valid_a_values
    kl_xy_plot = kl_X_Y_values
    kl_yx_plot = kl_Y_X_values

    # 创建图表
    plt.switch_backend('Agg')  # 使用非交互式后端
    plt.figure(figsize=(12, 8))

    # 绘制两条曲线
    plt.plot(a_plot, kl_xy_plot, 'b-', linewidth=2, label='KL(X||Y)', marker='o', markersize=4)
    plt.plot(a_plot, kl_yx_plot, 'r--', linewidth=2, label='KL(Y||X)', marker='s', markersize=4)

    # 设置图表属性
    plt.xlabel('Parameter a', fontsize=14)
    plt.ylabel('KL Divergence', fontsize=14)
    plt.title('KL Divergence vs Parameter a', fontsize=16, pad=20)
    plt.legend(fontsize=12)
    plt.grid(True, alpha=0.3)

    # 设置轴的范围和刻度
    plt.xlim(min(a_plot), max(a_plot))
    max_kl = max(max(kl_xy_plot), max(kl_yx_plot))
    plt.ylim(0, max_kl * 1.1)

    # 添加数值标签（最大值位置）
    if kl_xy_plot:
        max_kl_idx = kl_xy_plot.index(max(kl_xy_plot))
        plt.annotate(f'{kl_xy_plot[max_kl_idx]:.3f}',
                    xy=(a_plot[max_kl_idx], kl_xy_plot[max_kl_idx]),
                    xytext=(10, 10), textcoords='offset points',
                    fontsize=10, color='blue')

    # 保存图表
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    print(f"Chart saved as: {filename}")

    plt.close()
    print("plot_kl_vs_a function completed")


def analyze_optimal_a_vs_size(sizes=None, num_vectors_per_size=10, a_range=None,
                             distribution='uniform', filename="optimal_a_vs_size.jpg"):
    """
    分析不同向量长度下最佳a值的变化过程

    参数:
        sizes: 向量长度列表，默认[10, 50, 100, 200, 500, 1000]
        num_vectors_per_size: 每个长度生成的向量数量
        a_range: a值范围
        distribution: 向量分布类型
        filename: 保存的文件名
    """
    if sizes is None:
        sizes = [10, 50, 100, 200, 500, 1000]

    if a_range is None:
        a_range = torch.arange(2, 4.4, 0.01, device=device)

    print(f"Analyzing optimal a vs vector size for {distribution} distribution...")
    print(f"Sizes: {sizes}")
    print(f"Vectors per size: {num_vectors_per_size}")

    results = []

    for size in sizes:
        print(f"\n--- Analyzing size {size} ---")
        optimal_a_values = []

        for i in range(num_vectors_per_size):
            # 使用不同的种子
            seed =  random.randint(0,100000)#size * 1000 + i + 3000
            X = create_normalized_vector(size=size, seed=seed, distribution=distribution)

            # 找到这个向量对应的最小KL散度a值
            min_kl = float('inf')
            optimal_a = None

            for a in a_range:
                try:
                    results_dict = analyze_transformation(X, a.item())
                    kl_xy = results_dict['KL_X_Y']

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
            std_optimal_a = torch.tensor(optimal_a_values).std().item()
            results.append({
                'size': size,
                'optimal_a_values': optimal_a_values,
                'avg_optimal_a': avg_optimal_a,
                'std_optimal_a': std_optimal_a
            })
            print(f"Size {size}: {len(optimal_a_values)} vectors, avg optimal a = {avg_optimal_a:.3f} ± {std_optimal_a:.3f}")
        else:
            print(f"Size {size}: no valid optimal a values found")

    if not results:
        print("No results to plot")
        return

    # 提取数据用于绘图
    sizes_plot = [r['size'] for r in results]
    avg_a_plot = [r['avg_optimal_a'] for r in results]
    std_a_plot = [r['std_optimal_a'] for r in results]

    # 创建图表
    plt.switch_backend('Agg')
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

    # 子图1：平均最佳a值 vs 向量长度
    ax1.plot(sizes_plot, avg_a_plot, 'bo-', linewidth=2, markersize=8, label='Average Optimal a')
    ax1.fill_between(sizes_plot,
                     [a - s for a, s in zip(avg_a_plot, std_a_plot)],
                     [a + s for a, s in zip(avg_a_plot, std_a_plot)],
                     alpha=0.3, color='blue', label='±1 Std Dev')
    ax1.set_xlabel('Vector Size', fontsize=12)
    ax1.set_ylabel('Optimal Parameter a', fontsize=12)
    ax1.set_title(f'Average Optimal a vs Vector Size\n({distribution} distribution)', fontsize=14)
    ax1.grid(True, alpha=0.3)
    ax1.legend()

    # 设置x轴为对数刻度（如果大小差异很大）
    if max(sizes_plot) / min(sizes_plot) > 10:
        ax1.set_xscale('log')
        ax1.set_xlabel('Vector Size (log scale)', fontsize=12)

    # 子图2：所有数据点的散点图
    all_sizes = []
    all_a_values = []
    for r in results:
        all_sizes.extend([r['size']] * len(r['optimal_a_values']))
        all_a_values.extend(r['optimal_a_values'])

    ax2.scatter(all_sizes, all_a_values, alpha=0.6, s=30, color='red', edgecolors='black', linewidth=0.5)
    ax2.set_xlabel('Vector Size', fontsize=12)
    ax2.set_ylabel('Optimal Parameter a', fontsize=12)
    ax2.set_title(f'All Optimal a Values vs Vector Size\n({distribution} distribution)', fontsize=14)
    ax2.grid(True, alpha=0.3)

    # 设置x轴为对数刻度
    if max(sizes_plot) / min(sizes_plot) > 10:
        ax2.set_xscale('log')
        ax2.set_xlabel('Vector Size (log scale)', fontsize=12)

    plt.tight_layout()
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    print(f"\nOptimal a vs size analysis saved as: {filename}")

    plt.close()

    return results


def analyze_optimal_a_vs_size_multiple_distributions(sizes=None, num_vectors_per_size=5,
                                                    a_range=None, filename="optimal_a_migration.jpg"):
    """
    分析多个分布在不同向量长度下的最佳a值迁移过程
    """
    if sizes is None:
        sizes = [10, 50, 100, 200, 500, 1000]

    distributions = ['uniform', 'exponential', 'beta', 'gamma', 'normal']

    if a_range is None:
        a_range = torch.arange(2, 4.4, 0.01, device=device)

    print("Analyzing optimal a migration across multiple distributions and sizes...")

    # 为每个分布收集数据
    all_results = {}

    for distribution in distributions:
        print(f"\n=== Analyzing {distribution.upper()} distribution ===")
        results = []

        for size in sizes:
            optimal_a_values = []

            for i in range(num_vectors_per_size):
                seed = hash(f"{distribution}_{size}_{i}") % 100000
                X = create_normalized_vector(size=size, seed=seed, distribution=distribution)

                min_kl = float('inf')
                optimal_a = None

                for a in a_range:
                    try:
                        results_dict = analyze_transformation(X, a.item())
                        kl_xy = results_dict['KL_X_Y']

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
                results.append({
                    'size': size,
                    'avg_optimal_a': avg_optimal_a,
                    'count': len(optimal_a_values)
                })
                print(f"  Size {size}: avg optimal a = {avg_optimal_a:.3f} ({len(optimal_a_values)} vectors)")

        all_results[distribution] = results

    # 创建迁移图表
    plt.switch_backend('Agg')
    plt.figure(figsize=(14, 8))

    colors = ['blue', 'red', 'green', 'orange', 'purple']
    markers = ['o', 's', '^', 'D', 'x']

    for i, (distribution, results) in enumerate(all_results.items()):
        if results:
            sizes_plot = [r['size'] for r in results]
            avg_a_plot = [r['avg_optimal_a'] for r in results]

            plt.plot(sizes_plot, avg_a_plot, marker=markers[i], linewidth=2, markersize=8,
                    color=colors[i], label=f'{distribution.upper()}', alpha=0.8)

    plt.xlabel('Vector Size', fontsize=14)
    plt.ylabel('Average Optimal Parameter a', fontsize=14)
    plt.title('Optimal Parameter a Migration Across Vector Sizes\n(Different Distributions)', fontsize=16, pad=20)
    plt.grid(True, alpha=0.3)
    plt.legend(fontsize=12, bbox_to_anchor=(1.05, 1), loc='upper left')

    # 设置x轴为对数刻度
    plt.xscale('log')

    # 添加一些参考线
    plt.axhline(y=3.0, color='gray', linestyle='--', alpha=0.5, label='a=3.0 reference')

    plt.tight_layout()
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    print(f"\nOptimal a migration analysis saved as: {filename}")

    plt.close()

    return all_results


def plot_multiple_kl_vs_a(X_list, labels, a_range=None, filename="KL_A_multiple.jpg", kl_type="X_Y"):
    """
    在一张图上绘制多个分布的KL散度随参数a变化的图表

    参数:
        X_list: 输入向量列表
        labels: 标签列表，对应每个向量
        a_range: a值的范围，默认0.1到6.0，步长0.1
        filename: 保存的文件名
        kl_type: "X_Y" 或 "Y_X"，选择绘制KL(X||Y)还是KL(Y||X)
    """
    print(f"Plotting KL({kl_type.replace('_', '||')}) divergence for {len(X_list)} distributions")

    if a_range is None:
        a_range = torch.arange(2, 4.4, 0.01, device=device)  # 用户指定的a范围
    else:
        a_range = torch.tensor(a_range, dtype=torch.float32, device=device)

    # 根据a_range的step确定小数点位数
    if len(a_range) > 1:
        step = abs(a_range[1].item() - a_range[0].item())
        # 计算step的小数点位数
        if step >= 1:
            decimal_places = 1
        else:
            decimal_places = max(1, int(round(-math.log10(step))))
    else:
        decimal_places = 2  # 默认值

    print(f"a_range step: {step:.6f}, using {decimal_places} decimal places for annotation")

    # 设置颜色循环
    colors = ['blue', 'red', 'green', 'orange', 'purple', 'brown', 'pink', 'gray', 'olive', 'cyan']

    # 创建图表
    plt.switch_backend('Agg')
    plt.figure(figsize=(14, 10))

    all_kl_values = []

    for i, (X, label) in enumerate(zip(X_list, labels)):
        kl_values = []
        valid_a_values = []

        for a in a_range:
            try:
                results = analyze_transformation(X, a.item())
                if kl_type == "X_Y":
                    kl_val = results['KL_X_Y']
                else:  # kl_type == "Y_X"
                    kl_val = results['KL_Y_X']

                # 检查是否为有效数值
                if not (torch.isnan(torch.tensor(kl_val)) or torch.isinf(torch.tensor(kl_val))):
                    kl_values.append(kl_val)
                    valid_a_values.append(a.item())
                else:
                    break  # 一旦出现NaN就停止，避免后续都是NaN
            except Exception as e:
                print(f"跳过{label}的a={a.item():.1f}（计算错误: {e}）")
                break

        if valid_a_values and kl_values:
            color = colors[i % len(colors)]
            plt.plot(valid_a_values, kl_values, 'o-', linewidth=2, markersize=4,
                    color=color, label=label, alpha=0.8)
            all_kl_values.extend(kl_values)
            print(f"{label}: plotted {len(valid_a_values)} points, KL range: {min(kl_values):.6f} - {max(kl_values):.6f}")
        else:
            print(f"Warning: {label} has no valid KL values")

    # 设置图表属性
    kl_name = "KL(X||Y)" if kl_type == "X_Y" else "KL(Y||X)"
    plt.xlabel('Parameter a', fontsize=14)
    plt.ylabel('KL Divergence', fontsize=14)
    plt.title(f'{kl_name} vs Parameter a - Multi-distribution Comparison', fontsize=16, pad=20)
    plt.legend(fontsize=12, bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.grid(True, alpha=0.3)

    # 标记每个分布的最小KL散度对应的a值
    colors = ['blue', 'red', 'green', 'orange', 'purple', 'brown', 'pink', 'gray', 'olive', 'cyan']
    for i, (X, label) in enumerate(zip(X_list, labels)):
        min_kl = float('inf')
        min_a = None
        kl_values_for_dist = []
        a_values_for_dist = []

        for a in a_range:
            try:
                results = analyze_transformation(X, a.item())
                if kl_type == "X_Y":
                    kl_val = results['KL_X_Y']
                else:
                    kl_val = results['KL_Y_X']

                if not (torch.isnan(torch.tensor(kl_val)) or torch.isinf(torch.tensor(kl_val))):
                    kl_values_for_dist.append(kl_val)
                    a_values_for_dist.append(a.item())
                    if kl_val < min_kl:
                        min_kl = kl_val
                        min_a = a.item()
            except Exception:
                continue

        if min_a is not None:
            color = colors[i % len(colors)]
            plt.scatter([min_a], [min_kl], color=color, s=100, marker='*',
                       edgecolors='black', linewidth=2, zorder=5)
            # 根据step动态设置小数点位数
            format_str = f'{min_a:.{decimal_places}f}'
            plt.annotate(format_str,
                        xy=(min_a, min_kl),
                        xytext=(15, 15), textcoords='offset points',
                        fontsize=10, color=color, fontweight='bold',
                        bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8))

    # 设置轴的范围
    if all_kl_values:
        max_kl = max(all_kl_values)
        plt.ylim(0, max_kl * 1.1)

    plt.tight_layout()
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    print(f"Multi-distribution comparison chart saved as: {filename}")

    plt.close()


def analyze_optimal_a_distribution(num_vectors=50, vector_size=256, a_range=None,
                                 distributions=None, filename="optimal_a_distribution.jpg"):
    """
    统计多个随机向量最佳a值的分布 - 多分布对比

    参数:
        num_vectors: 每个分布的随机向量数量
        vector_size: 向量长度
        a_range: a值范围
        distributions: 分布类型列表，默认使用所有支持的分布
        filename: 保存的文件名
    """
    if distributions is None:
        distributions = ['uniform', 'exponential', 'beta', 'gamma', 'normal']

    print(f"Analyzing optimal a distribution for {num_vectors} vectors per distribution...")
    print(f"Distributions: {distributions}")

    
    if a_range is None:
        a_range = torch.arange(2, 4.4, 0.01)  # 用户指定的a范围
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
        print(f"\n--- Analyzing {distribution.upper()} distribution ---")

        optimal_a_values = []

        for i in range(num_vectors):
            if i % 10 == 0 and i > 0:
                print(f"  Processing vector {i+1}/{num_vectors}...")

            # 使用不同的种子生成随机向量
            seed = random.randint(0,100000)#idx * 1000 + i + 2000  # 确保不同分布和向量有不同种子
            
            X = create_normalized_vector(size=vector_size, seed=seed, distribution=distribution)

            # 找到这个向量对应的最小KL散度a值
            min_kl = float('inf')
            optimal_a = None

            for a in a_range:
                try:
                    results = analyze_transformation(X, a.item())
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
            ax.scatter(optimal_a_values, y_positions, alpha=0.7, s=20, color='blue',
                      edgecolors='none', linewidth=0)

            # 设置子图属性
            ax.set_xlabel('Optimal Parameter a', fontsize=12)
            ax.set_ylabel('', fontsize=12)  # 不显示y轴标签
            ax.set_title(f'{distribution.upper()}\n(n={len(optimal_a_values)})', fontsize=14, pad=10)
            ax.grid(True, alpha=0.3)

            # 设置x轴和y轴范围
            ax.set_xlim(3, 4)  # 根据a_range设置
            ax.set_ylim(0, 1)    # 固定y轴范围

            # 隐藏y轴刻度
            ax.set_yticks([])
            ax.set_yticklabels([])

    # 隐藏多余的子图
    for idx in range(len(distributions), len(axes)):
        axes[idx].set_visible(False)

    # 设置总标题
    fig.suptitle(f'Optimal Parameter a Distribution Across Different Random Strategies\n({num_vectors} vectors per distribution, size={vector_size})',
                fontsize=16, y=0.98)

    plt.tight_layout()
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    print(f"\nOptimal a distribution comparison chart saved as: {filename}")

    plt.close()

    return all_optimal_values


# 示例用法
if __name__ == "__main__":
    print("WM Transform Analysis")
    print("=" * 50)

    # 使用固定参数进行测试
    X = create_normalized_vector(size=5, seed=42, distribution='uniform')
    print(f"Test vector X (length={len(X)}, sum={X.sum().item():.6f}):")
    print(X)

    # 测试新的向量长度分析功能
    print("\nTesting optimal a migration across vector sizes...")
    try:
        migration_results = analyze_optimal_a_vs_size_multiple_distributions(
            sizes=[10, 50, 100],
            num_vectors_per_size=3,
            filename="optimal_a_migration.jpg"
        )
        print("Optimal a migration analysis completed successfully")
    except Exception as e:
        print(f"Optimal a migration analysis failed: {e}")
        import traceback
        traceback.print_exc()

    print("\nWM变换分析完成!")
    print("WM Transform Analysis")
    print("=" * 50)
    print("Starting main function")

    # 创建测试向量 - 展示不同分布
    print("Creating test vectors...")

    # 固定种子以便重现结果
    import random
    seed = random.randint(0,10000000)
    
    # 创建不同分布的向量
    distributions = [
        ('uniform', {}),
        ('exponential', {'rate': 1.0}),
        ('beta', {'alpha': 2.0, 'beta': 5.0}),
        ('gamma', {'shape': 2.0, 'rate': 1.0}),
        ('normal', {'loc': 1.0, 'scale': 0.5})
    ]

    # 存储所有向量和标签，用于多分布对比
    X_vectors = []
    labels = []
    a_range = torch.arange(0, 5, 0.01, device=device)  # 使用合理的范围和步长
    length = 2**10
    for dist_name, dist_params in distributions:
        print(f"\n--- {dist_name.upper()} DISTRIBUTION ---")
        X = create_normalized_vector(size=length, seed=seed, distribution=dist_name, **dist_params)
        print(f"Input vector X ({dist_name}, length={len(X)}, sum={X.sum().item():.6f}):")
        print(X)

        # 存储向量和标签
        X_vectors.append(X)
        labels.append(dist_name.upper())

        # 为每个分布生成KL散度图表
        try:
            plot_kl_vs_a(X, a_range=a_range,
                        filename=f"KL_A_{dist_name}.jpg")
            print(f"{dist_name} distribution chart generated successfully")
        except Exception as e:
            print(f"{dist_name} distribution chart generation failed: {e}")

    # 生成多分布对比图表
    print("\nGenerating multi-distribution comparison charts...")
    try:
        plot_multiple_kl_vs_a(X_vectors, labels, a_range=a_range,
                             filename="KL_X_Y_comparison.jpg", kl_type="X_Y")
        print("KL(X||Y) multi-distribution comparison chart generated successfully")
    except Exception as e:
        print(f"KL(X||Y) multi-distribution comparison chart generation failed: {e}")

    try:
        plot_multiple_kl_vs_a(X_vectors, labels, a_range=a_range,
                             filename="KL_Y_X_comparison.jpg", kl_type="Y_X")
        print("KL(Y||X) multi-distribution comparison chart generated successfully")
    except Exception as e:
        print(f"KL(Y||X) multi-distribution comparison chart generation failed: {e}")

    # 使用uniform分布进行后续分析
    X = create_normalized_vector(size=5, seed=seed, distribution='uniform')
    print(f"\nFinal vector X (uniform, length={len(X)}, sum={X.sum().item():.6f}):")
    print(X)
    print("Vector creation completed")

    print("绘图部分执行完成")

    # 分析最佳a值的分布
    print("\nAnalyzing optimal a distribution across all distributions...")
    try:
        #a_range = torch.arange(3, 4, 0.0003)
        optimal_a_values = analyze_optimal_a_distribution(
            num_vectors=30,  # 每个分布使用30个向量进行统计
            vector_size=length,
            a_range=a_range,
            distributions=['uniform', 'exponential', 'beta', 'gamma', 'normal'],
            filename="optimal_a_distribution.jpg"
        )
        total_vectors = sum(len(values) for values in optimal_a_values.values())
        print(f"Successfully analyzed {total_vectors} optimal a values across {len(optimal_a_values)} distributions")
    except Exception as e:
        print(f"Optimal a distribution analysis failed: {e}")
        import traceback
        traceback.print_exc()

    # 分析不同向量长度下的最佳a值迁移过程
    print("\nAnalyzing optimal a migration across vector sizes...")
    try:
        size = 2**torch.tensor(list(range(7,25,1))) 
        migration_results = analyze_optimal_a_vs_size_multiple_distributions(
            sizes=size,
            num_vectors_per_size=5,
            a_range=a_range,
            filename="optimal_a_migration.jpg"
        )
        print("Optimal a migration analysis completed successfully")
    except Exception as e:
        print(f"Optimal a migration analysis failed: {e}")
        import traceback
        traceback.print_exc()

    print("准备打印完成信息")

    print("\nWM变换分析完成!")
