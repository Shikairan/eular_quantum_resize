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
from .KL import kl_divergence_manual
from .wm_simple import create_normalized_vector


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
    return 1 + (1/a) * torch.log(x / 255.0)


def f(x, a=5.0):
    """
    计算f(x) = round(255*w(x))

    参数:
        x: 输入值或张量
        a: 参数，默认值为5.0
    返回:
        计算结果，四舍五入到最接近的整数
    """
    return torch.round(255 * w(x, a))


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
                # 生成不同分布的向量
                if distribution == 'uniform':
                    X = torch.rand(size, device=device)
                elif distribution == 'exponential':
                    rate = 1.0
                    X = torch.distributions.Exponential(rate).sample((size,)).to(device)
                elif distribution == 'beta':
                    alpha, beta = 2.0, 5.0
                    X = torch.distributions.Beta(alpha, beta).sample((size,)).to(device)
                elif distribution == 'gamma':
                    shape, rate = 2.0, 1.0
                    X = torch.distributions.Gamma(shape, rate).sample((size,)).to(device)
                elif distribution == 'normal':
                    loc, scale = 1.0, 0.5
                    X = torch.abs(torch.distributions.Normal(loc, scale).sample((size,)).to(device))
                else:
                    raise ValueError(f"Unsupported distribution: {distribution}")

                # 归一化
                X = X / X.sum()

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


# 示例用法
if __name__ == "__main__":
    print("Size Migration Plot Module")
    print("Use analyze_optimal_a_vs_size_multiple_distributions(sizes, num_vectors_per_size, a_range, filename) to create migration plots")