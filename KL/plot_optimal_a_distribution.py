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
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from KL.KL import kl_divergence_manual
from KL.wm_simple import create_normalized_vector


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
        a_range = torch.arange(2, 4.4, 0.01, device=device)  # 0.1 to 6.0 with step 0.1

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
            seed = idx * 1000 + i + 2000  # 确保不同分布和向量有不同种子
            if distribution == 'uniform':
                X = torch.rand(vector_size, device=device)
            elif distribution == 'exponential':
                rate = 1.0
                X = torch.distributions.Exponential(rate).sample((vector_size,)).to(device)
            elif distribution == 'beta':
                alpha, beta = 2.0, 5.0
                X = torch.distributions.Beta(alpha, beta).sample((vector_size,)).to(device)
            elif distribution == 'gamma':
                shape, rate = 2.0, 1.0
                X = torch.distributions.Gamma(shape, rate).sample((vector_size,)).to(device)
            elif distribution == 'normal':
                loc, scale = 1.0, 0.5
                X = torch.abs(torch.distributions.Normal(loc, scale).sample((vector_size,)).to(device))
            else:
                raise ValueError(f"Unsupported distribution: {distribution}")

            # 归一化
            X = X / X.sum()

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
        print(".3f")

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
            ax.set_xlim(2, 4.4)  # 根据a_range设置
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
    print("Optimal A Distribution Plot Module")
    print("Use analyze_optimal_a_distribution(num_vectors, vector_size, a_range, distributions, filename) to create distribution plots")