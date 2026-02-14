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

# 导入KL模块
from .KL import kl_divergence_manual


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


def create_normalized_vector(size=10, seed=42, distribution='uniform', **kwargs):
    """
    创建一个非负浮点向量，向量和为整数1

    参数:
        size: 向量长度
        seed: 随机种子
        distribution: 分布类型 ('uniform', 'exponential', 'beta', 'gamma', 'normal')
        **kwargs: 分布参数
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
                print(f"Skipping a={a.item():.1f} (invalid values)")
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
    plt.switch_backend('Agg')
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

    # 保存图表
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    print(f"Chart saved as: {filename}")

    plt.close()


# 示例用法
if __name__ == "__main__":
    # 这里可以添加测试代码
    print("KL Single Plot Module")
    print("Use plot_kl_vs_a(X, a_range, filename) to create single vector KL plots")