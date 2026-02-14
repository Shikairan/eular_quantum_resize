import torch
import torch.nn as nn
import math
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
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from KL.KL import kl_divergence_manual


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
        X = torch.tensor(X, dtype=torch.float32, device=device)

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


# 示例用法
if __name__ == "__main__":
    print("WM Transform Core Module")
    print("Core WM transformation functions without plotting")
    print("Use individual plot modules for visualization:")
    print("- plot_kl_single.py: Single vector KL plots")
    print("- plot_kl_comparison.py: Multi-vector KL comparison plots")
    print("- plot_optimal_a_distribution.py: Optimal a distribution plots")
    print("- plot_size_migration.py: Size migration plots")