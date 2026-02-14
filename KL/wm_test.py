import torch
import torch.nn as nn
import math
from KL import kl_divergence_manual


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


def create_normalized_vector(size=10, seed=42):
    """
    创建一个非负浮点向量，向量和为整数1

    参数:
        size: 向量长度
        seed: 随机种子
    返回:
        归一化后的非负向量
    """
    torch.manual_seed(seed)
    # 创建随机向量
    x = torch.rand(size)
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
        X = torch.tensor(X, dtype=torch.float32)
    if not isinstance(Y, torch.Tensor):
        Y = torch.tensor(Y, dtype=torch.float32)

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
        X = torch.tensor(X, dtype=torch.float32)

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
    print("WM变换分析")
    print("=" * 50)

    # 创建测试向量
    X = create_normalized_vector(size=5, seed=42)
    print(f"输入向量X (长度={len(X)}, 和={X.sum().item():.6f}):")
    print(X)

    # 分析变换过程
    results = analyze_transformation(X, a=5.0)

    print(f"\na = {results['a']}")
    print(f"w(X) = {results['w_X']}")
    print(f"f(w(X)) = {results['f_w_X']}")
    print(f"Y = m(f(w(X))) = {results['Y']}")
    print(f"Y的和 = {results['Y_sum']:.6f}")

    print(f"\nKL散度:")
    print(f"KL(X||Y) = {results['KL_X_Y']:.6f}")
    print(f"KL(Y||X) = {results['KL_Y_X']:.6f}")

    # 测试不同a值的影响
    print("\n测试不同a值的影响:")
    a_values = [1.0, 2.0, 5.0, 10.0]
    for a_test in a_values:
        results_a = analyze_transformation(X, a=a_test)
        print(f"a={a_test}: KL(X||Y)={results_a['KL_X_Y']:.6f}, KL(Y||X)={results_a['KL_Y_X']:.6f}")

    print("\nWM变换分析完成!")