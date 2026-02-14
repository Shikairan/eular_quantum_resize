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

                # 检查是否为有效数值（不是nan或inf）
                if not (torch.isnan(torch.tensor(kl_val)) or torch.isinf(torch.tensor(kl_val))):
                    kl_values.append(kl_val)
                    valid_a_values.append(a.item())
                else:
                    break  # 一旦出现NaN就停止，避免后续都是NaN
            except Exception as e:
                print(f"Skipping {label}'s a={a.item():.1f} (calculation error: {e})")
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

    # 设置轴的范围
    if all_kl_values:
        max_kl = max(all_kl_values)
        plt.ylim(0, max_kl * 1.1)

    plt.tight_layout()
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    print(f"Multi-distribution comparison chart saved as: {filename}")

    plt.close()


# 示例用法
if __name__ == "__main__":
    print("KL Comparison Plot Module")
    print("Use plot_multiple_kl_vs_a(X_list, labels, a_range, filename, kl_type) to create comparison plots")