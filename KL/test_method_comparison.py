#!/usr/bin/env python3
"""
测试Max-Normalized与原生方法对比功能的快速脚本
"""

import sys
import os
import torch
import random
import math

# 添加父目录到路径
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, parent_dir)

from KL.wm_max_normalized import analyze_transformation_max_normalized
from KL.wm_simple import create_normalized_vector, analyze_transformation
import matplotlib.pyplot as plt

def test_method_comparison():
    """快速测试方法对比功能"""
    print("🧪 Testing Method Comparison: Max-Normalized vs Original")
    print("=" * 60)

    # 测试参数
    vector_sizes = [16, 32, 64]  # 较少的向量大小用于快速测试
    num_vectors_per_size = 5     # 较少的向量数量
    distributions = ['uniform']
    a_range = torch.arange(2, 4.4, 0.05)  # 更大的步长以加快计算

    print(f"Test Parameters:")
    print(f"  - Vector sizes: {vector_sizes}")
    print(f"  - Vectors per size: {num_vectors_per_size}")
    print(f"  - Distributions: {distributions}")
    print()

    plt.switch_backend('Agg')
    fig, ax = plt.subplots(1, 1, figsize=(10, 6))

    colors = {'max_normalized': 'red', 'original': 'blue'}

    for distribution in distributions:
        print(f"--- Testing {distribution.upper()} distribution ---")

        results = {'max_normalized': [], 'original': []}

        for size in vector_sizes:
            print(f"  Vector size: {size}")

            # 为每种方法收集最优a值
            optimal_a_max_norm = []
            optimal_a_original = []

            for i in range(num_vectors_per_size):
                seed = random.randint(0, 100000)
                X = create_normalized_vector(size=size, seed=seed, distribution=distribution)

                # Max-Normalized方法
                min_kl = float('inf')
                optimal_a = None
                for a in a_range:
                    try:
                        results_max = analyze_transformation_max_normalized(X, a.item())
                        kl_xy = results_max['KL_X_Y']
                        if not (torch.isnan(torch.tensor(kl_xy)) or torch.isinf(torch.tensor(kl_xy))):
                            if kl_xy < min_kl:
                                min_kl = kl_xy
                                optimal_a = a.item()
                    except Exception:
                        continue
                if optimal_a is not None:
                    optimal_a_max_norm.append(optimal_a)

                # Original方法
                min_kl = float('inf')
                optimal_a = None
                for a in a_range:
                    try:
                        results_orig = analyze_transformation(X, a.item())
                        kl_xy = results_orig['KL_X_Y']
                        if not (torch.isnan(torch.tensor(kl_xy)) or torch.isinf(torch.tensor(kl_xy))):
                            if kl_xy < min_kl:
                                min_kl = kl_xy
                                optimal_a = a.item()
                    except Exception:
                        continue
                if optimal_a is not None:
                    optimal_a_original.append(optimal_a)

            # 计算平均值
            if optimal_a_max_norm:
                avg_max_norm = sum(optimal_a_max_norm) / len(optimal_a_max_norm)
                results['max_normalized'].append(avg_max_norm)
                print(".4f")
            else:
                results['max_normalized'].append(None)

            if optimal_a_original:
                avg_original = sum(optimal_a_original) / len(optimal_a_original)
                results['original'].append(avg_original)
                print(".4f")
            else:
                results['original'].append(None)

        # 绘制对比图
        x_positions = [i for i in range(len(vector_sizes))]  # 整数位置用于x轴

        # Max-Normalized方法
        valid_x_max = [x for x, y in zip(x_positions, results['max_normalized']) if y is not None]
        valid_y_max = [y for y in results['max_normalized'] if y is not None]
        if valid_x_max and valid_y_max:
            ax.plot(valid_x_max, valid_y_max, 'o-', color=colors['max_normalized'],
                   linewidth=3, markersize=8, label='Max-Normalized', alpha=0.8)

        # Original方法
        valid_x_orig = [x for x, y in zip(x_positions, results['original']) if y is not None]
        valid_y_orig = [y for y in results['original'] if y is not None]
        if valid_x_orig and valid_y_orig:
            ax.plot(valid_x_orig, valid_y_orig, 's-', color=colors['original'],
                   linewidth=3, markersize=8, label='Original', alpha=0.8)

    # 设置x轴标签为log2尺度
    ax.set_xticks(x_positions)
    ax.set_xticklabels([f'2^{int(math.log2(size))}' for size in vector_sizes])

    ax.set_xlabel('Vector Size (log₂ scale)', fontsize=14)
    ax.set_ylabel('Average Optimal Parameter a', fontsize=14)
    ax.set_title('Method Comparison: Max-Normalized vs Original', fontsize=16, pad=20)
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=12)

    plt.tight_layout()
    plt.savefig("test_method_comparison.jpg", dpi=300, bbox_inches='tight')
    print("\nTest comparison plot saved as: test_method_comparison.jpg")

    plt.close()

    print("✅ Method comparison test completed!")

if __name__ == "__main__":
    test_method_comparison()