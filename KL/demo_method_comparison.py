#!/usr/bin/env python3
"""
演示Max-Normalized与原生方法最优a值对比功能
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

def demo_method_comparison():
    """演示方法对比功能"""
    print("🎯 Method Comparison Demo: Max-Normalized vs Original")
    print("=" * 60)

    # 演示参数 - 使用较小的值以便快速演示
    vector_sizes = [16, 32, 64, 128]  # 以2为底的指数尺度
    num_vectors_per_size = 8          # 每个大小的向量数量
    distribution = 'uniform'          # 演示用统一分布
    a_range = torch.arange(2.5, 4.0, 0.1)  # a值范围

    print(f"Demo Parameters:")
    print(f"  - Vector sizes: {vector_sizes} (powers of 2)")
    print(f"  - Vectors per size: {num_vectors_per_size}")
    print(f"  - Distribution: {distribution}")
    print(f"  - a range: [{a_range.min().item():.1f}, {a_range.max().item():.1f}]")
    print()

    plt.switch_backend('Agg')
    fig, ax = plt.subplots(1, 1, figsize=(12, 8))

    colors = {'max_normalized': '#e74c3c', 'original': '#3498db'}
    markers = {'max_normalized': 'o', 'original': 's'}

    results = {'max_normalized': [], 'original': []}

    print(f"Analyzing {distribution} distribution across different vector sizes...")

    for i, size in enumerate(vector_sizes):
        print(f"  Vector size 2^{int(math.log2(size))}: ", end='')

        # 为每种方法收集最优a值
        optimal_a_max_norm = []
        optimal_a_original = []

        for j in range(num_vectors_per_size):
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
            print(".3f", end=' | ')
        else:
            results['max_normalized'].append(None)
            print("N/A   | ", end='')

        if optimal_a_original:
            avg_original = sum(optimal_a_original) / len(optimal_a_original)
            results['original'].append(avg_original)
            print(".3f")
        else:
            results['original'].append(None)
            print("N/A")

    # 绘制对比图
    x_positions = list(range(len(vector_sizes)))  # 整数位置用于x轴

    for method in ['max_normalized', 'original']:
        valid_data = [(i, val) for i, val in enumerate(results[method]) if val is not None]
        if valid_data:
            x_vals, y_vals = zip(*valid_data)
            ax.plot(x_vals, y_vals, marker=markers[method], linestyle='-', color=colors[method],
                   linewidth=3, markersize=10, label=f'{"Max-Normalized" if method == "max_normalized" else "Original"}',
                   alpha=0.8, markerfacecolor='white', markeredgewidth=2)

    # 设置x轴标签为log2尺度
    ax.set_xticks(x_positions)
    ax.set_xticklabels([f'$2^{{{int(math.log2(size))}}}$' for size in vector_sizes])

    ax.set_xlabel('Vector Size (log₂ scale)', fontsize=14)
    ax.set_ylabel('Average Optimal Parameter a', fontsize=14)
    ax.set_title(f'Max-Normalized vs Original WM Transform\n{distribution.title()} Distribution: Optimal a Comparison', fontsize=16, pad=20)
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=12, loc='best')

    # 添加数值标签
    for i, (max_norm, orig) in enumerate(zip(results['max_normalized'], results['original'])):
        if max_norm is not None:
            ax.annotate(f'{max_norm:.3f}', (i, max_norm),
                       textcoords="offset points", xytext=(0, 15), ha='center',
                       fontsize=11, color=colors['max_normalized'], fontweight='bold')
        if orig is not None:
            ax.annotate(f'{orig:.3f}', (i, orig),
                       textcoords="offset points", xytext=(0, -20), ha='center',
                       fontsize=11, color=colors['original'], fontweight='bold')

    # 添加说明文本
    ax.text(0.02, 0.98, 'Red: Max-Normalized (X_max normalization)\nBlue: Original (relative to 1.0)',
            transform=ax.transAxes, fontsize=10, verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))

    plt.tight_layout()
    plt.savefig("demo_method_comparison.jpg", dpi=300, bbox_inches='tight')
    print("\n📊 Demo comparison plot saved as: demo_method_comparison.jpg")

    # 打印总结
    print("\n📈 Summary:")
    print("The plot shows how the optimal parameter 'a' changes with vector size")
    print("for both Max-Normalized and Original watermarking transforms.")
    print("X-axis uses log2 scale to better visualize the scale effects.")

    plt.close()

    print("✅ Method comparison demo completed successfully!")

if __name__ == "__main__":
    demo_method_comparison()