#!/usr/bin/env python3
"""
生成最大值归一化WM变换最优a值分析图表的独立脚本

此脚本用于生成：
1. 不同分布下最优a值的分布图
2. 不同向量长度下最优a值的迁移图
3. Max-Normalized与原生方法的最优a值对比图（x轴为log2尺度）
"""

import sys
import os
import torch
import random
import math

# 添加父目录到路径
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, parent_dir)

from KL.wm_max_normalized import (
    analyze_optimal_a_distribution_max_normalized,
    analyze_optimal_a_vs_size_max_normalized,
    analyze_transformation_max_normalized
)
from KL.wm_simple import (
    create_normalized_vector,
    analyze_transformation
)

def generate_optimal_a_distribution_plot():
    """生成最优a值分布图"""
    print("🔍 Generating Optimal a Distribution Plot (Max-Normalized)")
    print("=" * 60)

    # 用户可以调整这些参数
    num_vectors = 20  # 每个分布的向量数量 (减少用于快速测试)
    vector_size = 64  # 向量长度
    distributions = ['uniform', 'exponential', 'beta']  # 减少分布数量

    print(f"Parameters:")
    print(f"  - Vectors per distribution: {num_vectors}")
    print(f"  - Vector size: {vector_size}")
    print(f"  - Distributions: {distributions}")
    print()

    optimal_values = analyze_optimal_a_distribution_max_normalized(
        num_vectors=num_vectors,
        vector_size=vector_size,
        distributions=distributions,
        filename="optimal_a_distribution_max_normalized.jpg"
    )

    # 打印统计信息
    print("\n📊 Optimal a Value Statistics:")
    print("-" * 40)
    for dist, values in optimal_values.items():
        if values:
            avg_a = sum(values) / len(values)
            min_a = min(values)
            max_a = max(values)
            print("8s")
        else:
            print("8s")

def generate_optimal_a_vs_size_plot():
    """生成最优a值vs向量大小迁移图"""
    print("\n📈 Generating Optimal a vs Vector Size Migration Plot")
    print("=" * 55)

    # 用户可以调整这些参数
    vector_sizes = [int(2**i) for i in range(7, 23)]  # [16, 32, 64, 128, 256] - 以2为底的指数尺度
    num_vectors_per_size = 20  # 每个长度使用多少个随机向量 (减少用于快速测试)
    distributions = ['uniform', 'exponential', 'beta']

    print(f"Parameters:")
    print(f"  - Vector sizes: {vector_sizes}")
    print(f"  - Vectors per size: {num_vectors_per_size}")
    print(f"  - Distributions: {distributions}")
    print()

    analyze_optimal_a_vs_size_max_normalized(
        vector_sizes=vector_sizes,
        num_vectors_per_size=num_vectors_per_size,
        distributions=distributions,
        filename="optimal_a_vs_size_max_normalized.jpg"
    )

def generate_method_comparison_plot():
    """生成Max-Normalized与原生方法的最优a值对比图表"""
    print("\n🔄 Generating Method Comparison Plot: Max-Normalized vs Original")
    print("=" * 70)

    # 用户可以调整这些参数
    vector_sizes = [int(2**i) for i in range(7, 23)]  # [16, 32, 64, 128, 256, 512] - 以2为底的指数尺度
    num_vectors_per_size = 15  # 每个长度使用多少个随机向量
    distributions = ['uniform', 'exponential', 'beta']
    a_range = torch.arange(2, 4.4, 0.01)

    print(f"Parameters:")
    print(f"  - Vector sizes: {vector_sizes} (2^4 to 2^9)")
    print(f"  - Vectors per size: {num_vectors_per_size}")
    print(f"  - Distributions: {distributions}")
    print(f"  - a range: [{a_range.min().item():.2f}, {a_range.max().item():.2f}]")
    print()

    import matplotlib.pyplot as plt
    plt.switch_backend('Agg')
    fig, axes = plt.subplots(len(distributions), 1, figsize=(12, 5*len(distributions)))
    if len(distributions) == 1:
        axes = [axes]

    colors = {'max_normalized': 'red', 'original': 'blue'}

    for idx, distribution in enumerate(distributions):
        print(f"--- Analyzing {distribution.upper()} distribution ---")

        ax = axes[idx]
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
            else:
                results['max_normalized'].append(None)

            if optimal_a_original:
                avg_original = sum(optimal_a_original) / len(optimal_a_original)
                results['original'].append(avg_original)
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
        ax.set_title(f'{distribution.title()} Distribution: Method Comparison', fontsize=16, pad=20)
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=12)

        # 添加数值标签
        for i, (max_norm, orig) in enumerate(zip(results['max_normalized'], results['original'])):
            if max_norm is not None:
                ax.annotate(f'{max_norm:.2f}', (i, max_norm),
                           textcoords="offset points", xytext=(0, 10), ha='center',
                           fontsize=10, color=colors['max_normalized'])
            if orig is not None:
                ax.annotate(f'{orig:.2f}', (i, orig),
                           textcoords="offset points", xytext=(0, -15), ha='center',
                           fontsize=10, color=colors['original'])

    plt.suptitle('Max-Normalized vs Original WM Transform: Optimal a Comparison\n(Across Different Vector Scales)', fontsize=18, y=0.95)
    plt.tight_layout()
    plt.savefig("method_comparison_optimal_a.jpg", dpi=300, bbox_inches='tight')
    print("Method comparison plot saved as: method_comparison_optimal_a.jpg")

    plt.close()

def main():
    """主函数"""
    print("🎨 Max-Normalized WM Transform: Optimal a Analysis")
    print("=" * 55)

    try:
        generate_optimal_a_distribution_plot()
        generate_optimal_a_vs_size_plot()
        generate_method_comparison_plot()

        print("\n✅ All plots generated successfully!")
        print("\n📁 Generated files:")
        print("   - optimal_a_distribution_max_normalized.jpg")
        print("   - optimal_a_vs_size_max_normalized.jpg")
        print("   - method_comparison_optimal_a.jpg")

        print("\n💡 Usage Tips:")
        print("   • Increase num_vectors for more accurate statistics")
        print("   • Adjust vector_size based on your application")
        print("   • Modify distributions list to focus on specific types")
        print("   • The comparison plot shows how optimal a values differ between methods")
        print("   • X-axis uses log2 scale for better visualization of scale effects")

    except Exception as e:
        print(f"❌ Plot generation failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
