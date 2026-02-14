#!/usr/bin/env python3
"""
快速演示最大值归一化WM变换的最优a值分析
使用较少的向量和较小的参数以加快演示速度
"""

import sys
import os

# 添加父目录到路径
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, parent_dir)

from KL.wm_max_normalized import (
    analyze_optimal_a_distribution_max_normalized,
    analyze_optimal_a_vs_size_max_normalized
)

def quick_optimal_a_demo():
    """快速演示最优a值分析"""
    print("🚀 Quick Max-Normalized Optimal a Analysis Demo")
    print("=" * 50)

    print("Generating optimal a distribution plot (quick version)...")
    optimal_values = analyze_optimal_a_distribution_max_normalized(
        num_vectors=15,  # 减少向量数量
        vector_size=50,  # 减小向量长度
        distributions=['uniform', 'exponential', 'beta'],
        filename="quick_optimal_a_distribution_max_normalized.jpg"
    )

    print("\n📊 Quick Statistics:")
    for dist, values in optimal_values.items():
        if values:
            avg_a = sum(values) / len(values)
            print("8s")

    print("\nGenerating optimal a vs size migration plot (quick version)...")
    analyze_optimal_a_vs_size_max_normalized(
        vector_sizes=[10, 20, 50],  # 减少向量长度选项
        num_vectors_per_size=8,     # 减少每个长度的向量数量
        distributions=['uniform', 'exponential'],
        filename="quick_optimal_a_vs_size_max_normalized.jpg"
    )

    print("\n✅ Quick demo completed!")
    print("\n📁 Generated files:")
    print("   - quick_optimal_a_distribution_max_normalized.jpg")
    print("   - quick_optimal_a_vs_size_max_normalized.jpg")

if __name__ == "__main__":
    quick_optimal_a_demo()