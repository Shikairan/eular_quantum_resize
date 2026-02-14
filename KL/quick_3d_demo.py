#!/usr/bin/env python3
"""
快速3D KL散度分析演示 - 使用较少的参数以加快速度
"""

import sys
import os
import torch
import numpy as np
import random
import math
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
from matplotlib import cm

# 添加父目录到路径
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, parent_dir)

from KL.wm_max_normalized import analyze_transformation_max_normalized
from KL.wm_simple import create_normalized_vector

def quick_3d_analysis():
    """快速3D分析演示"""
    print("🚀 Quick 3D KL Divergence Analysis Demo")
    print("=" * 45)

    # 使用较少的参数以加快速度
    vector_sizes = [16, 32, 64]  # 较少的向量大小
    a_values = torch.arange(2.5, 4.0, 0.2).numpy()  # 较大的步长
    distribution = 'uniform'
    num_vectors_per_size = 3  # 较少的样本数

    print(f"Quick demo parameters:")
    print(f"  - Vector sizes: {vector_sizes}")
    print(f"  - a values: [{a_values.min():.1f}, {a_values.max():.1f}] (step: {a_values[1]-a_values[0]:.1f})")
    print(f"  - Vectors per size: {num_vectors_per_size}")
    print()

    # 创建网格
    X, Y = np.meshgrid(
        [math.log2(size) for size in vector_sizes],
        a_values
    )
    Z = np.zeros_like(X, dtype=float)

    print("Computing KL surface (quick mode)...")

    # 计算KL散度
    for i, vector_size in enumerate(vector_sizes):
        print(f"  Size 2^{int(math.log2(vector_size))}: ", end='')

        # 使用固定种子以获得可重现的结果
        seed = 42 + i
        X_vector = create_normalized_vector(size=vector_size, seed=seed, distribution=distribution)

        for j, a in enumerate(a_values):
            try:
                results = analyze_transformation_max_normalized(X_vector, a)
                kl_xy = results['KL_X_Y']
                if not (torch.isnan(torch.tensor(kl_xy)) or torch.isinf(torch.tensor(kl_xy))):
                    Z[j, i] = kl_xy
                    print(".4f", end=' ')
                else:
                    Z[j, i] = float('nan')
                    print("N/A  ", end=' ')
            except Exception:
                Z[j, i] = float('nan')
                print("ERR  ", end=' ')

        print()

    # 创建3D图表
    plt.switch_backend('Agg')
    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(111, projection='3d')

    # 绘制曲面
    surf = ax.plot_surface(X, Y, Z, cmap=cm.viridis, alpha=0.8,
                          linewidth=0, antialiased=True, shade=True)

    # 添加颜色条
    cbar = fig.colorbar(surf, ax=ax, shrink=0.6, aspect=10)
    cbar.set_label('KL Divergence (X||Y)', fontsize=12)

    # 设置标签
    ax.set_xlabel('Vector Size (log₂ scale)', fontsize=12, labelpad=10)
    ax.set_ylabel('Parameter a', fontsize=12, labelpad=10)
    ax.set_zlabel('KL Divergence', fontsize=12, labelpad=5)

    # 设置标题
    ax.set_title(f'Quick 3D KL Surface: Max-Normalized WM Transform\nDirect KL values for all parameter combinations',
                fontsize=14, pad=15)

    # 设置X轴刻度
    x_ticks = [math.log2(size) for size in vector_sizes]
    x_labels = [f'2^{int(math.log2(size))}' for size in vector_sizes]
    ax.set_xticks(x_ticks)
    ax.set_xticklabels(x_labels)

    # 设置视角
    ax.view_init(elev=20, azim=45)

    plt.tight_layout()
    plt.savefig("quick_3d_demo.jpg", dpi=300, bbox_inches='tight')
    print("\n✅ Quick 3D demo saved as: quick_3d_demo.jpg")

    # 统计信息
    valid_z = Z[~np.isnan(Z)]
    if len(valid_z) > 0:
        print(".6f")
        print(".6f")
        print(".6f")

    plt.close()

if __name__ == "__main__":
    quick_3d_analysis()