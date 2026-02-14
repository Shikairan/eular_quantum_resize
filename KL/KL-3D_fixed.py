#!/usr/bin/env python3
"""
3D可视化KL散度分析 - Max-Normalized WM变换
修复版本：正确处理小数值显示问题
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
import matplotlib.colors as colors

# 添加父目录到路径
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, parent_dir)

from KL.wm_max_normalized import analyze_transformation_max_normalized
from KL.wm_simple import create_normalized_vector

def create_3d_kl_surface_fixed():
    """
    创建3D KL散度曲面图 - 修复版本
    正确处理非常小的KL值显示问题
    """
    print("🎯 Creating 3D KL Divergence Surface Plot (Fixed Version)")
    print("=" * 60)

    # 参数设置
    vector_sizes = [2**i for i in range(7, 24)]  # [16, 32, 64, 128, 256] - 2^4 to 2^8
    a_values = torch.arange(2.0, 4.5, 0.1).numpy()  # a值范围
    distribution = 'uniform'  # 使用的分布

    print(f"Parameters:")
    print(f"  - Vector sizes: {vector_sizes} (powers of 2)")
    print(f"  - a values: [{a_values.min():.1f}, {a_values.max():.1f}] (step: {a_values[1]-a_values[0]:.1f})")
    print(f"  - Distribution: {distribution}")
    print()

    # 创建网格
    X, Y = np.meshgrid(
        [math.log2(size) for size in vector_sizes],  # X轴：log2(向量长度)
        a_values  # Y轴：a值
    )
    Z = np.zeros_like(X, dtype=float)  # Z轴：KL散度

    print("Computing KL divergence surface for all parameter combinations...")

    # 对每个网格点计算KL散度
    for i, vector_size in enumerate(vector_sizes):
        log2_size = math.log2(vector_size)
        print(f"  Processing vector size 2^{int(log2_size)} ({vector_size})...")

        # 为每个向量长度使用固定的种子以获得可重现的结果
        seed = 42 + i  # 固定的种子
        X_vector = create_normalized_vector(size=vector_size, seed=seed, distribution=distribution)

        # 计算每个a值的KL散度
        for j, a in enumerate(a_values):
            try:
                results = analyze_transformation_max_normalized(X_vector, a)
                kl_xy = results['KL_X_Y']

                if not (torch.isnan(torch.tensor(kl_xy)) or torch.isinf(torch.tensor(kl_xy))):
                    Z[j, i] = kl_xy  # 直接存储KL散度值
                else:
                    Z[j, i] = 0#float('nan')

            except Exception as e:
                print(f"    Error at a={a:.1f}: {e}")
                Z[j, i] = 0#float('nan')

    # 创建3D图表
    plt.switch_backend('Agg')
    fig = plt.figure(figsize=(15, 10))
    ax = fig.add_subplot(111, projection='3d')

    # 处理数值范围，避免显示问题
    valid_z = Z[~np.isnan(Z)]
    if len(valid_z) > 0:
        z_min = max(valid_z.min(), 1e-12)  # 设置最小值为1e-12
        z_max = valid_z.max()

        print("\n📊 KL Values Statistics:")
        print(".2e")
        print(".2e")
        print(".2e")
        print(".2e")

        # 根据数值范围选择颜色映射
        if z_max / z_min > 10000:
            # 使用对数尺度
            norm = colors.LogNorm(vmin=z_min, vmax=z_max)
            surf = ax.plot_surface(X, Y, Z, cmap=cm.viridis, norm=norm,
                                 alpha=0.8, linewidth=0, antialiased=True)
            cbar_label = 'KL Divergence (X||Y) [log scale]'
        else:
            # 使用线性尺度
            surf = ax.plot_surface(X, Y, Z, cmap=cm.viridis,
                                 alpha=0.8, linewidth=0, antialiased=True)
            cbar_label = 'KL Divergence (X||Y)'

        # 添加颜色条
        cbar = fig.colorbar(surf, ax=ax, shrink=0.6, aspect=10)
        cbar.set_label(cbar_label, fontsize=12)

        # 如果使用对数尺度，设置颜色条的刻度
        if z_max / z_min > 10000:
            # 设置对数刻度
            import matplotlib.ticker as ticker
            cbar.set_ticks([1e-12, 1e-10, 1e-8, 1e-6, 1e-4])
            cbar.set_ticklabels(['1e-12', '1e-10', '1e-8', '1e-6', '1e-4'])
    else:
        # 如果没有有效值，创建空图表
        surf = ax.plot_surface(X, Y, Z, cmap=cm.viridis, alpha=0.8, linewidth=0, antialiased=True)
        cbar = fig.colorbar(surf, ax=ax, shrink=0.6, aspect=10)
        cbar.set_label('KL Divergence (X||Y)', fontsize=12)

    # 设置轴标签
    ax.set_xlabel('Vector Size (log₂ scale)', fontsize=14, labelpad=15)
    ax.set_ylabel('Parameter a', fontsize=14, labelpad=15)
    ax.set_zlabel('KL Divergence', fontsize=14, labelpad=10)

    # 设置标题
    title = (f'Max-Normalized WM Transform: 3D KL Divergence Surface\n'
             f'{distribution.title()} Distribution - Fixed Display')
    ax.set_title(title, fontsize=16, pad=20)

    # 设置X轴刻度标签
    x_ticks = [math.log2(size) for size in vector_sizes]
    x_labels = [f'2^{int(math.log2(size))}' for size in vector_sizes]
    ax.set_xticks(x_ticks)
    ax.set_xticklabels(x_labels)

    # 设置视角
    ax.view_init(elev=25, azim=45)

    # 添加网格
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig("KL-3D_surface_fixed.jpg", dpi=300, bbox_inches='tight')
    print("\n✅ Fixed 3D surface plot saved as: KL-3D_surface_fixed.jpg")

    plt.close()

    # 打印统计信息
    print("\n📊 Surface Statistics:")
    valid_z = Z[~np.isnan(Z)]
    if len(valid_z) > 0:
        print(f"  Min KL: {valid_z.min():.2e}")
        print(f"  Max KL: {valid_z.max():.2e}")
        print(f"  Mean KL: {valid_z.mean():.2e}")
        print(".2e")

        # 找到最小KL的位置
        min_idx = np.unravel_index(np.nanargmin(Z), Z.shape)
        min_log2_size = X[min_idx]
        min_a = Y[min_idx]
        min_kl = Z[min_idx]
        print(".3f")
    else:
        print("  No valid KL values found!")

def main():
    """主函数"""
    print("🗠 3D KL Divergence Analysis - Max-Normalized WM Transform (Fixed)")
    print("=" * 75)

    try:
        create_3d_kl_surface_fixed()

        print("\n✅ Fixed 3D plot generated successfully!")
        print("\n📁 Generated file:")
        print("   - KL-3D_surface_fixed.jpg (3D surface plot with proper scaling)")

        print("\n🔧 Fixes applied:")
        print("   • Added logarithmic color scaling for very small KL values")
        print("   • Set minimum value threshold to avoid display issues")
        print("   • Improved colorbar ticks for log scale")
        print("   • Better handling of numerical precision issues")

    except Exception as e:
        print(f"❌ 3D plot generation failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
