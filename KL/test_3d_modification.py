#!/usr/bin/env python3
"""
测试3D图表修改是否成功的简单脚本
"""

import sys
import os
import torch
import numpy as np
import math
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# 添加父目录到路径
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, parent_dir)

from KL.wm_max_normalized import analyze_transformation_max_normalized
from KL.wm_simple import create_normalized_vector

def test_3d_modification():
    """测试3D图表修改"""
    print("🧪 Testing 3D Plot Modification")
    print("=" * 35)

    # 简单的测试参数
    vector_sizes = [16, 32]  # 只有两个向量大小用于快速测试
    a_values = torch.arange(2.5, 3.5, 0.5).numpy()  # 只有两个a值
    distribution = 'uniform'

    print(f"Test parameters:")
    print(f"  - Vector sizes: {vector_sizes}")
    print(f"  - a values: {a_values}")
    print()

    # 创建网格
    X, Y = np.meshgrid(
        [math.log2(size) for size in vector_sizes],
        a_values
    )
    Z = np.zeros_like(X, dtype=float)

    print("Testing modified computation (direct KL values):")

    # 测试修改后的计算方式
    for i, vector_size in enumerate(vector_sizes):
        print(f"  Vector size 2^{int(math.log2(vector_size))}: ", end='')

        # 使用固定种子
        seed = 42 + i
        X_vector = create_normalized_vector(size=vector_size, seed=seed, distribution=distribution)

        for j, a in enumerate(a_values):
            try:
                results = analyze_transformation_max_normalized(X_vector, a)
                kl_xy = results['KL_X_Y']

                if not (torch.isnan(torch.tensor(kl_xy)) or torch.isinf(torch.tensor(kl_xy))):
                    Z[j, i] = kl_xy  # 直接存储KL值
                    print(".4f", end=' ')
                else:
                    Z[j, i] = float('nan')
                    print("NaN  ", end=' ')

            except Exception as e:
                Z[j, i] = float('nan')
                print("ERR  ", end=' ')

        print()

    # 创建简单的3D图表来验证
    plt.switch_backend('Agg')
    fig = plt.figure(figsize=(10, 6))
    ax = fig.add_subplot(111, projection='3d')

    # 只绘制有效数据点
    valid_mask = ~np.isnan(Z)
    if np.any(valid_mask):
        ax.scatter(X[valid_mask], Y[valid_mask], Z[valid_mask],
                  c=Z[valid_mask], cmap='viridis', s=100, alpha=0.8)

        # 添加数值标签
        for i in range(len(vector_sizes)):
            for j in range(len(a_values)):
                if not np.isnan(Z[j, i]):
                    ax.text(X[j, i], Y[j, i], Z[j, i],
                           '.4f', fontsize=10,
                           bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8))

    # 设置标签
    ax.set_xlabel('Vector Size (log₂ scale)')
    ax.set_ylabel('Parameter a')
    ax.set_zlabel('KL Divergence')

    # 设置标题
    ax.set_title('3D Modification Test: Direct KL Values\n(No averaging, all parameter combinations)')

    # 设置X轴刻度
    ax.set_xticks([math.log2(size) for size in vector_sizes])
    ax.set_xticklabels([f'2^{int(math.log2(size))}' for size in vector_sizes])

    plt.tight_layout()
    plt.savefig("test_3d_modification.jpg", dpi=300, bbox_inches='tight')
    print("\n✅ Test plot saved as: test_3d_modification.jpg")
    print("If you see individual points with labels, the modification is successful!")

    plt.close()

if __name__ == "__main__":
    test_3d_modification()