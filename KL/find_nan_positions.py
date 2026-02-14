#!/usr/bin/env python3
"""
找出3D图表中哪些位置产生NaN值
"""

import sys
import os
import torch
import numpy as np
import math

# 添加父目录到路径
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, parent_dir)

from KL.wm_max_normalized import analyze_transformation_max_normalized
from KL.wm_simple import create_normalized_vector

def find_nan_positions():
    """找出产生NaN的所有参数组合"""
    print("🔍 Finding NaN Positions in 3D Parameter Space")
    print("=" * 50)

    # 使用与3D图表相同的参数范围
    vector_sizes = [2**i for i in range(4, 9)]  # [16, 32, 64, 128, 256]
    a_values = torch.arange(2.0, 4.5, 0.1).numpy()
    distribution = 'uniform'

    print(f"Scanning parameter space:")
    print(f"  Vector sizes: {vector_sizes}")
    print(f"  a values: [{a_values.min():.1f}, {a_values.max():.1f}] (step: {a_values[1]-a_values[0]:.1f})")
    print(f"  Total combinations: {len(vector_sizes)} × {len(a_values)} = {len(vector_sizes) * len(a_values)}")
    print()

    nan_positions = []
    valid_positions = []
    error_positions = []

    for i, vector_size in enumerate(vector_sizes):
        log2_size = math.log2(vector_size)
        print(f"Processing vector size 2^{int(log2_size)} ({vector_size})...")

        # 使用固定种子
        seed = 42 + i

        for j, a in enumerate(a_values):
            try:
                X = create_normalized_vector(size=vector_size, seed=seed, distribution=distribution)
                results = analyze_transformation_max_normalized(X, a)

                kl_xy = results['KL_X_Y']

                if torch.isnan(torch.tensor(kl_xy)) or torch.isinf(torch.tensor(kl_xy)):
                    nan_positions.append((vector_size, a, kl_xy))
                    print(".1f")
                else:
                    valid_positions.append((vector_size, a, kl_xy))
                    print(".1f")

            except Exception as e:
                error_positions.append((vector_size, a, str(e)))
                print(".1f")

        print()

    # 统计结果
    print("📊 Summary:")
    print(f"  Total positions: {len(vector_sizes) * len(a_values)}")
    print(f"  Valid positions: {len(valid_positions)}")
    print(f"  NaN positions: {len(nan_positions)}")
    print(f"  Error positions: {len(error_positions)}")
    print()

    if nan_positions:
        print("🔴 NaN Positions:")
        for size, a, kl in nan_positions:
            print(".1f")

    if error_positions:
        print("❌ Error Positions:")
        for size, a, error in error_positions:
            print(".1f")

    if valid_positions:
        print("✅ Valid Positions (first 10):")
        for size, a, kl in valid_positions[:10]:
            print(".1f")
        if len(valid_positions) > 10:
            print(f"  ... and {len(valid_positions) - 10} more")

    return nan_positions, valid_positions, error_positions

def analyze_single_nan_case():
    """分析一个具体的NaN案例"""
    print("\n🔬 Analyzing Single NaN Case")
    print("=" * 30)

    # 选择一个可能产生NaN的参数组合
    vector_size = 256  # 大的向量
    a = 4.4  # 大的a值
    seed = 42 + 4  # 对应的种子
    distribution = 'uniform'

    print(f"Analyzing: size={vector_size}, a={a}, seed={seed}")

    try:
        X = create_normalized_vector(size=vector_size, seed=seed, distribution=distribution)
        print(f"X stats: min={X.min():.6f}, max={X.max():.6f}, sum={X.sum():.6f}")

        # 手动检查每一步
        X_max = torch.max(X)
        print(f"X_max: {X_max:.6f}")

        from KL.wm_max_normalized import w_max_normalized, f, m_max_normalized

        w_X = w_max_normalized(X, a, X_max)
        print(f"w(X) stats: min={w_X.min():.6e}, max={w_X.max():.6e}")
        print(f"w(X) inf count: {torch.sum(torch.isinf(w_X)).item()}")
        print(f"w(X) nan count: {torch.sum(torch.isnan(w_X)).item()}")

        f_w_X = f(w_X, a)
        print(f"f(w(X)) stats: min={f_w_X.min():.0f}, max={f_w_X.max():.0f}")
        print(f"f(w(X)) <=0 count: {torch.sum(f_w_X <= 0).item()}")
        print(f"f(w(X)) >65535 count: {torch.sum(f_w_X > 65535).item()}")

        Y = m_max_normalized(f_w_X, a, X_max)
        print(f"Y stats: min={Y.min():.6f}, max={Y.max():.6f}, sum={Y.sum():.6f}")
        print(f"Y inf count: {torch.sum(torch.isinf(Y)).item()}")
        print(f"Y nan count: {torch.sum(torch.isnan(Y)).item()}")

        # 检查X和Y的差异
        diff = torch.abs(X - Y)
        print(f"X-Y difference: max={diff.max():.6e}, mean={diff.mean():.6e}")

        # 计算KL散度
        results = analyze_transformation_max_normalized(X, a)
        print(f"KL(X||Y): {results['KL_X_Y']}")
        print(f"KL(Y||X): {results['KL_Y_X']}")

        # 检查是否是浮点精度问题
        if abs(results['KL_X_Y']) < 1e-10:
            print("Note: KL value is effectively zero (floating point precision)")

    except Exception as e:
        print(f"Exception occurred: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    nan_pos, valid_pos, error_pos = find_nan_positions()
    analyze_single_nan_case()

    print("\n💡 Conclusions:")
    if not nan_pos and not error_pos:
        print("✅ No NaN or error positions found! All parameter combinations work correctly.")
        print("The 'missing values' in 3D plot might be due to:")
        print("  1. Visualization artifacts in matplotlib")
        print("  2. Very small KL values being treated as zero")
        print("  3. Colorbar scaling issues")
    else:
        print(f"❌ Found {len(nan_pos)} NaN positions and {len(error_pos)} error positions")
        print("These need to be investigated and fixed.")