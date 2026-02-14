#!/usr/bin/env python3
"""
调试KL散度计算，找出为什么有些位置没有数值
"""

import sys
import os
import torch

# 添加父目录到路径
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, parent_dir)

from KL.wm_max_normalized import (
    analyze_transformation_max_normalized,
    w_max_normalized,
    m_max_normalized,
    f
)
from KL.wm_simple import create_normalized_vector

def debug_single_transformation():
    """调试单个变换过程"""
    print("🔍 Debugging Single Transformation")
    print("=" * 40)

    # 使用与3D图表相同的参数
    vector_size = 16
    a = 2.0
    seed = 42
    distribution = 'uniform'

    print(f"Parameters: size={vector_size}, a={a}, seed={seed}, dist={distribution}")

    # 创建测试向量
    X = create_normalized_vector(size=vector_size, seed=seed, distribution=distribution)
    print(f"Input vector X: min={X.min():.4f}, max={X.max():.4f}, sum={X.sum():.4f}")

    # 手动执行变换步骤
    X_max = torch.max(X)
    print(f"X_max: {X_max:.4f}")

    # 步骤1: w(x) = exp(a * (x - X_max))
    w_X = w_max_normalized(X, a, X_max)
    print(f"w(X): min={w_X.min():.4f}, max={w_X.max():.4f}")

    # 检查是否有inf或nan
    if torch.any(torch.isinf(w_X)):
        print("⚠️  WARNING: w(X) contains inf values!")
    if torch.any(torch.isnan(w_X)):
        print("⚠️  WARNING: w(X) contains nan values!")

    # 步骤2: f(x) = round(65534 * w(x)) + 1
    f_w_X = f(w_X, a)
    print(f"f(w(X)): min={f_w_X.min():.0f}, max={f_w_X.max():.0f}")

    # 检查量化结果
    if torch.any(f_w_X <= 0):
        print("⚠️  WARNING: f(w(X)) contains non-positive values!")
    if torch.any(torch.isinf(f_w_X)):
        print("⚠️  WARNING: f(w(X)) contains inf values!")

    # 步骤3: m(x) = X_max + (1/a) * log(f(w(x))/65535)
    Y = m_max_normalized(f_w_X, a, X_max)
    print(f"Y: min={Y.min():.4f}, max={Y.max():.4f}, sum={Y.sum():.4f}")

    # 检查逆变换结果
    if torch.any(torch.isinf(Y)):
        print("⚠️  WARNING: Y contains inf values!")
    if torch.any(torch.isnan(Y)):
        print("⚠️  WARNING: Y contains nan values!")

    # 计算KL散度
    results = analyze_transformation_max_normalized(X, a)
    print(f"KL(X||Y): {results['KL_X_Y']:.6f}")
    print(f"KL(Y||X): {results['KL_Y_X']:.6f}")

    # 检查KL计算是否有问题
    if torch.isnan(torch.tensor(results['KL_X_Y'])) or torch.isinf(torch.tensor(results['KL_X_Y'])):
        print("⚠️  WARNING: KL(X||Y) is NaN or inf!")

    return results

def debug_multiple_a_values():
    """调试多个a值的情况"""
    print("\n🔍 Debugging Multiple a Values")
    print("=" * 35)

    vector_size = 16
    a_values = [2.0, 2.5, 3.0, 3.5, 4.0, 4.4]
    seed = 42
    distribution = 'uniform'

    X = create_normalized_vector(size=vector_size, seed=seed, distribution=distribution)
    X_max = torch.max(X)

    print(f"Vector size: {vector_size}, X_max: {X_max:.4f}")
    print("a_value | KL(X||Y) | w(X)_range | f(w(X))_range | Y_range | warnings")
    print("-" * 80)

    for a in a_values:
        try:
            results = analyze_transformation_max_normalized(X, a)

            # 检查中间结果
            w_X = results['w_X']
            f_w_X = results['f_w_X']
            Y = results['Y']

            warnings = []
            if torch.any(torch.isinf(w_X)): warnings.append("w_inf")
            if torch.any(torch.isnan(w_X)): warnings.append("w_nan")
            if torch.any(f_w_X <= 0): warnings.append("f<=0")
            if torch.any(torch.isinf(f_w_X)): warnings.append("f_inf")
            if torch.any(torch.isinf(Y)): warnings.append("Y_inf")
            if torch.any(torch.isnan(Y)): warnings.append("Y_nan")
            if torch.isnan(torch.tensor(results['KL_X_Y'])): warnings.append("KL_nan")

            w_range = ".2f"
            f_range = ".0f"
            y_range = ".4f"

            warning_str = ", ".join(warnings) if warnings else "none"

            print(".1f")

        except Exception as e:
            print(".1f")

def debug_problematic_a_values():
    """专门调试有问题的a值"""
    print("\n🔍 Debugging Problematic a Values")
    print("=" * 38)

    vector_size = 16
    problematic_a_values = [4.2, 4.3, 4.4, 4.5, 5.0]
    seed = 42
    distribution = 'uniform'

    X = create_normalized_vector(size=vector_size, seed=seed, distribution=distribution)
    X_max = torch.max(X)

    print(f"Vector size: {vector_size}, X_max: {X_max:.4f}")

    for a in problematic_a_values:
        print(f"\n--- Testing a = {a} ---")
        try:
            # 手动检查每一步
            w_X = w_max_normalized(X, a, X_max)
            print(f"  w(X): min={w_X.min():.2e}, max={w_X.max():.2e}")

            if torch.any(torch.isinf(w_X)):
                print(f"  ❌ w(X) contains inf values! Max exp input: {a * (X - X_max).max():.2f}")
                continue

            f_w_X = f(w_X, a)
            print(f"  f(w(X)): min={f_w_X.min():.0f}, max={f_w_X.max():.0f}")

            if torch.any(f_w_X <= 0) or torch.any(f_w_X > 65535):
                print(f"  ❌ f(w(X)) out of valid range [1, 65535]!")
                continue

            Y = m_max_normalized(f_w_X, a, X_max)
            print(f"  Y: min={Y.min():.4f}, max={Y.max():.4f}")

            if torch.any(torch.isinf(Y)) or torch.any(torch.isnan(Y)):
                print(f"  ❌ Y contains inf or NaN values!")
                continue

            # 只有当所有步骤都成功时才计算KL
            results = analyze_transformation_max_normalized(X, a)
            print(f"  ✅ KL(X||Y): {results['KL_X_Y']:.6f}")

        except Exception as e:
            print(f"  ❌ Exception: {e}")

def debug_constants_issue():
    """调试常量问题"""
    print("\n🔍 Debugging Constants Issue")
    print("=" * 32)

    # 比较wm_max_normalized和wm_simple的常量使用
    print("wm_max_normalized.py constants:")
    print("  f(x) uses: round(65534 * x) + 1")
    print("  m(x) uses: log(x / 65535)")
    print()
    print("wm_simple.py constants:")
    print("  f(x) uses: round(255 * w(x))")
    print("  m(x) uses: log(x / 255)")
    print()
    print("Issue: Inconsistent scaling constants between forward and inverse transforms!")
    print("This can cause numerical issues when the transformed values exceed expected ranges.")

if __name__ == "__main__":
    debug_single_transformation()
    debug_multiple_a_values()
    debug_problematic_a_values()
    debug_constants_issue()

    print("\n💡 Root Cause Analysis:")
    print("1. Large a values (≥4.2) cause exponential overflow in w(x) = exp(a * (x - X_max))")
    print("2. The max exponent becomes: a * (X - X_max).max() ≈ 4.4 * 0 = 0 (OK) but wait...")
    print("3. Actually, for normalized vectors, (x - X_max) is always ≤ 0, so exp() should not overflow")
    print("4. The issue might be in the KL divergence calculation for perfect reconstructions")
    print("5. When X ≈ Y (perfect reconstruction), KL(X||Y) should be 0, but numerical precision might cause issues")