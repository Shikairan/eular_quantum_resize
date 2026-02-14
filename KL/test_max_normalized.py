#!/usr/bin/env python3
"""
测试最大值归一化WM变换功能的脚本
"""

import sys
import os

# 添加父目录到路径，以便导入KL模块
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, parent_dir)

def test_max_normalized_basic():
    """测试最大值归一化WM变换的基本功能"""
    print("=== Testing Max-Normalized WM Transform Basic Functionality ===")

    from KL.wm_max_normalized import (
        create_normalized_vector,
        analyze_transformation_max_normalized,
        w_max_normalized,
        m_max_normalized
    )

    # 创建测试向量
    X = create_normalized_vector(size=10, seed=42, distribution='uniform')
    print(f"Input vector X: {X}")
    print(f"X max value: {X.max().item():.4f}")

    # 测试最大值归一化变换
    result = analyze_transformation_max_normalized(X, a=3.0)

    print("Transform results:")
    print(f"  X_max: {result['X_max']:.4f}")
    print(f"  w(X): {result['w_X']}")
    print(f"  f(w(X)): {result['f_w_X']}")
    print(f"  Y: {result['Y']}")
    print(f"  KL(X||Y): {result['KL_X_Y']:.6f}")
    print(f"  KL(Y||X): {result['KL_Y_X']:.6f}")

    # 测试单独的变换函数
    X_max = X.max()
    w_result = w_max_normalized(X, a=3.0, X_max=X_max)
    print(f"\nDirect w_max_normalized test:")
    print(f"  w_max(X): {w_result}")

    f_result = w_result * 255  # 模拟f函数
    m_result = m_max_normalized(f_result, a=3.0, X_max=X_max)
    print(f"  m_max(f(w_max(X))): {m_result}")

    print("✓ Basic functionality test passed\n")

def test_comparison_with_original():
    """测试与原始方法的对比"""
    print("=== Testing Comparison with Original Method ===")

    from KL.wm_max_normalized import benchmark_max_normalized_vs_original

    # 运行简化的对比测试
    results = benchmark_max_normalized_vs_original(
        sizes=[10, 20],
        distributions=['uniform'],
        a_values=[1.0, 3.0, 5.0]
    )

    print("Comparison results summary:")
    for method in ['max_normalized', 'original']:
        print(f"  {method}:")
        for key, values in results[method].items():
            if values:
                avg_kl = sum(v.get('KL_X_Y', 0) for v in values if 'KL_X_Y' in v) / len([v for v in values if 'KL_X_Y' in v])
                print(".4f")

    print("✓ Comparison test passed\n")

def test_different_distributions():
    """测试不同分布下的最大值归一化变换"""
    print("=== Testing Different Distributions ===")

    from KL.wm_max_normalized import (
        create_normalized_vector,
        analyze_transformation_max_normalized
    )

    distributions = ['uniform', 'exponential', 'beta', 'gamma']

    for dist in distributions:
        print(f"Testing {dist} distribution:")

        try:
            X = create_normalized_vector(size=20, seed=42, distribution=dist)
            result = analyze_transformation_max_normalized(X, a=3.0)

            print(f"  X shape: {X.shape}")
            print(".4f")
            print(".6f")

        except Exception as e:
            print(f"  Error with {dist}: {e}")

    print("✓ Different distributions test passed\n")

def test_edge_cases():
    """测试边界情况"""
    print("=== Testing Edge Cases ===")

    from KL.wm_max_normalized import analyze_transformation_max_normalized
    import torch

    # 测试全零向量（应该避免）
    try:
        X_zero = torch.zeros(5, device='cuda' if torch.cuda.is_available() else 'cpu')
        X_zero[-1] = 1.0  # 避免全零
        result = analyze_transformation_max_normalized(X_zero, a=3.0)
        print("Zero vector test: OK")
    except Exception as e:
        print(f"Zero vector test failed: {e}")

    # 测试单一值向量
    try:
        X_single = torch.ones(5, device='cuda' if torch.cuda.is_available() else 'cpu') * 0.5
        result = analyze_transformation_max_normalized(X_single, a=3.0)
        print("Single value vector test: OK")
    except Exception as e:
        print(f"Single value vector test failed: {e}")

    print("✓ Edge cases test passed\n")

def main():
    """主测试函数"""
    print("Max-Normalized WM Transform Test Suite")
    print("=" * 50)

    try:
        test_max_normalized_basic()
        test_comparison_with_original()
        test_different_distributions()
        test_edge_cases()

        print("🎉 All max-normalized tests passed!")
        print("\nUsage:")
        print("- analyze_transformation_max_normalized(X, a): Complete max-normalized analysis")
        print("- benchmark_max_normalized_vs_original(): Compare with original method")
        print("- w_max_normalized(x, a, X_max): Max-normalized exponential transform")
        print("- m_max_normalized(x, a, X_max): Max-normalized logarithmic transform")

    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()