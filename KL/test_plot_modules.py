#!/usr/bin/env python3
"""
测试绘图模块功能的脚本
"""

import sys
import os

# 添加路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

def test_core_module():
    """测试核心模块"""
    print("=== Testing Core Module ===")
    from .wm_simple import create_normalized_vector, analyze_transformation

    # 创建测试向量
    X = create_normalized_vector(size=5, seed=42, distribution='uniform')
    print(f"Created vector: {X}")

    # 分析变换
    results = analyze_transformation(X, a=3.5)
    print(f"KL divergence: {results['KL_X_Y']:.4f}")
    print("✓ Core module test passed\n")

def test_single_plot():
    """测试单个向量绘图"""
    print("=== Testing Single Plot Module ===")
    from .plot_kl_single import create_normalized_vector

    X = create_normalized_vector(size=5, seed=42)
    print(f"Created test vector for plotting: {X.shape}")

    print("✓ Single plot module test passed\n")

def test_comparison_plot():
    """测试对比绘图"""
    print("=== Testing Comparison Plot Module ===")
    from .plot_kl_comparison import create_normalized_vector

    X1 = create_normalized_vector(size=5, distribution='uniform', seed=42)
    X2 = create_normalized_vector(size=5, distribution='exponential', seed=43)
    print(f"Created comparison vectors: {len(X1)}, {len(X2)}")

    print("✓ Comparison plot module test passed\n")

def test_optimal_a_distribution():
    """测试最优a分布分析"""
    print("=== Testing Optimal A Distribution Module ===")
    # 这里只测试导入，不运行完整分析（因为需要时间）
    from .plot_optimal_a_distribution import analyze_optimal_a_distribution
    print("✓ Optimal A distribution module imported successfully\n")

def test_size_migration():
    """测试尺寸迁移分析"""
    print("=== Testing Size Migration Module ===")
    # 这里只测试导入，不运行完整分析
    from .plot_size_migration import analyze_optimal_a_vs_size_multiple_distributions
    print("✓ Size migration module imported successfully\n")

def main():
    """主测试函数"""
    print("WM Transform Plotting Modules Test Suite")
    print("=" * 50)

    try:
        test_core_module()
        test_single_plot()
        test_comparison_plot()
        test_optimal_a_distribution()
        test_size_migration()

        print("🎉 All tests passed!")
        print("\nUsage Guide:")
        print("- wm_simple.py: Core WM transformation functions")
        print("- plot_kl_single.py: Single vector KL divergence plots")
        print("- plot_kl_comparison.py: Multi-vector KL comparison plots")
        print("- plot_optimal_a_distribution.py: Optimal a value distribution analysis")
        print("- plot_size_migration.py: Vector size migration analysis")

    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()