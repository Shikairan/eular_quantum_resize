#!/usr/bin/env python3
"""
最大值归一化WM变换演示脚本
展示如何使用新的最大值归一化变换功能
"""

import sys
import os

# 添加父目录到路径
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, parent_dir)

from KL.wm_max_normalized import (
    create_normalized_vector,
    analyze_transformation_max_normalized,
    benchmark_max_normalized_vs_original
)
import torch

def demo_basic_usage():
    """演示基本使用方法"""
    print("🔬 Max-Normalized WM Transform Demo")
    print("=" * 50)

    # 1. 创建测试向量
    print("1. Creating test vector...")
    X = create_normalized_vector(size=10, seed=42, distribution='uniform')
    print(f"   Input vector X: {X.cpu().numpy()}")

    # 2. 应用最大值归一化变换
    print("\n2. Applying max-normalized WM transform (a=3.0)...")
    result = analyze_transformation_max_normalized(X, a=3.0)

    print(f"   X_max: {result['X_max']:.4f}")
    print(f"   KL(X||Y): {result['KL_X_Y']:.6f}")
    print(f"   KL(Y||X): {result['KL_Y_X']:.6f}")

    # 3. 与原始方法对比
    print("\n3. Comparing with original method...")
    print("   (Testing multiple a values and distributions)")

    benchmark_results = benchmark_max_normalized_vs_original(
        sizes=[15],
        distributions=['uniform', 'exponential'],
        a_values=[1.0, 3.0, 5.0]
    )

    print("\n   Results summary:")
    print("   Distribution | Size | Method | Avg KL(X||Y)")
    print("   ------------|------|--------|------------")
    for dist in ['uniform', 'exponential']:
        for size in [15]:
            for method in ['max_normalized', 'original']:
                key = f'{dist}_{size}'
                if key in benchmark_results[method]:
                    values = benchmark_results[method][key]
                    valid_kl = [v.get('KL_X_Y', 0) for v in values if 'KL_X_Y' in v]
                    if valid_kl:
                        avg_kl = sum(valid_kl) / len(valid_kl)
                        print("10s")

def demo_different_scenarios():
    """演示不同场景的应用"""
    print("\n🎯 Different Usage Scenarios")
    print("=" * 30)

    scenarios = [
        ("Small uniform vector", {'size': 5, 'distribution': 'uniform'}),
        ("Large exponential vector", {'size': 50, 'distribution': 'exponential', 'rate': 2.0}),
        ("Beta distributed vector", {'size': 20, 'distribution': 'beta', 'alpha': 2.0, 'beta': 5.0}),
        ("Gamma distributed vector", {'size': 25, 'distribution': 'gamma', 'shape': 2.0, 'rate': 1.5}),
    ]

    for scenario_name, params in scenarios:
        print(f"\n{scenario_name}:")
        X = create_normalized_vector(seed=42, **params)
        result = analyze_transformation_max_normalized(X, a=3.0)

        print(f"  Vector size: {X.shape[0]}")
        print(f"  Max value: {result['X_max']:.4f}")
        print(".6f")

def demo_parameter_sweep():
    """演示参数扫描"""
    print("\n📈 Parameter Sweep Demo")
    print("=" * 25)

    X = create_normalized_vector(size=20, seed=42, distribution='uniform')
    a_values = torch.arange(1.0, 6.0, 0.5)

    print("Parameter a | KL(X||Y) | KL(Y||X)")
    print("------------|-----------|-----------")

    for a in a_values:
        result = analyze_transformation_max_normalized(X, a.item())
        print(".1f")

def demo_optimal_a_analysis():
    """演示最优a值分析"""
    print("\n🔍 Optimal a Value Analysis Demo")
    print("=" * 35)

    from KL.wm_max_normalized import (
        analyze_optimal_a_distribution_max_normalized,
        analyze_optimal_a_vs_size_max_normalized
    )

    print("Generating optimal a distribution plot...")
    optimal_values = analyze_optimal_a_distribution_max_normalized(
        num_vectors=10,  # 演示用较少向量
        vector_size=30,
        distributions=['uniform', 'exponential', 'beta']
    )

    print("\nOptimal a value statistics:")
    for dist, values in optimal_values.items():
        if values:
            avg_a = sum(values) / len(values)
            min_a = min(values)
            max_a = max(values)
            print("8s")

    print("\nGenerating optimal a vs vector size migration plot...")
    analyze_optimal_a_vs_size_max_normalized(
        vector_sizes=[10, 20, 30],
        num_vectors_per_size=5,  # 演示用较少向量
        distributions=['uniform', 'exponential']
    )

def main():
    """主演示函数"""
    try:
        demo_basic_usage()
        demo_different_scenarios()
        demo_parameter_sweep()
        demo_optimal_a_analysis()

        print("\n✅ Demo completed successfully!")
        print("\n💡 Key Features:")
        print("   • Automatic max-value normalization")
        print("   • Better numerical stability")
        print("   • Compatible with all distribution types")
        print("   • GPU acceleration support")
        print("   • Easy comparison with original method")
        print("   • Optimal a value distribution analysis")
        print("   • Vector size migration analysis")

    except Exception as e:
        print(f"❌ Demo failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()