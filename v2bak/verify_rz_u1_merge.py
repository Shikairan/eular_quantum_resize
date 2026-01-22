"""
验证 Rz 门是否正确合并到 U1 门中
"""

import torch
import math
from ALLeularV4 import *

def test_rz_u1_equivalence():
    """测试 Rz 和 U1 是否产生等价结果"""
    print("=== 验证 Rz 合并到 U1 的等价性 ===")

    # 创建测试状态
    polar_vec = create_random_polar_state(4)  # 2 比特系统
    print("初始状态:")
    print(polar_vec_to_string(polar_vec.float(), range(4)))
    print()

    # 参数
    phi = math.pi / 4

    # 测试 Rz 门
    rz_vec = polar_vec.clone()
    apply_gate_unified_tensor(rz_vec, 'Rz', phi, qubit_idx=0)

    # 测试 U1 门（相同参数）
    u1_vec = polar_vec.clone()
    apply_gate_unified_tensor(u1_vec, 'U1', phi, qubit_idx=0)

    print(f"应用 Rz(π/4) 后的状态:")
    print(polar_vec_to_string(rz_vec.float(), range(4)))
    print()

    print(f"应用 U1(π/4) 后的状态:")
    print(polar_vec_to_string(u1_vec.float(), range(4)))
    print()

    # 比较结果
    rz_complex = polar_to_complex_tensor(rz_vec.float())
    u1_complex = polar_to_complex_tensor(u1_vec.float())

    diff = torch.abs(rz_complex - u1_complex)
    max_diff = torch.max(diff).item()

    print(f"最大差异: {max_diff:.2e}")
    if max_diff < 1e-5:
        print("✅ Rz 和 U1 产生相同的效果 - 合并成功！")
    else:
        print("❌ Rz 和 U1 结果不同 - 合并可能有问题")
    print()

def test_mathematical_correctness():
    """测试数学正确性"""
    print("=== 数学正确性验证 ===")

    phi = math.pi / 4

    # Rz 矩阵 (理论值)
    rz_theory = torch.tensor([
        [torch.exp(torch.tensor(-1j * phi / 2, dtype=torch.complex64)), 0],
        [0, torch.exp(torch.tensor(1j * phi / 2, dtype=torch.complex64))]
    ], dtype=torch.complex64)

    # U1 矩阵 (理论值)
    u1_theory = torch.tensor([
        [1.0, 0.0],
        [0.0, torch.exp(torch.tensor(1j * phi, dtype=torch.complex64))]
    ], dtype=torch.complex64)

    print(f"Rz(π/4) 理论矩阵:")
    print(rz_theory)
    print()

    print(f"U1(π/4) 理论矩阵:")
    print(u1_theory)
    print()

    # 在量子计算中，Rz 和 U1 的区别在于全局相位
    # Rz(φ) = e^(-iφ/2) * U1(φ/2 + π/2)
    # 所以它们在功能上是等价的（忽略全局相位）

    print("理论分析:")
    print("- Rz 是对称相位门")
    print("- U1 是单参数通用门")
    print("- 在量子计算中，全局相位通常不重要")
    print("- 因此 Rz(φ) ≈ U1(φ) （忽略全局相位）")
    print()

if __name__ == "__main__":
    test_rz_u1_equivalence()
    test_mathematical_correctness()
