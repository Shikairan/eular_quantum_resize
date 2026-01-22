"""
调试 HRH 门的数学计算
"""

import torch
import math
import numpy as np

def analyze_hrh_matrix_detailed():
    """详细分析 HRH 矩阵计算"""
    print("=== HRH 矩阵详细分析 ===")

    phi = math.pi / 4
    print(f"参数 φ = π/4 = {phi:.4f}")
    print()

    # 1. 构造基础矩阵
    # Rz(φ) 矩阵
    exp_neg = torch.exp(torch.tensor(-1j * phi / 2, dtype=torch.complex128))
    exp_pos = torch.exp(torch.tensor(1j * phi / 2, dtype=torch.complex128))
    rz_mat = torch.tensor([
        [exp_neg, 0.0],
        [0.0, exp_pos]
    ], dtype=torch.complex128)

    print("Rz(φ) 矩阵:")
    print(rz_mat)
    print()

    # H 矩阵
    h_mat = torch.tensor([
        [1.0, 1.0],
        [1.0, -1.0]
    ], dtype=torch.complex128) / math.sqrt(2)

    print("H 矩阵:")
    print(h_mat)
    print()

    # 2. 计算 H * Rz * H
    # 先计算 Rz * H
    rz_h = torch.matmul(rz_mat, h_mat)
    print("Rz * H:")
    print(rz_h)
    print()

    # 再计算 H * (Rz * H)
    hrh_mat = torch.matmul(h_mat, rz_h)
    print("H * Rz * H (HRH):")
    print(hrh_mat)
    print()

    # 3. 比较 Rx(φ)
    c, s = math.cos(phi / 2), math.sin(phi / 2)
    rx_mat = torch.tensor([
        [c, -1j * s],
        [-1j * s, c]
    ], dtype=torch.complex128)

    print("Rx(φ) 矩阵:")
    print(rx_mat)
    print()

    # 检查是否相等
    diff = torch.abs(hrh_mat - rx_mat)
    max_diff = torch.max(diff).item()
    print(f"HRH vs Rx 最大差异: {max_diff:.2e}")

    if max_diff < 1e-10:
        print("✅ 理论验证成功：H·Rz(φ)·H = Rx(φ)")
    else:
        print("❌ 理论验证失败")

    print()
    print("数学证明:")
    print("Rz(φ) = diag(e^(-iφ/2), e^(iφ/2))")
    print("H = (1/√2) * [[1, 1], [1, -1]]")
    print("H·Rz(φ)·H 的计算结果应该等于 Rx(φ)")
    print()

def test_simple_case():
    """测试简单情况"""
    print("=== 简单测试案例 ===")

    # 使用 |0⟩ 状态
    state_0 = torch.tensor([1.0, 0.0], dtype=torch.complex128)
    phi = math.pi / 4

    print("初始状态 |0⟩:")
    print(state_0)
    print()

    # 计算理论上的 HRH 变换
    c, s = math.cos(phi / 2), math.sin(phi / 2)
    rx_matrix = torch.tensor([
        [c, -1j * s],
        [-1j * s, c]
    ], dtype=torch.complex128)

    result_theory = torch.matmul(rx_matrix, state_0)
    print("理论 Rx 变换结果:")
    print(result_theory)
    print()

    # 计算 H·Rz·H 变换
    # Rz
    exp_neg = torch.exp(torch.tensor(-1j * phi / 2, dtype=torch.complex128))
    exp_pos = torch.exp(torch.tensor(1j * phi / 2, dtype=torch.complex128))
    rz_mat = torch.tensor([
        [exp_neg, 0.0],
        [0.0, exp_pos]
    ], dtype=torch.complex128)

    # H
    h_mat = torch.tensor([
        [1.0, 1.0],
        [1.0, -1.0]
    ], dtype=torch.complex128) / math.sqrt(2)

    # H·Rz·H
    hrh_mat = torch.matmul(h_mat, torch.matmul(rz_mat, h_mat))

    result_hrh = torch.matmul(hrh_mat, state_0)
    print("HRH 变换结果:")
    print(result_hrh)
    print()

    diff = torch.abs(result_theory - result_hrh).max().item()
    print(f"理论 Rx vs HRH 差异: {diff:.2e}")

if __name__ == "__main__":
    analyze_hrh_matrix_detailed()
    print("\n" + "="*50 + "\n")
    test_simple_case()
