"""
比较 HRH 门、H+Rz+H 组合和 Rx 门的效果
"""

import torch
import math
import time
from ALLeularV4 import *

def test_hrh_vs_h_rz_h_vs_rx():
    """比较 HRH 门、H+Rz+H 组合和 Rx 门"""
    print("=== HRH 门 vs H+Rz+H 组合 vs Rx 门对比测试 ===")

    # 参数
    phi = math.pi / 4
    print(f"使用参数 φ = π/4 = {phi:.4f}")
    print()

    # 创建测试状态 (2 比特系统)
    polar_vec = create_random_polar_state(4)
    print("初始状态:")
    print(polar_vec_to_string(polar_vec.float(), range(4)))
    print()

    # 测试 1: 新建的 HRH 门
    hrh_vec = polar_vec.clone()
    apply_gate_unified_tensor(hrh_vec, 'HRH', phi, qubit_idx=0)

    print("应用 HRH(π/4) 后的状态:")
    print(polar_vec_to_string(hrh_vec.float(), range(4)))
    print()

    # 测试 2: H + Rz + H 组合
    h_rz_h_vec = polar_vec.clone()
    apply_gate_unified_tensor(h_rz_h_vec, 'H', qubit_idx=0)  # H
    apply_gate_unified_tensor(h_rz_h_vec, 'Rz', phi, qubit_idx=0)  # Rz
    apply_gate_unified_tensor(h_rz_h_vec, 'H', qubit_idx=0)  # H

    print("应用 H + Rz(π/4) + H 组合后的状态:")
    print(polar_vec_to_string(h_rz_h_vec.float(), range(4)))
    print()

    # 测试 3: Rx 门 (理论上应该等价于 HRH)
    rx_vec = polar_vec.clone()
    apply_gate_unified_tensor(rx_vec, 'Rx', phi, qubit_idx=0)

    print("应用 Rx(π/4) 后的状态:")
    print(polar_vec_to_string(rx_vec.float(), range(4)))
    print()

    # 比较结果
    print("=== 结果比较 ===")

    # HRH vs H+Rz+H
    hrh_complex = polar_to_complex_tensor(hrh_vec.float())
    h_rz_h_complex = polar_to_complex_tensor(h_rz_h_vec.float())
    diff_hrh = torch.abs(hrh_complex - h_rz_h_complex)
    max_diff_hrh = torch.max(diff_hrh).item()

    print(f"HRH vs H+Rz+H 最大差异: {max_diff_hrh:.2e}")
    if max_diff_hrh < 1e-5:
        print("✅ HRH 门与 H+Rz+H 组合等价")
    else:
        print("❌ HRH 门与 H+Rz+H 组合不等价")
    print()

    # HRH vs Rx
    rx_complex = polar_to_complex_tensor(rx_vec.float())
    diff_rx = torch.abs(hrh_complex - rx_complex)
    max_diff_rx = torch.max(diff_rx).item()

    print(f"HRH vs Rx 最大差异: {max_diff_rx:.2e}")
    if max_diff_rx < 1e-5:
        print("✅ HRH 门等价于 Rx 门")
    else:
        print("❌ HRH 门不等价于 Rx 门")
    print()

    # 计算归一化
    hrh_norm = torch.sqrt(torch.sum(torch.abs(hrh_complex)**2)).item()
    h_rz_h_norm = torch.sqrt(torch.sum(torch.abs(h_rz_h_complex)**2)).item()
    rx_norm = torch.sqrt(torch.sum(torch.abs(rx_complex)**2)).item()

    print("归一化检查:")
    print(".6f")
    print(".6f")
    print(".6f")

def analyze_hrh_matrix():
    """分析 HRH 矩阵的数学性质"""
    print("=== HRH 矩阵数学分析 ===")

    phi = math.pi / 4

    # 构造 Rz 矩阵
    exp_neg = torch.exp(torch.tensor(-1j * phi / 2, dtype=torch.complex64))
    exp_pos = torch.exp(torch.tensor(1j * phi / 2, dtype=torch.complex64))
    rz_mat = torch.stack([
        torch.stack([exp_neg, torch.tensor(0.0, dtype=torch.complex64)]),
        torch.stack([torch.tensor(0.0, dtype=torch.complex64), exp_pos])
    ])

    # H 矩阵
    h_mat = torch.tensor([[1, 1], [1, -1]], dtype=torch.complex64) / math.sqrt(2)

    # 计算 HRH = H * Rz * H
    hrh_mat = torch.matmul(h_mat, torch.matmul(rz_mat, h_mat))

    print(f"HRH(π/4) 矩阵:")
    print(hrh_mat)
    print()

    # 比较 Rx 矩阵
    c, s = math.cos(phi / 2), math.sin(phi / 2)
    rx_mat = torch.tensor([[c, -1j * s], [-1j * s, c]], dtype=torch.complex64)

    print(f"Rx(π/4) 矩阵:")
    print(rx_mat)
    print()

    # 检查是否相等
    diff = torch.abs(hrh_mat - rx_mat)
    max_diff = torch.max(diff).item()

    print(f"HRH vs Rx 矩阵差异: {max_diff:.2e}")
    if max_diff < 1e-10:
        print("✅ HRH 矩阵精确等于 Rx 矩阵")
        print("✅ 理论证明：H·Rz(φ)·H = Rx(φ)")
    else:
        print("❌ HRH 矩阵不等于 Rx 矩阵")

def performance_comparison():
    """性能对比测试"""
    print("=== 性能对比测试 ===")

    # 创建测试状态
    polar_vec = create_random_polar_state(64)  # 6 比特系统
    n_tests = 1000
    phi = math.pi / 4

    print(f"在 6 比特系统上测试 {n_tests} 次应用")
    print()

    # 测试 HRH 门
    hrh_vec = polar_vec.clone()
    torch.cuda.synchronize()
    start_time = time.time()

    for _ in range(n_tests):
        apply_gate_unified_tensor(hrh_vec, 'HRH', phi, qubit_idx=0)

    torch.cuda.synchronize()
    hrh_time = time.time() - start_time

    # 测试 H+Rz+H 组合
    h_rz_h_vec = polar_vec.clone()
    torch.cuda.synchronize()
    start_time = time.time()

    for _ in range(n_tests):
        apply_gate_unified_tensor(h_rz_h_vec, 'H', qubit_idx=0)
        apply_gate_unified_tensor(h_rz_h_vec, 'Rz', phi, qubit_idx=0)
        apply_gate_unified_tensor(h_rz_h_vec, 'H', qubit_idx=0)

    torch.cuda.synchronize()
    h_rz_h_time = time.time() - start_time

    # 测试 Rx 门
    rx_vec = polar_vec.clone()
    torch.cuda.synchronize()
    start_time = time.time()

    for _ in range(n_tests):
        apply_gate_unified_tensor(rx_vec, 'Rx', phi, qubit_idx=0)

    torch.cuda.synchronize()
    rx_time = time.time() - start_time

    # 结果
    print("性能对比 (1000 次应用):")
    print(f"HRH 门:     {hrh_time:.4f} 秒")
    print(f"H+Rz+H:     {h_rz_h_time:.4f} 秒")
    print(f"Rx 门:       {rx_time:.4f} 秒")
    print()
    print(f"HRH:     {h_rz_h_time/hrh_time:.2f}x")
    print(f"H+Rz+H: 1.00x (基准)")
    print(f"Rx:      {h_rz_h_time/rx_time:.2f}x")
    print()
    print("💡 分析:")
    print("- HRH 门：单次矩阵乘法，最优性能")
    print("- H+Rz+H：三次门应用，额外开销")
    print("- Rx 门：与 HRH 功能等价，性能相当")

if __name__ == "__main__":
    test_hrh_vs_h_rz_h_vs_rx()
    print("\n" + "="*60 + "\n")
    analyze_hrh_matrix()
    print("\n" + "="*60 + "\n")
    performance_comparison()
