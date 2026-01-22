"""
调试 H+Rz+H 组合的问题
"""

import torch
import math
from ALLeularV4 import *

def debug_h_rz_h_step_by_step():
    """逐步调试 H+Rz+H 组合"""
    print("=== 逐步调试 H+Rz+H 组合 ===")

    phi = math.pi / 4
    print(f"参数 φ = π/4 = {phi:.4f}")
    print()

    # 创建简单初始状态
    polar_vec = create_random_polar_state(4)  # 2 比特系统，|00⟩ 状态
    polar_vec[0] = torch.tensor([127, 0], dtype=torch.int8, device=device)  # |00⟩
    polar_vec[1] = torch.tensor([0, 0], dtype=torch.int8, device=device)   # |01⟩
    polar_vec[2] = torch.tensor([0, 0], dtype=torch.int8, device=device)   # |10⟩
    polar_vec[3] = torch.tensor([0, 0], dtype=torch.int8, device=device)   # |11⟩

    print("初始状态 (|00⟩):")
    print(polar_vec_to_string(polar_vec.float(), range(4)))
    print()

    # 步骤 1: 应用 H 门
    vec_after_h = polar_vec.clone()
    apply_gate_unified_tensor(vec_after_h, 'H', qubit_idx=0)

    print("步骤 1: 应用 H 门后的状态:")
    print(polar_vec_to_string(vec_after_h.float(), range(4)))
    print()

    # 步骤 2: 应用 Rz 门
    vec_after_h_rz = vec_after_h.clone()
    apply_gate_unified_tensor(vec_after_h_rz, 'Rz', phi, qubit_idx=0)

    print("步骤 2: 应用 Rz(π/4) 门后的状态:")
    print(polar_vec_to_string(vec_after_h_rz.float(), range(4)))
    print()

    # 步骤 3: 再次应用 H 门
    vec_after_h_rz_h = vec_after_h_rz.clone()
    apply_gate_unified_tensor(vec_after_h_rz_h, 'H', qubit_idx=0)

    print("步骤 3: 再次应用 H 门后的状态 (最终 H+Rz+H):")
    print(polar_vec_to_string(vec_after_h_rz_h.float(), range(4)))
    print()

    # 对比 HRH
    vec_hrh = polar_vec.clone()
    apply_gate_unified_tensor(vec_hrh, 'HRH', phi, qubit_idx=0)

    print("直接应用 HRH(π/4) 门后的状态:")
    print(polar_vec_to_string(vec_hrh.float(), range(4)))
    print()

    # 数值比较
    h_rz_h_complex = polar_to_complex_tensor(vec_after_h_rz_h.float())
    hrh_complex = polar_to_complex_tensor(vec_hrh.float())

    diff = torch.abs(h_rz_h_complex - hrh_complex)
    max_diff = torch.max(diff).item()

    print(f"H+Rz+H vs HRH 最大差异: {max_diff:.2e}")
    if max_diff < 1e-3:
        print("✅ 结果基本等价 (在 int8 精度范围内)")
    else:
        print("❌ 结果差异较大")
    print()

    # 检查每一步的数值稳定性
    print("=== 数值稳定性检查 ===")

    # 检查 H 门后的归一化
    h_complex = polar_to_complex_tensor(vec_after_h.float())
    h_norm = torch.sqrt(torch.sum(torch.abs(h_complex)**2)).item()
    print(f"H 门后归一化: {h_norm:.6f}")

    # 检查 H+Rz 后的归一化
    h_rz_complex = polar_to_complex_tensor(vec_after_h_rz.float())
    h_rz_norm = torch.sqrt(torch.sum(torch.abs(h_rz_complex)**2)).item()
    print(f"H+Rz 门后归一化: {h_rz_norm:.6f}")

    # 检查最终归一化
    final_norm = torch.sqrt(torch.sum(torch.abs(h_rz_h_complex)**2)).item()
    print(f"最终归一化: {final_norm:.6f}")

    hrh_norm = torch.sqrt(torch.sum(torch.abs(hrh_complex)**2)).item()
    print(f"HRH 归一化: {hrh_norm:.6f}")

def check_rz_implementation():
    """检查 Rz 门的实现"""
    print("=== 检查 Rz 门实现 ===")

    # 创建 |+⟩ 状态 (H|0⟩)
    polar_vec = torch.zeros(4, 2, dtype=torch.int8, device=device)
    polar_vec[0] = torch.tensor([91, 0], dtype=torch.int8, device=device)   # |00⟩ 约 0.707
    polar_vec[1] = torch.tensor([91, 0], dtype=torch.int8, device=device)   # |01⟩ 约 0.707

    print("初始状态 (|00⟩ + |01⟩)/√2:")
    print(polar_vec_to_string(polar_vec.float(), range(4)))
    print()

    phi = math.pi / 2.2

    # 应用 Rz
    rz_vec = polar_vec.clone()
    apply_gate_unified_tensor(rz_vec, 'Rz', phi, qubit_idx=0)

    print("应用 Rz(π/4) 后的状态:")
    print(polar_vec_to_string(rz_vec.float(), range(4)))
    print()

    # 理论上，Rz(φ)|+⟩ = e^(-iφ/2)|+⟩
    theory_phase = torch.exp(torch.tensor(-1j * phi / 2, dtype=torch.complex64))
    print(f"理论相位: {theory_phase}")

if __name__ == "__main__":
    debug_h_rz_h_step_by_step()
    print("\n" + "="*50 + "\n")
    check_rz_implementation()
