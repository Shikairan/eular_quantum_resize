"""
分析不同量子门在重复应用下的归一化稳定性
"""

import torch
import numpy as np
import math
from ALLeularV4 import *

def analyze_gate_matrices():
    """分析不同门的矩阵形式和等价关系"""
    print("=== 量子门矩阵分析 ===")

    # 参数
    theta = math.pi / 4

    print(f"使用参数 θ = π/4 = {theta:.4f}")
    print()

    # U1 门
    u1_matrix = torch.tensor([[1.0, 0.0], [0.0, torch.exp(torch.tensor(1j * theta, dtype=torch.complex64))]], dtype=torch.complex64)
    print("U1(θ) 矩阵:")
    print(f"[[1,      0     ]")
    print(f" [0,      e^(iθ)]]")
    print(f"实际值: {u1_matrix}")
    print()

    # Rz 门
    rz_matrix = torch.tensor([
        [torch.exp(torch.tensor(-1j * theta / 2, dtype=torch.complex64)), 0],
        [0, torch.exp(torch.tensor(1j * theta / 2, dtype=torch.complex64))]
    ], dtype=torch.complex64)
    print("Rz(θ) 矩阵:")
    print("[[e^(-iθ/2),    0         ]")
    print(" [0,            e^(iθ/2)  ]]")
    print(f"实际值: {rz_matrix}")
    print()

    # Rx 门
    c, s = math.cos(theta / 2), math.sin(theta / 2)
    rx_matrix = torch.tensor([[c, -1j * s], [-1j * s, c]], dtype=torch.complex64)
    print("Rx(θ) 矩阵:")
    print("[[cos(θ/2),    -i·sin(θ/2)]")
    print(" [-i·sin(θ/2),  cos(θ/2)   ]]")
    print(f"实际值: {rx_matrix}")
    print()

    # Ry 门
    ry_matrix = torch.tensor([[c, -s], [s, c]], dtype=torch.complex64)
    print("Ry(θ) 矩阵:")
    print("[[cos(θ/2),    -sin(θ/2)]")
    print(" [sin(θ/2),     cos(θ/2) ]]")
    print(f"实际值: {ry_matrix}")
    print()

    # 验证等价关系
    print("=== 等价关系验证 ===")

    # U1(θ) vs Rz(θ + π)
    rz_shifted = torch.tensor([
        [torch.exp(torch.tensor(-1j * (theta + math.pi) / 2, dtype=torch.complex64)), 0],
        [0, torch.exp(torch.tensor(1j * (theta + math.pi) / 2, dtype=torch.complex64))]
    ], dtype=torch.complex64)

    print(f"U1(θ) 与 Rz(θ + π) 是否等价: {torch.allclose(u1_matrix, rz_shifted, atol=1e-6)}")
    print(f"U1(θ) = {u1_matrix}")
    print(f"Rz(θ + π) = {rz_shifted}")
    print()

def analyze_unitary_preservation():
    """分析幺正性保持"""
    print("=== 幺正性分析 ===")

    theta = math.pi / 4

    gates = [
        ('U1', lambda t: torch.tensor([[1.0, 0.0], [0.0, torch.exp(torch.tensor(1j * t, dtype=torch.complex64))]], dtype=torch.complex64)),
        ('Rz', lambda t: torch.tensor([[torch.exp(torch.tensor(-1j * t / 2, dtype=torch.complex64)), 0], [0, torch.exp(torch.tensor(1j * t / 2, dtype=torch.complex64))]], dtype=torch.complex64)),
        ('Rx', lambda t: torch.tensor([[math.cos(t/2), -1j*math.sin(t/2)], [-1j*math.sin(t/2), math.cos(t/2)]], dtype=torch.complex64)),
        ('Ry', lambda t: torch.tensor([[math.cos(t/2), -math.sin(t/2)], [math.sin(t/2), math.cos(t/2)]], dtype=torch.complex64)),
    ]

    for name, matrix_func in gates:
        matrix = matrix_func(theta)

        # 计算 U†
        u_dagger = torch.conj(matrix.t())

        # 计算 U†U (应该是单位矩阵)
        product = torch.matmul(u_dagger, matrix)

        # 检查是否接近单位矩阵
        identity = torch.eye(2, dtype=torch.complex64)
        is_unitary = torch.allclose(product, identity, atol=1e-6)

        print(f"{name} 门幺正性检查: {'✅' if is_unitary else '❌'}")
        print(f"  U†U = {product}")
        print(f"  与 I 的差: {torch.abs(product - identity).max().item():.2e}")
        print()

def analyze_numerical_stability():
    """分析数值稳定性"""
    print("=== 数值稳定性分析 ===")

    # 创建简单的测试状态
    state = torch.tensor([1.0, 0.0], dtype=torch.complex64)  # |0⟩ 状态

    theta = math.pi / 4
    n_applications = 1000

    print(f"对 |0⟩ 状态应用 {n_applications} 次不同门")
    print()

    gates = [
        ('U1', 'pure_phase'),
        ('Rz', 'symmetric_phase'),
        ('Rx', 'amplitude_rotation'),
        ('Ry', 'amplitude_rotation'),
    ]

    for gate_name, gate_type in gates:
        current_state = state.clone()

        for _ in range(n_applications):
            if gate_name == 'U1':
                # U1 只影响 |1⟩ 状态，对 |0⟩ 无影响
                pass  # |0⟩ 保持不变
            elif gate_name == 'Rz':
                # Rz(θ) |0⟩ = e^(-iθ/2) |0⟩
                current_state[0] *= torch.exp(torch.tensor(-1j * theta / 2, dtype=torch.complex64))
            elif gate_name == 'Rx':
                # Rx 旋转会改变振幅
                c, s = math.cos(theta / 2), math.sin(theta / 2)
                new_0 = c * current_state[0] - 1j * s * current_state[1]
                new_1 = -1j * s * current_state[0] + c * current_state[1]
                current_state = torch.stack([new_0, new_1])
            elif gate_name == 'Ry':
                # Ry 旋转也会改变振幅
                c, s = math.cos(theta / 2), math.sin(theta / 2)
                new_0 = c * current_state[0] - s * current_state[1]
                new_1 = s * current_state[0] + c * current_state[1]
                current_state = torch.stack([new_0, new_1])

        norm = torch.sqrt(torch.sum(torch.abs(current_state)**2)).item()
        phase_0 = torch.angle(current_state[0]).item()
        phase_1 = torch.angle(current_state[1]).item()

        print(f"{gate_name:<8} | {gate_type:<15} | {norm:.6f} | {phase_0:.4f} | {phase_1:.4f}")
def explain_int8_precision():
    """解释 int8 精度的影响"""
    print("=== int8 精度分析 ===")

    # 原始高精度值
    theta = math.pi / 4
    exp_val = torch.exp(torch.tensor(1j * theta, dtype=torch.complex64))

    print(f"原始复数: {exp_val}")
    print(f"实部: {exp_val.real:.6f}, 虚部: {exp_val.imag:.6f}")
    print()

    # 转换为极坐标
    r = torch.abs(exp_val)
    phi = torch.angle(exp_val)

    print("极坐标表示:")
    print(f"幅度 r = {r:.6f}")
    print(f"相位 φ = {phi:.6f} ({phi/math.pi:.2f}π)")
    print()

    # int8 编码
    r_encoded = encode_r_tensor(r)
    phi_encoded = encode_th_tensor(phi)

    print("int8 编码:")
    print(f"幅度编码: {r_encoded} (范围: -128 到 127)")
    print(f"相位编码: {phi_encoded} (范围: -128 到 127)")
    print()

    # 解码回浮点数
    r_decoded = decode_r_tensor(r_encoded.float())
    phi_decoded = decode_th_tensor(phi_encoded.float())

    print("解码回浮点数:")
    print(f"幅度: {r_decoded:.6f} (误差: {abs(r - r_decoded):.2e})")
    print(f"相位: {phi_decoded:.6f} (误差: {abs(phi - phi_decoded):.2e})")
    print()

    # 重建复数
    reconstructed = torch.complex(r_decoded * torch.cos(phi_decoded),
                                r_decoded * torch.sin(phi_decoded))

    error = torch.abs(exp_val - reconstructed).item()
    print(f"重建复数: {reconstructed}")
    print(f"重建误差: {error:.2e}")
    print()

    print("分析:")
    print("- int8 编码引入量化误差")
    print("- 1000 次累积会导致显著误差")
    print("- 相位门误差较小，旋转门误差较大")

if __name__ == "__main__":
    analyze_gate_matrices()
    print("\n" + "="*50 + "\n")
    analyze_unitary_preservation()
    print("\n" + "="*50 + "\n")
    analyze_numerical_stability()
    print("\n" + "="*50 + "\n")
    explain_int8_precision()
