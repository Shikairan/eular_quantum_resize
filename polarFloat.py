"""
纯浮点数版本的极坐标量子计算实现
使用 PyTorch 和 float32 直接存储极坐标表示（不进行 int8 编码）

✨ 主要功能：
1. 纯极坐标状态向量操作（使用 float32 直接存储）
2. 支持完整的量子门集合
3. 统一的门应用函数，支持序列处理
4. 与 polarALL.py 完全兼容的接口，但使用更高精度

⚠️ 注意：
- 本文件使用 PyTorch 和 float32 存储
- 用于验证和对照 polarALL.py 版本的正确性
- 精度更高，但内存占用更大
"""

import torch
import math
import cmath
import random
import numpy as np
from typing import List, Tuple

# 设置设备
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# ===== 极坐标转换函数（float32 版本）=====

def polar_to_complex_tensor_float(polar_vec: torch.Tensor) -> torch.Tensor:
    """float32 版本：极坐标转复数"""
    r = polar_vec[:, 0]  # 幅度 (float32)
    th = polar_vec[:, 1]  # 相位 (float32, 弧度)
    real_part = r * torch.cos(th)
    imag_part = r * torch.sin(th)
    return torch.complex(real_part, imag_part)

def complex_to_polar_tensor_float(complex_vec: torch.Tensor) -> torch.Tensor:
    """float32 版本：复数转极坐标"""
    r = torch.abs(complex_vec)
    th = torch.angle(complex_vec)
    return torch.stack([r, th], dim=1)

# ===== 量子门矩阵定义 (PyTorch 版本) =====

H_MAT_TORCH = torch.tensor([[1, 1], [1, -1]], dtype=torch.complex64, device=device) / math.sqrt(2)
S_MAT_TORCH = torch.tensor([[1, 0], [0, 1j]], dtype=torch.complex64, device=device)
T_MAT_TORCH = torch.tensor([[1, 0], [0, cmath.exp(1j * math.pi / 4)]], dtype=torch.complex64, device=device)
X_MAT_TORCH = torch.tensor([[0, 1], [1, 0]], dtype=torch.complex64, device=device)
Y_MAT_TORCH = torch.tensor([[0, -1j], [1j, 0]], dtype=torch.complex64, device=device)
Z_MAT_TORCH = torch.tensor([[1, 0], [0, -1]], dtype=torch.complex64, device=device)

# ===== PyTorch 版本的量子门函数 =====

def hadamard_polar_tensor_float(z0_batch: torch.Tensor, z1_batch: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """float32 版本：Hadamard 门极坐标实现（float32向量，批量版本）"""
    batch_size = z0_batch.shape[0]

    # 将极坐标转换为复数 (batch_size, )
    c0 = polar_to_complex_tensor_float(z0_batch)
    c1 = polar_to_complex_tensor_float(z1_batch)

    # 应用 Hadamard 矩阵到每个状态对
    # 确保所有张量在同一设备上
    device = c0.device
    h_mat = H_MAT_TORCH.to(device)
    result = torch.matmul(torch.stack([c0, c1], dim=1), h_mat.t())

    # 转换回极坐标 float32
    polar_result = complex_to_polar_tensor_float(result)
    return polar_result[:, 0], polar_result[:, 1]

def x_polar_tensor_float(z0_batch: torch.Tensor, z1_batch: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """float32 版本：X 门极坐标实现（float32向量，批量版本）"""
    # X 门：交换完整的状态向量
    return z1_batch, z0_batch

def y_polar_tensor_float(z0_batch: torch.Tensor, z1_batch: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """float32 版本：Y 门极坐标实现（float32向量，批量版本）"""
    result_z0 = z1_batch.clone()  # 原 z1 的幅度给 z0
    result_z1 = z0_batch.clone()  # 原 z0 的幅度给 z1

    # 调整相位：新 z0 相位 + π，新 z1 相位 + π/2
    result_z0[:, 1] = z1_batch[:, 1] - math.pi / 2
    result_z1[:, 1] = z0_batch[:, 1] + math.pi / 2

    return result_z0, result_z1

def z_polar_tensor_float(z0_batch: torch.Tensor, z1_batch: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """float32 版本：Z 门极坐标实现（float32向量，批量版本）"""
    # Z 门：|0⟩ -> |0⟩, |1⟩ -> -|1⟩，相当于给 |1⟩ 添加 π 相位
    result_z0 = z0_batch.clone()
    result_z1 = z1_batch.clone()
    # 更新相位：相位翻转 π
    result_z1[:, 1] = z1_batch[:, 1] + math.pi

    return result_z0, result_z1

def s_polar_tensor_float(z0_batch: torch.Tensor, z1_batch: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """float32 版本：S 门极坐标实现（float32向量，批量版本）"""
    # S 门：|0⟩ -> |0⟩, |1⟩ -> i|1⟩，相当于给 |1⟩ 添加 π/2 相位
    result_z0 = z0_batch.clone()
    result_z1 = z1_batch.clone()
    # 更新相位：添加 π/2
    result_z1[:, 1] = z1_batch[:, 1] + math.pi / 2

    return result_z0, result_z1

def t_polar_tensor_float(z0_batch: torch.Tensor, z1_batch: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """float32 版本：T 门极坐标实现（float32向量，批量版本）"""
    # T 门：|0⟩ -> |0⟩, |1⟩ -> e^(iπ/4)|1⟩，相当于给 |1⟩ 添加 π/4 相位
    result_z0 = z0_batch.clone()
    result_z1 = z1_batch.clone()
    # 更新相位：添加 π/4
    result_z1[:, 1] = z1_batch[:, 1] + math.pi / 4

    return result_z0, result_z1

def rx_polar_tensor_float(z0_batch: torch.Tensor, z1_batch: torch.Tensor, theta: float) -> Tuple[torch.Tensor, torch.Tensor]:
    """float32 版本：Rx 门极坐标实现（float32向量，批量版本）"""
    batch_size = z0_batch.shape[0]

    # 将极坐标转换为复数 (batch_size, )
    c0 = polar_to_complex_tensor_float(z0_batch)
    c1 = polar_to_complex_tensor_float(z1_batch)

    # 构造 Rx 矩阵
    device = c0.device
    c, s = math.cos(theta / 2), math.sin(theta / 2)
    rx_mat = torch.tensor([[c, -1j * s], [-1j * s, c]], dtype=torch.complex64, device=device)

    # 应用矩阵到每个状态对
    result = torch.matmul(torch.stack([c0, c1], dim=1), rx_mat.t())

    # 转换回极坐标 float32
    polar_result = complex_to_polar_tensor_float(result)
    return polar_result[:, 0], polar_result[:, 1]

def ry_polar_tensor_float(z0_batch: torch.Tensor, z1_batch: torch.Tensor, theta: float) -> Tuple[torch.Tensor, torch.Tensor]:
    """float32 版本：Ry 门极坐标实现（float32向量，批量版本）"""
    batch_size = z0_batch.shape[0]

    # 将极坐标转换为复数 (batch_size, )
    c0 = polar_to_complex_tensor_float(z0_batch)
    c1 = polar_to_complex_tensor_float(z1_batch)

    # 构造 Ry 矩阵
    device = c0.device
    c, s = math.cos(theta / 2), math.sin(theta / 2)
    ry_mat = torch.tensor([[c, -s], [s, c]], dtype=torch.complex64, device=device)

    # 应用矩阵到每个状态对
    result = torch.matmul(torch.stack([c0, c1], dim=1), ry_mat.t())

    # 转换回极坐标 float32
    polar_result = complex_to_polar_tensor_float(result)
    return polar_result[:, 0], polar_result[:, 1]

def rz_polar_tensor_float(z0_batch: torch.Tensor, z1_batch: torch.Tensor, phi: float) -> Tuple[torch.Tensor, torch.Tensor]:
    """float32 版本：Rz 门极坐标实现（直接相位操作，float32向量，批量版本）
    Rz(φ) 给 |0⟩ 添加 -φ/2 相位，给 |1⟩ 添加 φ/2 相位
    相当于 U1(φ/2) 再乘全局相位 e^(-iφ/2)，但全局相位不影响结果
    """
    result_z0 = z0_batch.clone()
    result_z1 = z1_batch.clone()
    # Rz(φ): |0⟩ -> e^(-iφ/2)|0⟩, |1⟩ -> e^(iφ/2)|1⟩
    result_z0[:, 1] = z0_batch[:, 1] - phi / 2
    result_z1[:, 1] = z1_batch[:, 1] + phi / 2

    return result_z0, result_z1

def hrh_polar_tensor_float(z0_batch: torch.Tensor, z1_batch: torch.Tensor, phi: float) -> Tuple[torch.Tensor, torch.Tensor]:
    """float32 版本：HRH 门极坐标实现（H-Rz-H 组合门，float32向量，批量版本）
    HRH(φ) = H * Rz(φ) * H，表示围绕 Y 轴旋转 φ 角度
    """
    batch_size = z0_batch.shape[0]

    # 将极坐标转换为复数 (batch_size, )
    c0 = polar_to_complex_tensor_float(z0_batch)
    c1 = polar_to_complex_tensor_float(z1_batch)
    device = c0.device

    # 构造 HRH 矩阵：H * Rz(φ) * H
    # 首先构造 Rz(φ)
    exp_neg = torch.exp(torch.tensor(-1j * phi / 2, dtype=torch.complex64, device=device))
    exp_pos = torch.exp(torch.tensor(1j * phi / 2, dtype=torch.complex64, device=device))
    rz_mat = torch.stack([
        torch.stack([exp_neg, torch.tensor(0.0, dtype=torch.complex64, device=device)]),
        torch.stack([torch.tensor(0.0, dtype=torch.complex64, device=device), exp_pos])
    ])

    # HRH = H * Rz * H
    h_mat = H_MAT_TORCH.to(device)
    hrh_mat = torch.matmul(h_mat, torch.matmul(rz_mat, h_mat))

    # 应用矩阵到每个状态对
    result = torch.matmul(torch.stack([c0, c1], dim=1), hrh_mat.t())

    # 转换回极坐标 float32
    polar_result = complex_to_polar_tensor_float(result)
    return polar_result[:, 0], polar_result[:, 1]

def u1_polar_tensor_float(z0_batch: torch.Tensor, z1_batch: torch.Tensor, lambda_param: float) -> Tuple[torch.Tensor, torch.Tensor]:
    """float32 版本：U1 门极坐标实现（直接相位操作，float32向量，批量版本）
    U1(λ) 只给 |1⟩ 状态添加相位 λ：|0⟩ -> |0⟩, |1⟩ -> e^(iλ)|1⟩
    """
    result_z0 = z0_batch.clone()
    result_z1 = z1_batch.clone()
    # 只给 |1⟩ 状态添加相位 λ
    result_z1[:, 1] = z1_batch[:, 1] + lambda_param

    return result_z0, result_z1

def u2_polar_tensor_float(z0_batch: torch.Tensor, z1_batch: torch.Tensor, phi: float, lambda_param: float) -> Tuple[torch.Tensor, torch.Tensor]:
    """float32 版本：U2 门极坐标实现（float32向量，批量版本）"""
    batch_size = z0_batch.shape[0]

    # 将极坐标转换为复数 (batch_size, )
    c0 = polar_to_complex_tensor_float(z0_batch)
    c1 = polar_to_complex_tensor_float(z1_batch)
    device = c0.device

    # 构造 U2 矩阵
    inv_sqrt2 = 1 / math.sqrt(2)
    exp_phi = torch.exp(torch.tensor(1j * phi, dtype=torch.complex64, device=device))
    exp_lambda = torch.exp(torch.tensor(1j * lambda_param, dtype=torch.complex64, device=device))
    u2_mat = torch.stack([
        torch.stack([torch.tensor(inv_sqrt2, dtype=torch.complex64, device=device),
                    -inv_sqrt2 * exp_lambda]),
        torch.stack([inv_sqrt2 * exp_phi, inv_sqrt2 * exp_phi * exp_lambda])
    ])

    # 应用矩阵到每个状态对
    result = torch.matmul(torch.stack([c0, c1], dim=1), u2_mat.t())

    # 转换回极坐标 float32
    polar_result = complex_to_polar_tensor_float(result)
    return polar_result[:, 0], polar_result[:, 1]

def u3_polar_tensor_float(z0_batch: torch.Tensor, z1_batch: torch.Tensor, theta: float, phi: float, lambda_param: float) -> Tuple[torch.Tensor, torch.Tensor]:
    """float32 版本：U3 门极坐标实现（float32向量，批量版本）"""
    batch_size = z0_batch.shape[0]

    # 将极坐标转换为复数 (batch_size, )
    c0 = polar_to_complex_tensor_float(z0_batch)
    c1 = polar_to_complex_tensor_float(z1_batch)
    device = c0.device

    # 构造 U3 矩阵
    c_val, s_val = math.cos(theta / 2), math.sin(theta / 2)
    exp_phi = torch.exp(torch.tensor(1j * phi, dtype=torch.complex64, device=device))
    exp_lambda = torch.exp(torch.tensor(1j * lambda_param, dtype=torch.complex64, device=device))
    u3_mat = torch.stack([
        torch.stack([torch.tensor(c_val, dtype=torch.complex64, device=device),
                    -exp_lambda * s_val]),
        torch.stack([exp_phi * s_val, exp_phi * exp_lambda * c_val])
    ])

    # 应用矩阵到每个状态对
    result = torch.matmul(torch.stack([c0, c1], dim=1), u3_mat.t())

    # 转换回极坐标 float32
    polar_result = complex_to_polar_tensor_float(result)
    return polar_result[:, 0], polar_result[:, 1]

# ===== 控制门实现 =====

def apply_cnot_polar_tensor_float(polar_vec: torch.Tensor, control_idx: int, target_idx: int):
    """
    应用 CNOT 门（float32 极坐标版本）

    Args:
        polar_vec: 极坐标状态向量，形状 (N, 2)，dtype=torch.float32
        control_idx: 控制比特索引
        target_idx: 目标比特索引
    """
    n = polar_vec.shape[0]
    nqubit = int(math.log2(n))
    assert 2 ** nqubit == n
    assert control_idx != target_idx
    assert 0 <= control_idx < nqubit and 0 <= target_idx < nqubit

    # 对于 CNOT：当控制比特为 |1⟩ 时，翻转目标比特
    control_mask = 1 << control_idx
    target_mask = 1 << target_idx

    # 找到所有控制比特为 1 的状态
    indices_to_swap = []
    for i in range(n):
        if (i & control_mask) != 0:  # 控制比特为 1
            # 找到对应的目标比特翻转后的状态
            j = i ^ target_mask  # 翻转目标比特
            if i < j:  # 避免重复处理
                indices_to_swap.append((i, j))

    # 执行交换
    for i, j in indices_to_swap:
        temp = polar_vec[i].clone()
        polar_vec[i] = polar_vec[j]
        polar_vec[j] = temp

def apply_cz_polar_tensor_float(polar_vec: torch.Tensor, qubit1_idx: int, qubit2_idx: int):
    """
    应用 CZ 门（float32 极坐标版本）

    Args:
        polar_vec: 极坐标状态向量，形状 (N, 2)，dtype=torch.float32
        qubit1_idx: 第一个比特索引
        qubit2_idx: 第二个比特索引
    """
    n = polar_vec.shape[0]
    nqubit = int(math.log2(n))
    assert 2 ** nqubit == n
    assert qubit1_idx != qubit2_idx
    assert 0 <= qubit1_idx < nqubit and 0 <= qubit2_idx < nqubit

    # CZ 门：当两个比特都为 |1⟩ 时，相位翻转 π
    mask1 = 1 << qubit1_idx
    mask2 = 1 << qubit2_idx

    for i in range(n):
        if (i & mask1) != 0 and (i & mask2) != 0:  # 两个比特都为 1
            # 相位翻转 π
            polar_vec[i, 1] += math.pi

def apply_cy_polar_tensor_float(polar_vec: torch.Tensor, control_idx: int, target_idx: int):
    """应用 CY 门（float32 极坐标版本）"""
    apply_controlled_gate_polar_tensor_float(polar_vec, control_idx, target_idx, y_polar_tensor_float)

def apply_ch_polar_tensor_float(polar_vec: torch.Tensor, control_idx: int, target_idx: int):
    """应用 CH 门（float32 极坐标版本）"""
    apply_controlled_gate_polar_tensor_float(polar_vec, control_idx, target_idx, hadamard_polar_tensor_float)

def apply_cs_polar_tensor_float(polar_vec: torch.Tensor, control_idx: int, target_idx: int):
    """应用 CS 门（float32 极坐标版本）"""
    apply_controlled_gate_polar_tensor_float(polar_vec, control_idx, target_idx, s_polar_tensor_float)

def apply_ct_polar_tensor_float(polar_vec: torch.Tensor, control_idx: int, target_idx: int):
    """应用 CT 门（float32 极坐标版本）"""
    apply_controlled_gate_polar_tensor_float(polar_vec, control_idx, target_idx, t_polar_tensor_float)

def apply_crx_polar_tensor_float(polar_vec: torch.Tensor, control_idx: int, target_idx: int, theta: float):
    """应用 CRx 门（float32 极坐标版本）"""
    apply_controlled_gate_polar_tensor_float(polar_vec, control_idx, target_idx, rx_polar_tensor_float, theta)

def apply_cry_polar_tensor_float(polar_vec: torch.Tensor, control_idx: int, target_idx: int, theta: float):
    """应用 CRy 门（float32 极坐标版本）"""
    apply_controlled_gate_polar_tensor_float(polar_vec, control_idx, target_idx, ry_polar_tensor_float, theta)

def apply_crz_polar_tensor_float(polar_vec: torch.Tensor, control_idx: int, target_idx: int, phi: float):
    """应用 CRz 门（float32 极坐标版本）"""
    apply_controlled_gate_polar_tensor_float(polar_vec, control_idx, target_idx, rz_polar_tensor_float, phi)

def apply_cu1_polar_tensor_float(polar_vec: torch.Tensor, control_idx: int, target_idx: int, lambda_param: float):
    """应用 CU1 门（float32 极坐标版本）"""
    apply_controlled_gate_polar_tensor_float(polar_vec, control_idx, target_idx, u1_polar_tensor_float, lambda_param)

def apply_cu2_polar_tensor_float(polar_vec: torch.Tensor, control_idx: int, target_idx: int, phi: float, lambda_param: float):
    """应用 CU2 门（float32 极坐标版本）"""
    apply_controlled_gate_polar_tensor_float(polar_vec, control_idx, target_idx, u2_polar_tensor_float, phi, lambda_param)

def apply_cu3_polar_tensor_float(polar_vec: torch.Tensor, control_idx: int, target_idx: int, theta: float, phi: float, lambda_param: float):
    """应用 CU3 门（float32 极坐标版本）"""
    apply_controlled_gate_polar_tensor_float(polar_vec, control_idx, target_idx, u3_polar_tensor_float, theta, phi, lambda_param)

def apply_controlled_gate_polar_tensor_float(polar_vec: torch.Tensor, control_idx: int, target_idx: int, gate_func, *gate_params):
    """
    应用控制门（float32 极坐标版本）
    使用张量操作完全消除 for 循环

    Args:
        polar_vec: 极坐标状态向量，形状 (N, 2)
        control_idx: 控制比特索引
        target_idx: 目标比特索引
        gate_func: 要应用的单比特门函数（批量版本）
        *gate_params: 门函数需要的参数
    """
    n = polar_vec.shape[0]
    nqubit = int(math.log2(n))
    assert 2 ** nqubit == n
    assert control_idx != target_idx
    assert 0 <= control_idx < nqubit and 0 <= target_idx < nqubit

    control_mask = 1 << control_idx
    target_mask = 1 << target_idx

    # 使用 PyTorch 张量操作生成所有需要处理的状态对索引
    # 控制门：当控制比特为 1 时，对目标比特应用单比特门

    # 方法：找到所有控制比特为 1 且目标比特为 0 的状态
    all_indices = torch.arange(n, device=device, dtype=torch.long)

    # 控制比特为 1 且目标比特为 0 的状态
    control_set_target_clear = (all_indices & control_mask) != 0
    control_set_target_clear &= (all_indices & target_mask) == 0

    # 这些状态就是 idx0
    idx0_batch = all_indices[control_set_target_clear]

    # 对应的 idx1 是设置目标比特
    idx1_batch = idx0_batch | target_mask

    # 验证 idx1 也在有效范围内且控制比特仍然为 1
    valid_mask = (idx1_batch < n) & ((idx1_batch & control_mask) != 0)

    idx0_batch = idx0_batch[valid_mask]
    idx1_batch = idx1_batch[valid_mask]

    # 如果没有需要处理的状态对，直接返回
    if len(idx0_batch) == 0:
        return

    # 批量提取完整的状态向量
    states0 = polar_vec[idx0_batch]  # 形状: (batch_size, 2) [幅度, 相位]
    states1 = polar_vec[idx1_batch]  # 形状: (batch_size, 2) [幅度, 相位]

    # 批量应用门函数 - 直接传递所有参数
    new_states0, new_states1 = gate_func(states0, states1, *gate_params)

    # 将结果写回原向量
    polar_vec[idx0_batch] = new_states0
    polar_vec[idx1_batch] = new_states1

# ===== 核心门应用函数 =====

def apply_polar_gate_tensor_float(polar_vec: torch.Tensor, gate_func, *args, qubit_idx=0):
    """
    对指定的比特位应用量子门（float32 极坐标版本）
    使用张量操作替换 for 循环以提高性能

    Args:
        polar_vec: 极坐标状态向量，形状 (N, 2)
        gate_func: 门函数
        *args: 门函数需要的参数
        qubit_idx: 要操作的比特位索引（0 表示最低位）
    """
    n = polar_vec.shape[0]
    nqubit = int(math.log2(n))
    assert 2 ** nqubit == n, f"状态向量长度必须是 2 的幂次，当前长度: {n}"
    assert 0 <= qubit_idx < nqubit, f"比特位索引必须在 [0, {nqubit}) 范围内"

    # 使用 PyTorch 张量操作生成所有需要处理的状态对索引
    # 对于第 qubit_idx 个比特，我们需要处理所有第 qubit_idx 位为 0 的状态及其对应的第 qubit_idx 位为 1 的状态

    # 生成所有状态索引
    all_indices = torch.arange(n, device=device)

    # 找到第 qubit_idx 位为 0 的所有状态索引
    mask = (all_indices & (1 << qubit_idx)) == 0
    idx0_batch = all_indices[mask]  # 第 qubit_idx 位为 0 的状态

    # 对应的第 qubit_idx 位为 1 的状态索引
    idx1_batch = idx0_batch | (1 << qubit_idx)

    # 批量提取完整的状态向量
    states0 = polar_vec[idx0_batch]  # 形状: (batch_size, 2) [幅度, 相位]
    states1 = polar_vec[idx1_batch]  # 形状: (batch_size, 2) [幅度, 相位]

    # 批量应用门函数
    new_states0, new_states1 = gate_func(states0, states1, *args)

    # 将结果写回原向量
    polar_vec[idx0_batch] = new_states0
    polar_vec[idx1_batch] = new_states1

# ===== 门定义字典 =====

POLAR_GATES_FLOAT = {
    'Y':  y_polar_tensor_float,
    'X':  x_polar_tensor_float,
    'S':  s_polar_tensor_float,
    'T':  t_polar_tensor_float,
    'Z':  z_polar_tensor_float,
    'Rx': rx_polar_tensor_float,
    'Ry': ry_polar_tensor_float,
    'Rz': rz_polar_tensor_float,
    'U1': u1_polar_tensor_float,
    'U2': u2_polar_tensor_float,
    'U3': u3_polar_tensor_float,
    'H': hadamard_polar_tensor_float,
    'HRH': hrh_polar_tensor_float
}

# ===== 统一的门应用函数 =====

def apply_gate_unified_tensor_float(polar_vec: torch.Tensor, gate_name: str, *gate_params, qubit_idx=0, control_idx=None, target_idx=None):
    """
    统一的量子门应用函数（float32 极坐标版本）

    Args:
        polar_vec: 极坐标状态向量，形状 (N, 2)
        gate_name: 门名称
        *gate_params: 门函数需要的参数
        qubit_idx: 单比特门的比特索引
        control_idx: 控制比特索引（用于控制门）
        target_idx: 目标比特索引（用于控制门）
    """
    # 控制门（需要 control_idx 和 target_idx）
    controlled_gates_float = {
        'CX': lambda vec, c, t: apply_cnot_polar_tensor_float(vec, c, t),
        'CY': lambda vec, c, t: apply_cy_polar_tensor_float(vec, c, t),
        'CZ': lambda vec, c, t: apply_cz_polar_tensor_float(vec, c, t),
        'CH': lambda vec, c, t: apply_ch_polar_tensor_float(vec, c, t),
        'CS': lambda vec, c, t: apply_cs_polar_tensor_float(vec, c, t),
        'CT': lambda vec, c, t: apply_ct_polar_tensor_float(vec, c, t),
        'CRx': lambda vec, c, t, theta: apply_crx_polar_tensor_float(vec, c, t, theta),
        'CRy': lambda vec, c, t, theta: apply_cry_polar_tensor_float(vec, c, t, theta),
        'CRz': lambda vec, c, t, phi: apply_crz_polar_tensor_float(vec, c, t, phi),
        'CU1': lambda vec, c, t, lambda_param: apply_cu1_polar_tensor_float(vec, c, t, lambda_param),
        'CU2': lambda vec, c, t, phi, lambda_param: apply_cu2_polar_tensor_float(vec, c, t, phi, lambda_param),
        'CU3': lambda vec, c, t, theta, phi, lambda_param: apply_cu3_polar_tensor_float(vec, c, t, theta, phi, lambda_param),
    }

    if gate_name in controlled_gates_float:
        # 控制门
        if control_idx is None:
            control_idx = 0
        if target_idx is None:
            target_idx = 1

        func_polar = controlled_gates_float[gate_name]

        if gate_name in ['CX', 'CZ']:
            # CNOT 和 CZ 使用特殊函数
            func_polar(polar_vec, control_idx, target_idx)
        else:
            # 其他控制门
            if gate_params:
                func_polar(polar_vec, control_idx, target_idx, *gate_params)
            else:
                func_polar(polar_vec, control_idx, target_idx)

    elif gate_name in ['CNOT', 'CZ']:
        # 兼容旧名称
        if control_idx is None:
            control_idx = 0
        if target_idx is None:
            target_idx = 1
        if gate_name == 'CNOT':
            apply_cnot_polar_tensor_float(polar_vec, control_idx, target_idx)
        elif gate_name == 'CZ':
            apply_cz_polar_tensor_float(polar_vec, control_idx, target_idx)

    else:
        # 单比特门
        apply_polar_gate_tensor_float(polar_vec, POLAR_GATES_FLOAT[gate_name], *gate_params, qubit_idx=qubit_idx)

# ===== 辅助函数 =====

def create_random_polar_state_float(n_amps: int) -> torch.Tensor:
    """创建随机 float32 极坐标 int8 状态向量"""
    # 创建随机复数向量
    real_parts = torch.randn(n_amps, device=device)
    imag_parts = torch.randn(n_amps, device=device)
    complex_vec = torch.complex(real_parts, imag_parts)

    # 归一化
    norm = torch.sqrt(torch.sum(torch.abs(complex_vec)**2))
    complex_vec = complex_vec / norm

    # 转换为极坐标 float32
    polar_vec = complex_to_polar_tensor_float(complex_vec)
    return polar_vec

def polar_vec_to_string_float(polar_vec: torch.Tensor, indices=None) -> str:
    """将 float32 极坐标向量转换为字符串表示"""
    if indices is None:
        indices = range(len(polar_vec))

    result = []
    for i in indices:
        r = polar_vec[i, 0].item()
        th = polar_vec[i, 1].item()
        complex_val = complex(r * math.cos(th), r * math.sin(th))
        result.append(f"{i:02b}: {complex_val:.5f}")
    return " | ".join(result)

# ===== 序列处理主函数 =====

def create_initial_vec_like_v2(n_amps: int) -> List[complex]:
    """按照 ALLeularV2.py 第903行的确切格式创建初始向量（未归一化）"""
    complex_vector = [complex(random.random(), random.random()) for _ in range(n_amps)]
    #norm = math.sqrt(sum(abs(z)**2 for z in complex_vector))
    #normalized_vector = [z / norm for z in complex_vector]
    norm = np.linalg.norm(complex_vector)
    normalized_vector = complex_vector / norm
    return normalized_vector

def process_sequence_polar_float(initial_vec: List[complex], seq: List[Tuple], verbose: bool = False) -> Tuple[torch.Tensor, List[torch.Tensor]]:
    """
    处理量子门序列（float32 极坐标版本），记录每一步的状态变化
    在函数内部将复数向量转换为 float32 极坐标表示

    Args:
        initial_vec: 初始状态向量（复数形式，与 ALLeularV2.py 第903行格式相同）
        seq: 量子门序列，格式为 [(gate_name, param_str, params, qubit_info...), ...]
            对于单比特门: (gate_name, param_str, params, qubit_idx)
            对于控制门: (gate_name, param_str, params, control_idx, target_idx)
        verbose: 是否打印详细信息

    Returns:
        (final_vec, state_history): 最终状态向量和每一步的状态历史
    """
    n_amps = len(initial_vec)
    n_qubits = int(math.log2(n_amps))
    assert 2 ** n_qubits == n_amps, f"状态向量长度必须是 2 的幂，当前: {n_amps}"

    # 将初始复数向量转换为 PyTorch 张量并转换为极坐标 float32
    complex_tensor = torch.tensor(initial_vec, dtype=torch.complex64, device=device)

    # 转换为极坐标 float32 格式
    vec = complex_to_polar_tensor_float(complex_tensor)

    # 记录状态历史
    state_history = [vec.clone()]  # 初始状态

    if verbose:
        print(f"处理 {len(seq)} 个量子门，系统有 {n_qubits} 个比特")
        print(f"初始状态（前8个）:")
        print(polar_vec_to_string_float(vec, range(min(8, n_amps))))

    # 处理每个门
    for step, gate_tuple in enumerate(seq):
        # 解析序列格式：支持不同的参数数量
        if len(gate_tuple) == 4:
            # 单比特门: (name, param_str, params, qubit_idx)
            name, param_str, gate_params, qubit_idx = gate_tuple
            is_controlled = False
        elif len(gate_tuple) == 5:
            # 控制门: (name, param_str, params, control_idx, target_idx)
            name, param_str, gate_params, control_idx, target_idx = gate_tuple
            is_controlled = True
        else:
            raise ValueError(f"无效的序列格式: {gate_tuple}")

        # 检查是否是控制门（额外确认）
        is_controlled = is_controlled or (name.startswith('C') or name in ['CNOT', 'CZ'])

        if is_controlled:
            # 控制门：使用序列中指定的控制和目标比特位
            apply_gate_unified_tensor_float(vec, name, *gate_params, control_idx=control_idx, target_idx=target_idx)
            gate_info = f"{name}(控制={control_idx}, 目标={target_idx})"
        else:
            # 单比特门：使用序列中指定的比特位
            apply_gate_unified_tensor_float(vec, name, *gate_params, qubit_idx=qubit_idx)
            gate_info = f"{name}(比特={qubit_idx})"

        # 记录每一步的状态
        state_history.append(vec.clone())

        if verbose and (step % 50 == 49 or step == len(seq) - 1):
            param_display = param_str if param_str else ''
            print(f"\nAfter {step+1} gates - Last: {gate_info}{param_display}:")
            print("当前状态（前8个）:")
            print(polar_vec_to_string_float(vec, range(min(8, n_amps))))

    if verbose:
        print("\n最终状态归一化检查:")
        complex_vec = polar_to_complex_tensor_float(vec)
        norm = torch.sqrt(torch.sum(torch.abs(complex_vec)**2)).item()
        print(f"归一化误差: {abs(norm - 1.0):.2e}")

    return vec, state_history

# ===== 使用示例 =====

if __name__ == "__main__":
    print("polarFloat.py float32 极坐标量子计算示例")
    print("=" * 50)

    # 示例 1: 按照代码块 (1-10) 的使用方式
    print("示例 1: 基本使用方式")

    # 生成随机序列（与 polarALL.py 相同的方式）
    from polarALL import random_sequence_tensor
    seq = random_sequence_tensor(100, 0, 0, n_qubits=5, shuffle=True)

    # 创建初始向量（与 ALLeularV2.py 第903行相同）
    initial_vec = create_initial_vec_like_v2(32)

    norm = np.linalg.norm(initial_vec)
    print(f"初始向量归一化误差: {abs(norm - 1.0):.2e}")

    # 处理序列，获取最终状态和历史记录
    final_vec, state_history = process_sequence_polar_float(initial_vec, seq, verbose=True)

    print(f"记录了 {len(state_history)} 步状态变化")
    print(f"每步都可以访问: state_history[step]")
    print(f"最终状态归一化误差: {abs(torch.sqrt(torch.sum(torch.abs(polar_to_complex_tensor_float(final_vec))**2)).item() - 1.0):.2e}")

    print("\n示例完成！")