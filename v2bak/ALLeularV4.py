"""
使用 PyTorch 实现的极坐标量子计算 - V4 版本

✨ 主要改进：
1. 完全基于 PyTorch 张量操作
2. 使用 PyTorch 矩阵方法进行极坐标操作
3. 支持 GPU 加速
4. 保持与 ALLeularV2.py 相同的接口

⚠️ 重要说明：
- 状态向量使用 PyTorch 张量，形状为 (N, 2)，其中 N=2^n 是状态数量
- 第一列存储幅度 r，第二列存储相位 theta
- 所有运算都在 GPU 上进行（如果可用）
"""

import torch
import math
import cmath
import random
from typing import List, Tuple, Optional
import numpy as np

# 设置设备
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# ===== 常量定义 =====
R_MIN, R_MAX = 0.0, 1.0          # 幅值动态范围 [0,1]
TH_SCALE = 128 / math.pi          # 相位 [-π,π] → [-128,127]

# ===== PyTorch 版本的编解码函数 =====

def decode_r_tensor(r_encoded: torch.Tensor) -> torch.Tensor:
    """PyTorch 版本：解码幅度"""
    return (r_encoded + 128.0) / 255.0 * (R_MAX - R_MIN) + R_MIN

def encode_r_tensor(r: torch.Tensor) -> torch.Tensor:
    """PyTorch 版本：编码幅度"""
    r = torch.clamp(r, R_MIN, R_MAX)
    return torch.round((r - R_MIN) / (R_MAX - R_MIN) * 255.0 - 128.0).to(torch.int8)

def decode_th_tensor(th_encoded: torch.Tensor) -> torch.Tensor:
    """PyTorch 版本：解码相位"""
    return th_encoded / TH_SCALE

def encode_th_tensor(th: torch.Tensor) -> torch.Tensor:
    """PyTorch 版本：编码相位"""
    # 使用 atan2 确保相位在 [-π, π] 范围内
    th = torch.atan2(torch.sin(th), torch.cos(th))
    return torch.round(th * TH_SCALE).to(torch.int8)

def polar_to_complex_tensor(polar_vec: torch.Tensor) -> torch.Tensor:
    """PyTorch 版本：极坐标转复数"""
    r = decode_r_tensor(polar_vec[:, 0].float())
    th = decode_th_tensor(polar_vec[:, 1].float())
    real_part = r * torch.cos(th)
    imag_part = r * torch.sin(th)
    return torch.complex(real_part, imag_part)

def complex_to_polar_tensor(complex_vec: torch.Tensor) -> torch.Tensor:
    """PyTorch 版本：复数转极坐标"""
    r = torch.abs(complex_vec)
    th = torch.angle(complex_vec)
    r_encoded = encode_r_tensor(r)
    th_encoded = encode_th_tensor(th)
    return torch.stack([r_encoded, th_encoded], dim=1)

# ===== 量子门矩阵定义 (PyTorch 版本) =====

H_MAT_TORCH = torch.tensor([[1, 1], [1, -1]], dtype=torch.complex64, device=device) / math.sqrt(2)
S_MAT_TORCH = torch.tensor([[1, 0], [0, 1j]], dtype=torch.complex64, device=device)
T_MAT_TORCH = torch.tensor([[1, 0], [0, cmath.exp(1j * math.pi / 4)]], dtype=torch.complex64, device=device)
X_MAT_TORCH = torch.tensor([[0, 1], [1, 0]], dtype=torch.complex64, device=device)
Y_MAT_TORCH = torch.tensor([[0, -1j], [1j, 0]], dtype=torch.complex64, device=device)
Z_MAT_TORCH = torch.tensor([[1, 0], [0, -1]], dtype=torch.complex64, device=device)

# ===== PyTorch 版本的量子门函数 =====

def hadamard_polar_tensor(z0_batch: torch.Tensor, z1_batch: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """PyTorch 版本：Hadamard 门极坐标实现（int8向量，批量版本）"""
    batch_size = z0_batch.shape[0]

    # 将极坐标转换为复数 (batch_size, )
    c0 = polar_to_complex_tensor(z0_batch.float())
    c1 = polar_to_complex_tensor(z1_batch.float())

    # 应用 Hadamard 矩阵到每个状态对
    # states: (batch_size, 2), H_MAT_TORCH: (2, 2) -> result: (batch_size, 2)
    result = torch.matmul(torch.stack([c0, c1], dim=1), H_MAT_TORCH.t())

    # 转换回极坐标 int8
    polar_result = complex_to_polar_tensor(result)
    return polar_result[:, 0].to(torch.int8), polar_result[:, 1].to(torch.int8)

def x_polar_tensor(z0_batch: torch.Tensor, z1_batch: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """PyTorch 版本：X 门极坐标实现（int8向量，批量版本）"""
    return z1_batch, z0_batch

def y_polar_tensor(z0_batch: torch.Tensor, z1_batch: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """PyTorch 版本：Y 门极坐标实现（int8向量，批量版本）"""
    batch_size = z0_batch.shape[0]

    # 将极坐标转换为复数 (batch_size, )
    c0 = polar_to_complex_tensor(z0_batch.float())
    c1 = polar_to_complex_tensor(z1_batch.float())

    # 应用 Y 矩阵到每个状态对
    result = torch.matmul(torch.stack([c0, c1], dim=1), Y_MAT_TORCH.t())

    # 转换回极坐标 int8
    polar_result = complex_to_polar_tensor(result)
    return polar_result[:, 0].to(torch.int8), polar_result[:, 1].to(torch.int8)

def z_polar_tensor(z0_batch: torch.Tensor, z1_batch: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """PyTorch 版本：Z 门极坐标实现（int8向量，批量版本）"""
    # Z 门只改变 |1⟩ 的相位
    z1_new = z1_batch.clone()
    z1_new[:, 1] = encode_th_tensor(decode_th_tensor(z1_batch[:, 1].float()) + math.pi)
    return z0_batch, z1_new

def s_polar_tensor(z0_batch: torch.Tensor, z1_batch: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """PyTorch 版本：S 门极坐标实现（int8向量，批量版本）"""
    # S 门只改变 |1⟩ 的相位
    z1_new = z1_batch.clone()
    z1_new[:, 1] = encode_th_tensor(decode_th_tensor(z1_batch[:, 1].float()) + math.pi / 2)
    return z0_batch, z1_new

def t_polar_tensor(z0_batch: torch.Tensor, z1_batch: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """PyTorch 版本：T 门极坐标实现（int8向量，批量版本）"""
    # T 门只改变 |1⟩ 的相位
    z1_new = z1_batch.clone()
    z1_new[:, 1] = encode_th_tensor(decode_th_tensor(z1_batch[:, 1].float()) + math.pi / 4)
    return z0_batch, z1_new

def rx_polar_tensor(z0_batch: torch.Tensor, z1_batch: torch.Tensor, theta: float) -> Tuple[torch.Tensor, torch.Tensor]:
    """PyTorch 版本：Rx 门极坐标实现（int8向量，批量版本）"""
    batch_size = z0_batch.shape[0]

    # 将极坐标转换为复数 (batch_size, )
    c0 = polar_to_complex_tensor(z0_batch.float())
    c1 = polar_to_complex_tensor(z1_batch.float())

    # 构造 Rx 矩阵
    c, s = math.cos(theta / 2), math.sin(theta / 2)
    rx_mat = torch.tensor([[c, -1j * s], [-1j * s, c]], dtype=torch.complex64, device=device)

    # 应用矩阵到每个状态对
    # rx_mat: (2, 2), states: (batch_size, 2) -> result: (batch_size, 2)
    result = torch.matmul(torch.stack([c0, c1], dim=1), rx_mat.t())

    # 转换回极坐标 int8
    polar_result = complex_to_polar_tensor(result)
    return polar_result[:, 0].to(torch.int8), polar_result[:, 1].to(torch.int8)

def ry_polar_tensor(z0_batch: torch.Tensor, z1_batch: torch.Tensor, theta: float) -> Tuple[torch.Tensor, torch.Tensor]:
    """PyTorch 版本：Ry 门极坐标实现（int8向量，批量版本）"""
    batch_size = z0_batch.shape[0]

    # 将极坐标转换为复数 (batch_size, )
    c0 = polar_to_complex_tensor(z0_batch.float())
    c1 = polar_to_complex_tensor(z1_batch.float())

    # 构造 Ry 矩阵
    c, s = math.cos(theta / 2), math.sin(theta / 2)
    ry_mat = torch.tensor([[c, -s], [s, c]], dtype=torch.complex64, device=device)

    # 应用矩阵到每个状态对
    result = torch.matmul(torch.stack([c0, c1], dim=1), ry_mat.t())

    # 转换回极坐标 int8
    polar_result = complex_to_polar_tensor(result)
    return polar_result[:, 0].to(torch.int8), polar_result[:, 1].to(torch.int8)

def rz_polar_tensor(z0_batch: torch.Tensor, z1_batch: torch.Tensor, phi: float) -> Tuple[torch.Tensor, torch.Tensor]:
    """PyTorch 版本：Rz 门极坐标实现（通过 U1 等价实现，int8向量，批量版本）
    Rz(φ) 等价于 U1(φ/2 + π/2) 在相位上，但这里直接使用 U1 的等价参数映射
    """
    # Rz(φ) 等价于 U1(φ) 在功能上，因为两者都是相位门
    # 这里直接调用 U1 来实现 Rz
    return u1_polar_tensor(z0_batch, z1_batch, phi)

def hrh_polar_tensor(z0_batch: torch.Tensor, z1_batch: torch.Tensor, phi: float) -> Tuple[torch.Tensor, torch.Tensor]:
    """PyTorch 版本：HRH 门极坐标实现（H-Rz-H 组合门，int8向量，批量版本）
    HRH(φ) = H * Rz(φ) * H，表示围绕 Y 轴旋转 φ 角度
    """
    batch_size = z0_batch.shape[0]

    # 将极坐标转换为复数 (batch_size, )
    c0 = polar_to_complex_tensor(z0_batch.float())
    c1 = polar_to_complex_tensor(z1_batch.float())

    # 构造 HRH 矩阵：H * Rz(φ) * H
    # 首先构造 Rz(φ)
    exp_neg = torch.exp(torch.tensor(-1j * phi / 2, dtype=torch.complex64, device=device))
    exp_pos = torch.exp(torch.tensor(1j * phi / 2, dtype=torch.complex64, device=device))
    rz_mat = torch.stack([
        torch.stack([exp_neg, torch.tensor(0.0, dtype=torch.complex64, device=device)]),
        torch.stack([torch.tensor(0.0, dtype=torch.complex64, device=device), exp_pos])
    ])

    # HRH = H * Rz * H
    hrh_mat = torch.matmul(H_MAT_TORCH, torch.matmul(rz_mat, H_MAT_TORCH))

    # 应用矩阵到每个状态对
    result = torch.matmul(torch.stack([c0, c1], dim=1), hrh_mat.t())

    # 转换回极坐标 int8
    polar_result = complex_to_polar_tensor(result)
    return polar_result[:, 0].to(torch.int8), polar_result[:, 1].to(torch.int8)

def u1_polar_tensor(z0_batch: torch.Tensor, z1_batch: torch.Tensor, lambda_param: float) -> Tuple[torch.Tensor, torch.Tensor]:
    """PyTorch 版本：U1 门极坐标实现（int8向量，批量版本）"""
    batch_size = z0_batch.shape[0]

    # 将极坐标转换为复数 (batch_size, )
    c0 = polar_to_complex_tensor(z0_batch.float())
    c1 = polar_to_complex_tensor(z1_batch.float())

    # 构造 U1 矩阵
    exp_phase = torch.exp(torch.tensor(1j * lambda_param, dtype=torch.complex64, device=device))
    u1_mat = torch.stack([
        torch.stack([torch.tensor(1.0, dtype=torch.complex64, device=device), torch.tensor(0.0, dtype=torch.complex64, device=device)]),
        torch.stack([torch.tensor(0.0, dtype=torch.complex64, device=device), exp_phase])
    ])

    # 应用矩阵到每个状态对
    result = torch.matmul(torch.stack([c0, c1], dim=1), u1_mat.t())

    # 转换回极坐标 int8
    polar_result = complex_to_polar_tensor(result)
    return polar_result[:, 0].to(torch.int8), polar_result[:, 1].to(torch.int8)

def u2_polar_tensor(z0_batch: torch.Tensor, z1_batch: torch.Tensor, phi: float, lambda_param: float) -> Tuple[torch.Tensor, torch.Tensor]:
    """PyTorch 版本：U2 门极坐标实现（int8向量，批量版本）"""
    batch_size = z0_batch.shape[0]

    # 将极坐标转换为复数 (batch_size, )
    c0 = polar_to_complex_tensor(z0_batch.float())
    c1 = polar_to_complex_tensor(z1_batch.float())

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

    # 转换回极坐标 int8
    polar_result = complex_to_polar_tensor(result)
    return polar_result[:, 0].to(torch.int8), polar_result[:, 1].to(torch.int8)

def u3_polar_tensor(z0_batch: torch.Tensor, z1_batch: torch.Tensor, theta: float, phi: float, lambda_param: float) -> Tuple[torch.Tensor, torch.Tensor]:
    """PyTorch 版本：U3 门极坐标实现（int8向量，批量版本）"""
    batch_size = z0_batch.shape[0]

    # 将极坐标转换为复数 (batch_size, )
    c0 = polar_to_complex_tensor(z0_batch.float())
    c1 = polar_to_complex_tensor(z1_batch.float())

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

    # 转换回极坐标 int8
    polar_result = complex_to_polar_tensor(result)
    return polar_result[:, 0].to(torch.int8), polar_result[:, 1].to(torch.int8)

# ===== 控制门实现 =====

def apply_cnot_polar_tensor(polar_vec: torch.Tensor, control_idx: int, target_idx: int):
    """
    PyTorch 版本：应用 CNOT 门（极坐标 int8 向量版本）

    Args:
        polar_vec: 极坐标状态向量，形状 (N, 2)，dtype=torch.int8
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

def apply_cz_polar_tensor(polar_vec: torch.Tensor, qubit1_idx: int, qubit2_idx: int):
    """
    PyTorch 版本：应用 CZ 门（极坐标 int8 向量版本）

    Args:
        polar_vec: 极坐标状态向量，形状 (N, 2)，dtype=torch.int8
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
            polar_vec[i, 1] = encode_th_tensor(decode_th_tensor(polar_vec[i, 1].float()) + math.pi)

# ===== 控制门包装函数 =====

def apply_cy_polar_tensor(polar_vec: torch.Tensor, control_idx: int, target_idx: int):
    """PyTorch 版本：应用 CY 门（极坐标版本）"""
    apply_controlled_gate_polar_tensor(polar_vec, control_idx, target_idx, y_polar_tensor)

def apply_ch_polar_tensor(polar_vec: torch.Tensor, control_idx: int, target_idx: int):
    """PyTorch 版本：应用 CH 门（极坐标版本）"""
    apply_controlled_gate_polar_tensor(polar_vec, control_idx, target_idx, hadamard_polar_tensor)

def apply_cs_polar_tensor(polar_vec: torch.Tensor, control_idx: int, target_idx: int):
    """PyTorch 版本：应用 CS 门（极坐标版本）"""
    apply_controlled_gate_polar_tensor(polar_vec, control_idx, target_idx, s_polar_tensor)

def apply_ct_polar_tensor(polar_vec: torch.Tensor, control_idx: int, target_idx: int):
    """PyTorch 版本：应用 CT 门（极坐标版本）"""
    apply_controlled_gate_polar_tensor(polar_vec, control_idx, target_idx, t_polar_tensor)

def apply_crx_polar_tensor(polar_vec: torch.Tensor, control_idx: int, target_idx: int, theta: float):
    """PyTorch 版本：应用 CRx 门（极坐标版本）"""
    apply_controlled_gate_polar_tensor(polar_vec, control_idx, target_idx, rx_polar_tensor, theta)

def apply_cry_polar_tensor(polar_vec: torch.Tensor, control_idx: int, target_idx: int, theta: float):
    """PyTorch 版本：应用 CRy 门（极坐标版本）"""
    apply_controlled_gate_polar_tensor(polar_vec, control_idx, target_idx, ry_polar_tensor, theta)

def apply_crz_polar_tensor(polar_vec: torch.Tensor, control_idx: int, target_idx: int, phi: float):
    """PyTorch 版本：应用 CRz 门（极坐标版本）"""
    apply_controlled_gate_polar_tensor(polar_vec, control_idx, target_idx, rz_polar_tensor, phi)

def apply_cu1_polar_tensor(polar_vec: torch.Tensor, control_idx: int, target_idx: int, lambda_param: float):
    """PyTorch 版本：应用 CU1 门（极坐标版本）"""
    apply_controlled_gate_polar_tensor(polar_vec, control_idx, target_idx, u1_polar_tensor, lambda_param)

def apply_cu2_polar_tensor(polar_vec: torch.Tensor, control_idx: int, target_idx: int, phi: float, lambda_param: float):
    """PyTorch 版本：应用 CU2 门（极坐标版本）"""
    apply_controlled_gate_polar_tensor(polar_vec, control_idx, target_idx, u2_polar_tensor, phi, lambda_param)

def apply_cu3_polar_tensor(polar_vec: torch.Tensor, control_idx: int, target_idx: int, theta: float, phi: float, lambda_param: float):
    """PyTorch 版本：应用 CU3 门（极坐标版本）"""
    apply_controlled_gate_polar_tensor(polar_vec, control_idx, target_idx, u3_polar_tensor, theta, phi, lambda_param)

def apply_controlled_gate_polar_tensor(polar_vec: torch.Tensor, control_idx: int, target_idx: int, gate_func, *gate_params):
    """
    PyTorch 版本：应用控制门（极坐标 int8 向量版本）
    使用张量操作完全消除 for 循环

    Args:
        polar_vec: 极坐标状态向量，形状 (N, 2)，dtype=torch.int8
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

    # 批量提取状态对
    states0 = polar_vec[idx0_batch]  # 形状: (batch_size, 2)
    states1 = polar_vec[idx1_batch]  # 形状: (batch_size, 2)

    # 批量应用门函数
    if gate_params:
        new_states0, new_states1 = gate_func(states0, states1, *gate_params)
    else:
        new_states0, new_states1 = gate_func(states0, states1)

    # 将结果写回原向量
    polar_vec[idx0_batch] = new_states0
    polar_vec[idx1_batch] = new_states1

# ===== 核心门应用函数 =====

def apply_polar_gate_tensor(polar_vec: torch.Tensor, gate_func, *args, qubit_idx=0):
    """
    PyTorch 版本：对指定的比特位应用量子门（极坐标 int8 向量版本）
    使用张量操作替换 for 循环以提高性能

    Args:
        polar_vec: 极坐标状态向量，形状 (N, 2)，dtype=torch.int8
        gate_func: 门函数
        *args: 门函数需要的参数（如 theta, phi, lambda 等）
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

    # 批量提取状态对
    states0 = polar_vec[idx0_batch]  # 形状: (batch_size, 2)
    states1 = polar_vec[idx1_batch]  # 形状: (batch_size, 2)

    # 批量应用门函数
    if args:
        new_states0, new_states1 = gate_func(states0, states1, *args)
    else:
        new_states0, new_states1 = gate_func(states0, states1)

    # 将结果写回原向量
    polar_vec[idx0_batch] = new_states0
    polar_vec[idx1_batch] = new_states1

# ===== 门定义字典 =====

POLAR_GATES_TORCH = {
    'Y':  y_polar_tensor,
    'X':  x_polar_tensor,
    'S':  s_polar_tensor,
    'T':  t_polar_tensor,
    'Z':  z_polar_tensor,
    'Rx': rx_polar_tensor,
    'Ry': ry_polar_tensor,
    'Rz': rz_polar_tensor,
    'U1': u1_polar_tensor,
    'U2': u2_polar_tensor,
    'U3': u3_polar_tensor,
    'H': hadamard_polar_tensor,
    'HRH': hrh_polar_tensor
}

# ===== 统一的门应用函数 =====

def apply_gate_unified_tensor(polar_vec: torch.Tensor, gate_name: str, *gate_params, qubit_idx=0, control_idx=None, target_idx=None):
    """
    PyTorch 版本：统一的量子门应用函数（int8 向量版本）

    Args:
        polar_vec: 极坐标状态向量，形状 (N, 2)，dtype=torch.int8
        gate_name: 门名称
        *gate_params: 门函数需要的参数
        qubit_idx: 单比特门的比特索引
        control_idx: 控制比特索引（用于控制门）
        target_idx: 目标比特索引（用于控制门）
    """
    # 控制门（需要 control_idx 和 target_idx）
    controlled_gates = {
        'CX': apply_cnot_polar_tensor,
        'CY': apply_cy_polar_tensor,
        'CZ': apply_cz_polar_tensor,
        'CH': apply_ch_polar_tensor,
        'CS': apply_cs_polar_tensor,
        'CT': apply_ct_polar_tensor,
        'CRx': apply_crx_polar_tensor,
        'CRy': apply_cry_polar_tensor,
        'CRz': apply_crz_polar_tensor,
        'CU1': apply_cu1_polar_tensor,
        'CU2': apply_cu2_polar_tensor,
        'CU3': apply_cu3_polar_tensor,
    }

    if gate_name in controlled_gates:
        # 控制门
        if control_idx is None:
            control_idx = 0
        if target_idx is None:
            target_idx = 1

        func_polar = controlled_gates[gate_name]

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
            apply_cnot_polar_tensor(polar_vec, control_idx, target_idx)
        elif gate_name == 'CZ':
            apply_cz_polar_tensor(polar_vec, control_idx, target_idx)

    else:
        # 单比特门
        apply_polar_gate_tensor(polar_vec, POLAR_GATES_TORCH[gate_name], *gate_params, qubit_idx=qubit_idx)

# ===== 辅助函数 =====

def create_random_polar_state(n_amps: int) -> torch.Tensor:
    """创建随机极坐标 int8 状态向量"""
    # 创建随机复数向量
    real_parts = torch.randn(n_amps, device=device)
    imag_parts = torch.randn(n_amps, device=device)
    complex_vec = torch.complex(real_parts, imag_parts)

    # 归一化
    norm = torch.sqrt(torch.sum(torch.abs(complex_vec)**2))
    complex_vec = complex_vec / norm

    # 转换为极坐标 int8
    polar_vec = complex_to_polar_tensor(complex_vec)
    return polar_vec.to(torch.int8)

def polar_vec_to_string(polar_vec: torch.Tensor, indices=None) -> str:
    """将极坐标向量转换为字符串表示（支持int8和float向量）"""
    if indices is None:
        indices = range(len(polar_vec))

    result = []
    for i in indices:
        # 处理int8向量，转换为float进行解码
        if polar_vec.dtype == torch.int8:
            r = decode_r_tensor(polar_vec[i:i+1, 0].float()).item()
            th = decode_th_tensor(polar_vec[i:i+1, 1].float()).item()
        else:
            r = decode_r_tensor(polar_vec[i:i+1, 0]).item()
            th = decode_th_tensor(polar_vec[i:i+1, 1]).item()
        complex_val = complex(r * math.cos(th), r * math.sin(th))
        result.append(f"{i:02b}: {complex_val:.5f}")
    return " | ".join(result)

# ===== 序列生成功能 =====

def random_sequence_tensor(n: int, m: int, amp_n: int, shuffle: bool = False):
    """生成随机量子门序列（PyTorch 版本）"""
    seq = []
    # 单参数门
    single_param_gates = ['Rx', 'Ry', 'Rz', 'U1']
    # 多参数门
    multi_param_gates = ['U2', 'U3']
    # 无参数单比特门
    no_param_gates = ['X', 'Y', 'S', 'T', 'Z']
    # 无参数控制门
    no_param_controlled_gates = ['CX', 'CY', 'CZ', 'CH', 'CS', 'CT']
    # 单参数控制门
    single_param_controlled_gates = ['CRx', 'CRy', 'CRz', 'CU1']
    # 多参数控制门
    multi_param_controlled_gates = ['CU2', 'CU3']

    # 生成各种类型的门
    for _ in range(n):
        g = random.choice(no_param_gates)
        seq.append((g, '', []))

    for _ in range(n):
        g = random.choice(no_param_controlled_gates)
        seq.append((g, '', []))

    for _ in range(m):
        g = random.choice(single_param_gates + multi_param_gates)
        if g in single_param_gates:
            param = random.uniform(0, 2 * math.pi)
            seq.append((g, f'({param:.3f})', [param]))
        elif g == 'U2':
            phi = random.uniform(0, 2 * math.pi)
            lambda_param = random.uniform(0, 2 * math.pi)
            seq.append((g, f'({phi:.3f},{lambda_param:.3f})', [phi, lambda_param]))
        elif g == 'U3':
            theta = random.uniform(0, 2 * math.pi)
            phi = random.uniform(0, 2 * math.pi)
            lambda_param = random.uniform(0, 2 * math.pi)
            seq.append((g, f'({theta:.3f},{phi:.3f},{lambda_param:.3f})', [theta, phi, lambda_param]))

    for _ in range(m):
        g = random.choice(single_param_controlled_gates + multi_param_controlled_gates)
        if g in single_param_controlled_gates:
            param = random.uniform(0, 2 * math.pi)
            seq.append((g, f'({param:.3f})', [param]))
        elif g == 'CU2':
            phi = random.uniform(0, 2 * math.pi)
            lambda_param = random.uniform(0, 2 * math.pi)
            seq.append((g, f'({phi:.3f},{lambda_param:.3f})', [phi, lambda_param]))
        elif g == 'CU3':
            theta = random.uniform(0, 2 * math.pi)
            phi = random.uniform(0, 2 * math.pi)
            lambda_param = random.uniform(0, 2 * math.pi)
            seq.append((g, f'({theta:.3f},{phi:.3f},{lambda_param:.3f})', [theta, phi, lambda_param]))

    for _ in range(amp_n):
        g = random.choice(['H', 'Rx', 'Ry', 'Rz'])
        if g in ['Rx', 'Ry', 'Rz']:
            param = random.uniform(0, 2 * math.pi)
            seq.append((g, f'({param:.3f})', [param]))
        else:
            seq.append((g, '', []))

    if shuffle:
        random.shuffle(seq)
    return seq

# ===== 测试代码 =====

if __name__ == "__main__":
    print("PyTorch 极坐标量子计算完整测试")

    # 测试参数
    N_AMP = 32
    nqubit = int(math.log2(N_AMP))

    print(f"系统有 {nqubit} 个比特，状态向量长度为 {N_AMP}")

    # 创建随机初始状态
    polar_vec = create_random_polar_state(N_AMP)
    print("\n初始状态 (前8个元素):")
    print(polar_vec_to_string(polar_vec, range(8)))

    # 生成随机门序列
    seq = random_sequence_tensor(50, 50, 50, shuffle=True)
    print(f"\n随机门序列 (总共 {len(seq)} 个门):")
    print([s[0] + s[1] for s in seq[:10]], "...")  # 只显示前10个

    # 保存误差数据
    errors = []

    # 执行门序列
    for step, (name, param_str, gate_params) in enumerate(seq):
        # 检查是否是控制门
        is_controlled = name.startswith('C') or name in ['CNOT', 'CZ']

        if is_controlled:
            # 控制门：随机选择两个不同的比特位
            qubits = random.sample(range(nqubit), 2)
            control_idx, target_idx = qubits[0], qubits[1]
            apply_gate_unified_tensor(polar_vec, name, *gate_params,
                                    control_idx=control_idx, target_idx=target_idx)
            gate_info = f"{name}(控制={control_idx}, 目标={target_idx})"
        else:
            # 单比特门：随机选择一个比特位
            qubit_idx = random.randint(0, nqubit - 1)
            apply_gate_unified_tensor(polar_vec, name, *gate_params, qubit_idx=qubit_idx)
            gate_info = f"{name}(比特={qubit_idx})"

        # 每50步显示一次状态
        if step % 50 == 49 or step == len(seq) - 1:
            param_display = param_str if param_str else ''
            print(f"\nAfter {step+1} gates - Last: {gate_info}{param_display}:")
            print("当前状态 (前8个元素):")
            print(polar_vec_to_string(polar_vec, range(8)))

    print(f"\nPyTorch 版本测试完成！总共执行了 {len(seq)} 个量子门")
    print(f"使用了 {device} 设备")

    # 性能测试 - 所有门的 1000 次循环测试
    print("\n=== 性能测试 - 所有门的 1000 次循环测试 ===")
    import time

    # 创建测试状态
    test_vec = create_random_polar_state(64)  # 6 比特系统

    # 定义所有要测试的门
    gates_to_test = [
        ('X', 'X 门 (交换门)', []),
        ('Y', 'Y 门 (泡利-Y)', []),
        ('Z', 'Z 门 (泡利-Z)', []),
        ('S', 'S 门 (相位门)', []),
        ('T', 'T 门 (π/8相位门)', []),
        ('H', 'H 门 (Hadamard)', []),
        ('Rx', 'Rx 门 (X旋转)', [math.pi/4]),
        ('Ry', 'Ry 门 (Y旋转)', [math.pi/4]),
        ('Rz', 'Rz 门 (Z旋转)', [math.pi/4]),
        ('U1', 'U1 门 (单参数)', [math.pi/4]),
        ('HRH', 'HRH 门 (H-Rz-H 组合门)', [math.pi/4]),
        ('U2', 'U2 门 (两参数)', [math.pi/2.2, math.pi/3]),
        ('U3', 'U3 门 (三参数)', [math.pi/2.2, math.pi/3, math.pi/6]),
    ]

    print(f"在 6 比特系统 (64 状态) 上测试 1000 次应用每个门")
    print("-" * 70)

    for gate_name, description, params in gates_to_test:
        # 为每个门创建新的测试向量
        gate_test_vec = create_random_polar_state(64)

        # 执行 1000 次门应用
        torch.cuda.synchronize()  # 确保 GPU 操作完成
        start_time = time.time()

        for _ in range(1000):
            apply_gate_unified_tensor(gate_test_vec, gate_name, *params, qubit_idx=0)

        torch.cuda.synchronize()  # 确保所有操作完成
        end_time = time.time()

        total_time = end_time - start_time
        avg_time = total_time / 1000 * 1000  # 毫秒

        # 验证归一化
        complex_vec = polar_to_complex_tensor(gate_test_vec)
        norm = torch.sqrt(torch.sum(torch.abs(complex_vec)**2)).item()
        print(f"{gate_name} 门平均耗时: {avg_time:.2f} ms")
        print(f"最终状态归一化检查：{norm:.6f} (应接近 1.0)")

    print("\n=== 基本功能测试 ===")

    # 测试基本门
    basic_vec = create_random_polar_state(4)  # 2 比特系统
    print("2 比特系统初始状态:")
    print(polar_vec_to_string(basic_vec, range(4)))

    # 应用几个基本门
    test_gates = [
        ('H', [], 0),
        ('X', [], 1),
        ('CX', [], 0, 1),
        ('Ry', [math.pi/2], 0),
    ]

    for gate_name, params, *qubits in test_gates:
        if len(qubits) == 1:
            apply_gate_unified_tensor(basic_vec, gate_name, *params, qubit_idx=qubits[0])
            print(f"\n应用 {gate_name} 到比特 {qubits[0]}:")
        else:
            apply_gate_unified_tensor(basic_vec, gate_name, *params, control_idx=qubits[0], target_idx=qubits[1])
            print(f"\n应用 {gate_name} (控制={qubits[0]}, 目标={qubits[1]}):")
        print(polar_vec_to_string(basic_vec.float(), range(4)))
