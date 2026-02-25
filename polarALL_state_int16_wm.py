"""
纯极坐标版本的量子计算实现 - int16优化版本
使用 PyTorch 和 int16(int16幅度)+int8(相位) 编码的极坐标表示

✨ 主要优化：
1. 幅度使用单个 int16 编码（合并原有的两个 uint8 概念）
2. 相位使用 int8 编码以节省空间
3. 保持与 polarALL_state_3.py 完全兼容的接口

⚠️ 注意：
- 本文件使用 PyTorch 和混合精度编码 (int16 + int8)
- 用于在保持精度的同时优化存储空间
"""

import torch
import math
import cmath
import random
import numpy as np
from typing import List, Tuple
from utils import random_sequence_tensor, create_initial_vec_like_v2
from vector_withWM import PolarVector, PolarStateEncoded

# 设置设备并创建向量实例
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
vector = PolarVector(device, amplitude_dtype=torch.int16, phase_dtype=torch.int16)
#print(f"Using device: {device}")
#print(f"Polar vector info: {vector.get_info()}")


# ===== 量子门矩阵定义 (PyTorch 版本) =====

H_MAT_TORCH = torch.tensor([[1, 1], [1, -1]], dtype=torch.complex64, device=device) / math.sqrt(2)
S_MAT_TORCH = torch.tensor([[1, 0], [0, 1j]], dtype=torch.complex64, device=device)
T_MAT_TORCH = torch.tensor([[1, 0], [0, cmath.exp(1j * math.pi / 4)]], dtype=torch.complex64, device=device)
X_MAT_TORCH = torch.tensor([[0, 1], [1, 0]], dtype=torch.complex64, device=device)
Y_MAT_TORCH = torch.tensor([[0, -1j], [1j, 0]], dtype=torch.complex64, device=device)
Z_MAT_TORCH = torch.tensor([[1, 0], [0, -1]], dtype=torch.complex64, device=device)


# ===== PyTorch 版本的量子门函数 =====

def hadamard_polar_tensor(z0_batch: torch.Tensor, z1_batch: torch.Tensor, scale_batch: torch.Tensor,
                          amplitude_max=None, wm_a=None, **kwargs) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, float]:
    """PyTorch 版本：Hadamard 门极坐标实现（int16 + int8 混合版本，批量版本）"""
    batch_size = z0_batch.shape[0]

    # 门内解码：传入 amplitude_max/wm_a 而非依赖 vector.current_max
    c0 = vector.polar_to_complex_tensor(z0_batch, scale_batch, amplitude_max=amplitude_max, wm_a=wm_a)
    c1 = vector.polar_to_complex_tensor(z1_batch, scale_batch, amplitude_max=amplitude_max, wm_a=wm_a)

    # 应用 Hadamard 矩阵到每个状态对
    state_vector = torch.stack([c0, c1], dim=1)  # (batch_size, 2)
    h_mat = H_MAT_TORCH.to(dtype=state_vector.dtype, device=state_vector.device)
    result = torch.matmul(state_vector, h_mat)

    c0_result = result[:, 0]
    c1_result = result[:, 1]

    # WM 关键：用共享的 amplitude_max 编码 c0 和 c1，否则解码时用单一 max 会出错
    r0, r1 = torch.abs(c0_result), torch.abs(c1_result)
    shared_max = max(r0.max().item(), r1.max().item(), amplitude_max or 0.0)
    polar_result0, scale_result0 = vector.complex_to_polar_tensor(c0_result, amplitude_max=shared_max)
    polar_result1, scale_result1 = vector.complex_to_polar_tensor(c1_result, amplitude_max=shared_max)
    combined_scale = torch.maximum(scale_result0, scale_result1)
    return polar_result0.squeeze(), polar_result1.squeeze(), combined_scale.squeeze(), shared_max


def x_polar_tensor(z0_batch: torch.Tensor, z1_batch: torch.Tensor, scale_batch: torch.Tensor,
                   amplitude_max=None, wm_a=None, **kwargs) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, float]:
    """PyTorch 版本：X 门极坐标实现（int16 + int8 混合版本，批量版本）"""
    # X 门：交换完整的状态向量（包括极坐标和缩放信息），不改变 amplitude_max
    amp = amplitude_max if amplitude_max is not None else 1.0
    return z1_batch, z0_batch, scale_batch, amp


def y_polar_tensor(z0_batch: torch.Tensor, z1_batch: torch.Tensor, scale_batch: torch.Tensor,
                   amplitude_max=None, wm_a=None, **kwargs) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, float]:
    """PyTorch 版本：Y 门极坐标实现（int16 + int8 混合版本，批量版本）
    Y 门：|0⟩ -> i|1⟩, |1⟩ -> -i|0⟩
    在极坐标中：交换幅度，调整相位
    """
    result_z0 = z1_batch.clone()
    result_z1 = z0_batch.clone()
    result_z0[:, 1] = vector.add_phase_encoded(z1_batch[:, 1], -math.pi / 2)
    result_z1[:, 1] = vector.add_phase_encoded(z0_batch[:, 1], math.pi / 2)
    amp = amplitude_max if amplitude_max is not None else 1.0
    return result_z0, result_z1, scale_batch, amp


def z_polar_tensor(z0_batch: torch.Tensor, z1_batch: torch.Tensor, scale_batch: torch.Tensor,
                   amplitude_max=None, wm_a=None, **kwargs) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, float]:
    """PyTorch 版本：Z 门极坐标实现（int16 + int8 混合版本，批量版本）"""
    result_z0 = z0_batch.clone()
    result_z1 = z1_batch.clone()
    result_z1[:, 1] = vector.add_phase_encoded(z1_batch[:, 1], math.pi)
    amp = amplitude_max if amplitude_max is not None else 1.0
    return result_z0, result_z1, scale_batch, amp


def s_polar_tensor(z0_batch: torch.Tensor, z1_batch: torch.Tensor, scale_batch: torch.Tensor,
                   amplitude_max=None, wm_a=None, **kwargs) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, float]:
    """PyTorch 版本：S 门极坐标实现"""
    result_z0 = z0_batch.clone()
    result_z1 = z1_batch.clone()
    result_z1[:, 1] = vector.add_phase_encoded(z1_batch[:, 1], math.pi / 2)
    amp = amplitude_max if amplitude_max is not None else 1.0
    return result_z0, result_z1, scale_batch, amp


def t_polar_tensor(z0_batch: torch.Tensor, z1_batch: torch.Tensor, scale_batch: torch.Tensor,
                   amplitude_max=None, wm_a=None, **kwargs) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, float]:
    """PyTorch 版本：T 门极坐标实现"""
    result_z0 = z0_batch.clone()
    result_z1 = z1_batch.clone()
    result_z1[:, 1] = vector.add_phase_encoded(z1_batch[:, 1], math.pi / 4)
    amp = amplitude_max if amplitude_max is not None else 1.0
    return result_z0, result_z1, scale_batch, amp


def rx_polar_tensor(z0_batch: torch.Tensor, z1_batch: torch.Tensor, scale_batch: torch.Tensor, theta: float,
                    amplitude_max=None, wm_a=None, **kwargs) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, float]:
    """int16 + int8 混合版本：Rx 门极坐标实现（批量版本）"""
    c0 = vector.polar_to_complex_tensor(z0_batch, scale_batch, amplitude_max=amplitude_max, wm_a=wm_a)
    c1 = vector.polar_to_complex_tensor(z1_batch, scale_batch, amplitude_max=amplitude_max, wm_a=wm_a)
    device = c0.device
    dtype = c0.dtype
    c, s = math.cos(theta / 2), math.sin(theta / 2)
    rx_mat = torch.tensor([[c, -1j * s], [-1j * s, c]], dtype=dtype, device=device)
    result = torch.matmul(torch.stack([c0, c1], dim=1), rx_mat)
    c0_result, c1_result = result[:, 0], result[:, 1]
    r0, r1 = torch.abs(c0_result), torch.abs(c1_result)
    shared_max = max(r0.max().item(), r1.max().item(), amplitude_max or 0.0)
    polar_result0, scale_result0 = vector.complex_to_polar_tensor(c0_result, amplitude_max=shared_max)
    polar_result1, scale_result1 = vector.complex_to_polar_tensor(c1_result, amplitude_max=shared_max)
    combined_scale = torch.maximum(scale_result0, scale_result1)
    return polar_result0.squeeze(), polar_result1.squeeze(), combined_scale.squeeze(), shared_max


def ry_polar_tensor(z0_batch: torch.Tensor, z1_batch: torch.Tensor, scale_batch: torch.Tensor, theta: float,
                    amplitude_max=None, wm_a=None, **kwargs) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, float]:
    """int16 + int8 混合版本：Ry 门极坐标实现（批量版本）"""
    c0 = vector.polar_to_complex_tensor(z0_batch, scale_batch, amplitude_max=amplitude_max, wm_a=wm_a)
    c1 = vector.polar_to_complex_tensor(z1_batch, scale_batch, amplitude_max=amplitude_max, wm_a=wm_a)
    device = c0.device
    dtype = c0.dtype
    c, s = math.cos(theta / 2), math.sin(theta / 2)
    ry_mat = torch.tensor([[c, -s], [s, c]], dtype=dtype, device=device)
    result = torch.matmul(torch.stack([c0, c1], dim=1), ry_mat)
    c0_result, c1_result = result[:, 0], result[:, 1]
    r0, r1 = torch.abs(c0_result), torch.abs(c1_result)
    shared_max = max(r0.max().item(), r1.max().item(), amplitude_max or 0.0)
    polar_result0, scale_result0 = vector.complex_to_polar_tensor(c0_result, amplitude_max=shared_max)
    polar_result1, scale_result1 = vector.complex_to_polar_tensor(c1_result, amplitude_max=shared_max)
    combined_scale = torch.maximum(scale_result0, scale_result1)
    return polar_result0.squeeze(), polar_result1.squeeze(), combined_scale.squeeze(), shared_max


def rz_polar_tensor(z0_batch: torch.Tensor, z1_batch: torch.Tensor, scale_batch: torch.Tensor, phi: float,
                    amplitude_max=None, wm_a=None, **kwargs) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, float]:
    """Rz(φ) 门：直接相位操作，无 decode/encode"""
    result_z0 = z0_batch.clone()
    result_z1 = z1_batch.clone()
    result_z0[:, 1] = vector.add_phase_encoded(z0_batch[:, 1], -phi / 2)
    result_z1[:, 1] = vector.add_phase_encoded(z1_batch[:, 1], phi / 2)
    amp = amplitude_max if amplitude_max is not None else 1.0
    return result_z0, result_z1, scale_batch, amp


def u1_polar_tensor(z0_batch: torch.Tensor, z1_batch: torch.Tensor, scale_batch: torch.Tensor, lambda_param: float,
                    amplitude_max=None, wm_a=None, **kwargs) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, float]:
    """U1(λ) 只给 |1⟩ 添加相位 λ"""
    result_z0 = z0_batch.clone()
    result_z1 = z1_batch.clone()
    result_z1[:, 1] = vector.add_phase_encoded(z1_batch[:, 1], lambda_param)
    amp = amplitude_max if amplitude_max is not None else 1.0
    return result_z0, result_z1, scale_batch, amp


def u2_polar_tensor(z0_batch: torch.Tensor, z1_batch: torch.Tensor, scale_batch: torch.Tensor, phi: float, lambda_param: float,
                    amplitude_max=None, wm_a=None, **kwargs) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, float]:
    """U2 门极坐标实现"""
    c0 = vector.polar_to_complex_tensor(z0_batch, scale_batch, amplitude_max=amplitude_max, wm_a=wm_a)
    c1 = vector.polar_to_complex_tensor(z1_batch, scale_batch, amplitude_max=amplitude_max, wm_a=wm_a)
    dtype = c0.dtype
    device = c0.device
    inv_sqrt2 = 1 / math.sqrt(2)
    exp_phi = torch.exp(torch.tensor(1j * phi, dtype=dtype, device=device))
    exp_lambda = torch.exp(torch.tensor(1j * lambda_param, dtype=dtype, device=device))
    u2_mat = torch.stack([
        torch.stack([torch.tensor(inv_sqrt2, dtype=dtype, device=device), -inv_sqrt2 * exp_lambda]),
        torch.stack([inv_sqrt2 * exp_phi, inv_sqrt2 * exp_phi * exp_lambda])
    ])
    result = torch.matmul(torch.stack([c0, c1], dim=1), u2_mat)
    c0_result, c1_result = result[:, 0], result[:, 1]
    r0, r1 = torch.abs(c0_result), torch.abs(c1_result)
    shared_max = max(r0.max().item(), r1.max().item(), amplitude_max or 0.0)
    polar_result0, scale_result0 = vector.complex_to_polar_tensor(c0_result, amplitude_max=shared_max)
    polar_result1, scale_result1 = vector.complex_to_polar_tensor(c1_result, amplitude_max=shared_max)
    combined_scale = torch.maximum(scale_result0, scale_result1)
    return polar_result0.squeeze(), polar_result1.squeeze(), combined_scale.squeeze(), shared_max


def u3_polar_tensor(z0_batch: torch.Tensor, z1_batch: torch.Tensor, scale_batch: torch.Tensor, theta: float, phi: float, lambda_param: float,
                    amplitude_max=None, wm_a=None, **kwargs) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, float]:
    """U3 门极坐标实现"""
    c0 = vector.polar_to_complex_tensor(z0_batch, scale_batch, amplitude_max=amplitude_max, wm_a=wm_a)
    c1 = vector.polar_to_complex_tensor(z1_batch, scale_batch, amplitude_max=amplitude_max, wm_a=wm_a)
    dtype = c0.dtype
    device = c0.device
    c_val, s_val = math.cos(theta / 2), math.sin(theta / 2)
    exp_phi = torch.exp(torch.tensor(1j * phi, dtype=dtype, device=device))
    exp_lambda = torch.exp(torch.tensor(1j * lambda_param, dtype=dtype, device=device))
    u3_mat = torch.stack([
        torch.stack([torch.tensor(c_val, dtype=dtype, device=device), -exp_lambda * s_val]),
        torch.stack([exp_phi * s_val, exp_phi * exp_lambda * c_val])
    ])
    result = torch.matmul(torch.stack([c0, c1], dim=1), u3_mat)
    c0_result, c1_result = result[:, 0], result[:, 1]
    r0, r1 = torch.abs(c0_result), torch.abs(c1_result)
    shared_max = max(r0.max().item(), r1.max().item(), amplitude_max or 0.0)
    polar_result0, scale_result0 = vector.complex_to_polar_tensor(c0_result, amplitude_max=shared_max)
    polar_result1, scale_result1 = vector.complex_to_polar_tensor(c1_result, amplitude_max=shared_max)
    combined_scale = torch.maximum(scale_result0, scale_result1)
    return polar_result0.squeeze(), polar_result1.squeeze(), combined_scale.squeeze(), shared_max


# ===== 极坐标门字典 =====

POLAR_GATES_TORCH = {
    'H': hadamard_polar_tensor,
    'X': x_polar_tensor,
    'Y': y_polar_tensor,
    'Z': z_polar_tensor,
    'S': s_polar_tensor,
    'T': t_polar_tensor,
    'Rx': rx_polar_tensor,
    'Ry': ry_polar_tensor,
    'Rz': rz_polar_tensor,
    'U1': u1_polar_tensor,
    'U2': u2_polar_tensor,
    'U3': u3_polar_tensor,
}


# ===== 统一门应用函数 =====

def apply_polar_gate_tensor(polar_vec: torch.Tensor, scale_vec: torch.Tensor,
                          gate_func, *gate_params, qubit_idx: int = 0,
                          amplitude_max: float = None, wm_a: float = None) -> Tuple[torch.Tensor, torch.Tensor, float]:
    """
    统一的极坐标量子门应用函数（int16 + int8 混合版本）
    使用正确的张量积结构进行状态配对

    Args:
        polar_vec: 极坐标状态向量
        scale_vec: 缩放向量
        gate_func: 门函数，返回 (s0, s1, scale, amplitude_max)
        qubit_idx: 目标量子比特索引
        amplitude_max: WM 解码时的幅度最大值
        wm_a: WM 参数

    Returns:
        (new_polar_vec, new_scale_vec, new_amplitude_max)
    """
    n = polar_vec.shape[0]
    n_qubits = int(math.log2(n))
    assert 2 ** n_qubits == n, f"状态向量长度必须是 2 的幂次，当前长度: {n}"
    assert 0 <= qubit_idx < n_qubits, f"比特位索引必须在 [0, {n_qubits}) 范围内"

    # 正确的量子计算门应用逻辑：基于张量积结构的状态配对
    # 对于第 qubit_idx 个比特，配对间隔是 2^qubit_idx，每个块的大小是 2^(qubit_idx+1)

    step = 1 << qubit_idx      # 配对间隔 = 2^qubit_idx
    block_size = step << 1     # 块大小 = 2^(qubit_idx+1)

    # 使用张量操作收集所有需要处理的状态对
    idx0_list = []
    idx1_list = []

    # 遍历所有块
    for base in range(0, n, block_size):
        # 在每个块内，配对状态
        for offset in range(step):
            idx0 = base + offset
            idx1 = base + offset + step
            if idx1 < n:  # 确保索引有效
                idx0_list.append(idx0)
                idx1_list.append(idx1)

    # 转换为张量
    if len(idx0_list) > 0:
        idx0_batch = torch.tensor(idx0_list, dtype=torch.long, device=device)
        idx1_batch = torch.tensor(idx1_list, dtype=torch.long, device=device)

        # 批量提取完整的状态向量
        states0 = polar_vec[idx0_batch]  # 形状: (batch_size, 2) [幅度, 相位]
        states1 = polar_vec[idx1_batch]  # 形状: (batch_size, 2) [幅度, 相位]
        scale_batch = scale_vec[idx0_batch]  # 对应的缩放因子

        # 应用门函数（传入 amplitude_max/wm_a 用于门内解码，返回 new_amplitude_max）
        ret = gate_func(states0, states1, scale_batch, *gate_params, amplitude_max=amplitude_max, wm_a=wm_a)
        new_states0, new_states1, new_scale_batch, new_amplitude_max = ret[:4]

        # 将结果写回原向量
        polar_vec[idx0_batch] = new_states0
        polar_vec[idx1_batch] = new_states1
        scale_vec[idx0_batch] = new_scale_batch

        # 门后更新 amplitude_max
        out_amplitude_max = new_amplitude_max

    return polar_vec, scale_vec, out_amplitude_max


def apply_gate_polar_tensor(state: PolarStateEncoded,
                          gate_name: str, gate_params: List[float] = None,
                          control_idx: int = None, target_idx: int = None) -> PolarStateEncoded:
    """
    高级门应用函数，支持控制门（基于PolarStateEncoded）

    Args:
        state: PolarStateEncoded状态对象
        gate_name: 门名称 ('H', 'X', 'Y', 'Z', 'S', 'T', 'Rx', 'Ry', 'Rz', 'CNOT', 'CZ')
        gate_params: 门参数列表
        control_idx: 控制量子比特索引（用于控制门）
        target_idx: 目标量子比特索引

    Returns:
        new_state: 更新后的PolarStateEncoded状态对象
    """
    if gate_params is None:
        gate_params = []

    # 获取当前状态的向量
    polar_vec = state.get_polar_vec()
    scale_vec = state.get_scale_vec()

    # 处理门名称别名
    if gate_name == 'CX':
        gate_name = 'CNOT'

    controlled_gates_no_param = ['CNOT', 'CZ', 'CH', 'CY', 'CS', 'CT']
    controlled_gates_single_param = ['CRx', 'CRy', 'CRz', 'CU1']
    controlled_gates_multi_param = ['CU2', 'CU3']

    if gate_name in controlled_gates_no_param + controlled_gates_single_param + controlled_gates_multi_param:
        # 控制门
        if control_idx is None:
            raise ValueError(f"控制门 {gate_name} 需要指定 control_idx")
        if target_idx is None:
            raise ValueError(f"控制门 {gate_name} 需要指定 target_idx")

        amp_max = state.get_amplitude_max()
        wm_a_val = state.get_wm_a()
        if gate_name == 'CNOT':
            apply_cnot_polar_tensor(polar_vec, scale_vec, control_idx, target_idx)
            new_amp_max = amp_max
        elif gate_name == 'CZ':
            apply_cz_polar_tensor(polar_vec, scale_vec, control_idx, target_idx)
            new_amp_max = amp_max
        elif gate_name == 'CH':
            polar_vec, scale_vec, new_amp_max = apply_ch_polar_tensor(polar_vec, scale_vec, control_idx, target_idx, amplitude_max=amp_max, wm_a=wm_a_val)
        elif gate_name == 'CY':
            polar_vec, scale_vec, new_amp_max = apply_cy_polar_tensor(polar_vec, scale_vec, control_idx, target_idx, amplitude_max=amp_max, wm_a=wm_a_val)
        elif gate_name == 'CS':
            polar_vec, scale_vec, new_amp_max = apply_cs_polar_tensor(polar_vec, scale_vec, control_idx, target_idx, amplitude_max=amp_max, wm_a=wm_a_val)
        elif gate_name == 'CT':
            polar_vec, scale_vec, new_amp_max = apply_ct_polar_tensor(polar_vec, scale_vec, control_idx, target_idx, amplitude_max=amp_max, wm_a=wm_a_val)
        elif gate_name == 'CRx':
            if not gate_params:
                raise ValueError(f"CRx 需要参数 theta")
            polar_vec, scale_vec, new_amp_max = apply_crx_polar_tensor(polar_vec, scale_vec, control_idx, target_idx, gate_params[0], amplitude_max=amp_max, wm_a=wm_a_val)
        elif gate_name == 'CRy':
            if not gate_params:
                raise ValueError(f"CRy 需要参数 theta")
            polar_vec, scale_vec, new_amp_max = apply_cry_polar_tensor(polar_vec, scale_vec, control_idx, target_idx, gate_params[0], amplitude_max=amp_max, wm_a=wm_a_val)
        elif gate_name == 'CRz':
            if not gate_params:
                raise ValueError(f"CRz 需要参数 phi")
            polar_vec, scale_vec, new_amp_max = apply_crz_polar_tensor(polar_vec, scale_vec, control_idx, target_idx, gate_params[0], amplitude_max=amp_max, wm_a=wm_a_val)
        elif gate_name == 'CU1':
            if not gate_params:
                raise ValueError(f"CU1 需要参数 lambda")
            polar_vec, scale_vec, new_amp_max = apply_cu1_polar_tensor(polar_vec, scale_vec, control_idx, target_idx, gate_params[0], amplitude_max=amp_max, wm_a=wm_a_val)
        elif gate_name == 'CU2':
            if len(gate_params) < 2:
                raise ValueError(f"CU2 需要参数 phi, lambda")
            polar_vec, scale_vec, new_amp_max = apply_cu2_polar_tensor(polar_vec, scale_vec, control_idx, target_idx, gate_params[0], gate_params[1], amplitude_max=amp_max, wm_a=wm_a_val)
        elif gate_name == 'CU3':
            if len(gate_params) < 3:
                raise ValueError(f"CU3 需要参数 theta, phi, lambda")
            polar_vec, scale_vec, new_amp_max = apply_cu3_polar_tensor(polar_vec, scale_vec, control_idx, target_idx, gate_params[0], gate_params[1], gate_params[2], amplitude_max=amp_max, wm_a=wm_a_val)
    else:
        # 单比特门
        polar_vec, scale_vec, new_amp_max = apply_polar_gate_tensor(
            polar_vec, scale_vec, POLAR_GATES_TORCH[gate_name], *gate_params,
            qubit_idx=target_idx or 0, amplitude_max=state.get_amplitude_max(), wm_a=state.get_wm_a())

    # 门后更新 polar_vec、scale_vec、amplitude_max
    state.update_polar_vec(polar_vec, scale_vec, amplitude_max=new_amp_max)
    return state


def apply_controlled_gate_polar_tensor(polar_vec: torch.Tensor, scale_vec: torch.Tensor,
                                     control_idx: int, target_idx: int, gate_func, *gate_params,
                                     amplitude_max: float = None, wm_a: float = None) -> Tuple[torch.Tensor, torch.Tensor, float]:
    """
    应用控制门（int16 + int8 混合版本）
    返回 (polar_vec, scale_vec, new_amplitude_max)
    """
    n = polar_vec.shape[0]
    n_qubit = int(math.log2(n))
    assert 2 ** n_qubit == n
    assert control_idx != target_idx
    assert 0 <= control_idx < n_qubit and 0 <= target_idx < n_qubit

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

    out_amplitude_max = amplitude_max if amplitude_max is not None else 1.0
    if len(idx0_batch) == 0:
        return polar_vec, scale_vec, out_amplitude_max

    states0 = polar_vec[idx0_batch]
    states1 = polar_vec[idx1_batch]
    scale_batch = scale_vec[idx0_batch]
    ret = gate_func(states0, states1, scale_batch, *gate_params, amplitude_max=amplitude_max, wm_a=wm_a)
    new_states0, new_states1, new_scale_batch, new_amplitude_max = ret[:4]
    polar_vec[idx0_batch] = new_states0
    polar_vec[idx1_batch] = new_states1
    scale_vec[idx0_batch] = new_scale_batch
    out_amplitude_max = new_amplitude_max
    return polar_vec, scale_vec, out_amplitude_max


def apply_cnot_polar_tensor(polar_vec: torch.Tensor, scale_vec: torch.Tensor, control_idx: int, target_idx: int):
    """
    应用 CNOT 门（与 polarALL_state_3 一致）
    当控制比特为 |1⟩ 时，翻转目标比特（交换 |...0⟩ 与 |...1⟩ 的幅度）
    """
    n = polar_vec.shape[0]
    n_qubit = int(math.log2(n))
    assert 2 ** n_qubit == n
    assert control_idx != target_idx
    assert 0 <= control_idx < n_qubit and 0 <= target_idx < n_qubit

    control_mask = 1 << control_idx
    target_mask = 1 << target_idx

    indices_to_swap = []
    for i in range(n):
        if (i & control_mask) != 0:  # 控制比特为 1
            j = i ^ target_mask  # 翻转目标比特
            if i < j:  # 避免重复处理
                indices_to_swap.append((i, j))

    for i, j in indices_to_swap:
        temp = polar_vec[i].clone()
        temp_scale = scale_vec[i].clone()
        polar_vec[i] = polar_vec[j]
        scale_vec[i] = scale_vec[j]
        polar_vec[j] = temp
        scale_vec[j] = temp_scale


def apply_ch_polar_tensor(polar_vec: torch.Tensor, scale_vec: torch.Tensor, control_idx: int, target_idx: int,
                          amplitude_max=None, wm_a=None):
    """应用 CH 门（int16 + int8 混合版本）"""
    return apply_controlled_gate_polar_tensor(polar_vec, scale_vec, control_idx, target_idx, hadamard_polar_tensor,
                                              amplitude_max=amplitude_max, wm_a=wm_a)


def apply_cs_polar_tensor(polar_vec: torch.Tensor, scale_vec: torch.Tensor, control_idx: int, target_idx: int,
                          amplitude_max=None, wm_a=None):
    """应用 CS 门（int16 + int8 混合版本）"""
    return apply_controlled_gate_polar_tensor(polar_vec, scale_vec, control_idx, target_idx, s_polar_tensor,
                                              amplitude_max=amplitude_max, wm_a=wm_a)


def apply_ct_polar_tensor(polar_vec: torch.Tensor, scale_vec: torch.Tensor, control_idx: int, target_idx: int,
                          amplitude_max=None, wm_a=None):
    """应用 CT 门（int16 + int8 混合版本）"""
    return apply_controlled_gate_polar_tensor(polar_vec, scale_vec, control_idx, target_idx, t_polar_tensor,
                                              amplitude_max=amplitude_max, wm_a=wm_a)


def apply_cy_polar_tensor(polar_vec: torch.Tensor, scale_vec: torch.Tensor, control_idx: int, target_idx: int,
                          amplitude_max=None, wm_a=None):
    """应用 CY 门（与 polarALL_state_3 一致）"""
    return apply_controlled_gate_polar_tensor(polar_vec, scale_vec, control_idx, target_idx, y_polar_tensor,
                                              amplitude_max=amplitude_max, wm_a=wm_a)


def apply_crx_polar_tensor(polar_vec: torch.Tensor, scale_vec: torch.Tensor, control_idx: int, target_idx: int, theta: float,
                           amplitude_max=None, wm_a=None):
    """应用 CRx 门（与 polarALL_state_3 一致）"""
    return apply_controlled_gate_polar_tensor(polar_vec, scale_vec, control_idx, target_idx, rx_polar_tensor, theta,
                                              amplitude_max=amplitude_max, wm_a=wm_a)


def apply_cry_polar_tensor(polar_vec: torch.Tensor, scale_vec: torch.Tensor, control_idx: int, target_idx: int, theta: float,
                           amplitude_max=None, wm_a=None):
    """应用 CRy 门（与 polarALL_state_3 一致）"""
    return apply_controlled_gate_polar_tensor(polar_vec, scale_vec, control_idx, target_idx, ry_polar_tensor, theta,
                                              amplitude_max=amplitude_max, wm_a=wm_a)


def apply_crz_polar_tensor(polar_vec: torch.Tensor, scale_vec: torch.Tensor, control_idx: int, target_idx: int, phi: float,
                           amplitude_max=None, wm_a=None):
    """应用 CRz 门（与 polarALL_state_3 一致）"""
    return apply_controlled_gate_polar_tensor(polar_vec, scale_vec, control_idx, target_idx, rz_polar_tensor, phi,
                                              amplitude_max=amplitude_max, wm_a=wm_a)


def apply_cu1_polar_tensor(polar_vec: torch.Tensor, scale_vec: torch.Tensor, control_idx: int, target_idx: int, lambda_param: float,
                          amplitude_max=None, wm_a=None):
    """应用 CU1 门（与 polarALL_state_3 一致）"""
    return apply_controlled_gate_polar_tensor(polar_vec, scale_vec, control_idx, target_idx, u1_polar_tensor, lambda_param,
                                              amplitude_max=amplitude_max, wm_a=wm_a)


def apply_cu2_polar_tensor(polar_vec: torch.Tensor, scale_vec: torch.Tensor, control_idx: int, target_idx: int, phi: float, lambda_param: float,
                          amplitude_max=None, wm_a=None):
    """应用 CU2 门（与 polarALL_state_3 一致）"""
    return apply_controlled_gate_polar_tensor(polar_vec, scale_vec, control_idx, target_idx, u2_polar_tensor, phi, lambda_param,
                                              amplitude_max=amplitude_max, wm_a=wm_a)


def apply_cu3_polar_tensor(polar_vec: torch.Tensor, scale_vec: torch.Tensor, control_idx: int, target_idx: int, theta: float, phi: float, lambda_param: float,
                           amplitude_max=None, wm_a=None):
    """应用 CU3 门（与 polarALL_state_3 一致）"""
    return apply_controlled_gate_polar_tensor(polar_vec, scale_vec, control_idx, target_idx, u3_polar_tensor, theta, phi, lambda_param,
                                              amplitude_max=amplitude_max, wm_a=wm_a)


def apply_cz_polar_tensor(polar_vec: torch.Tensor, scale_vec: torch.Tensor,
                         control_idx: int, target_idx: int) -> None:
    """
    应用 CZ 门（与 polarALL_state_3 完全一致）
    CZ 门：当控制比特与目标比特都为 |1⟩ 时，相位翻转 π
    """
    n = polar_vec.shape[0]
    n_qubit = int(math.log2(n))
    assert 2 ** n_qubit == n
    assert control_idx != target_idx
    assert 0 <= control_idx < n_qubit and 0 <= target_idx < n_qubit

    mask1 = 1 << control_idx
    mask2 = 1 << target_idx

    for i in range(n):
        if (i & mask1) != 0 and (i & mask2) != 0:  # 两个比特都为 1
            polar_vec[i, 1] = vector.add_phase_encoded(polar_vec[i:i+1, 1], math.pi)[0]


# ===== 辅助函数 =====

def create_random_polar_state(n_amps: int) -> PolarStateEncoded:
    """创建随机极坐标 int16 + int8 状态向量。使用 create_random_state 编码后的 vector.current_max 作为 amplitude_max"""
    polar_vec, scale_vec = vector.create_random_state(n_amps)
    amp_max = vector.current_max if vector.current_max is not None else 1.0
    return PolarStateEncoded(polar_vec, scale_vec, amp_max, vector.a, vector)


def polar_vec_to_string(polar_vec: torch.Tensor, scale_vec: torch.Tensor = None, indices=None) -> str:
    """将 int16 + int8 混合极坐标向量转换为字符串表示"""
    return vector.to_string(polar_vec, scale_vec, indices)


# ===== 核心处理函数 =====

def process_sequence_polar(initial_vec: List[complex], seq: List[Tuple], verbose: bool = False) -> Tuple[PolarStateEncoded, List[PolarStateEncoded]]:
    """
    处理量子门序列（int16 + int8 混合版本），记录每一步的状态变化

    Args:
        initial_vec: 初始复数状态向量
        seq: 量子门序列，格式为 [(gate_name, param_str, params, qubit_info...), ...]
            对于单比特门: (gate_name, param_str, params, qubit_idx)
            对于控制门: (gate_name, param_str, params, control_idx, target_idx)
        verbose: 是否打印详细信息

    Returns:
        (final_state, state_history): 最终PolarStateEncoded对象和每一步的PolarStateEncoded状态历史
    """
    n_amps = len(initial_vec)
    n_qubits = int(math.log2(n_amps))
    assert 2 ** n_qubits == n_amps, f"状态向量长度必须是 2 的幂，当前: {n_amps}"

    # 将初始复数向量转换为 PyTorch 张量并编码为极坐标格式
    complex_tensor = torch.tensor(initial_vec, dtype=torch.complex64, device=device)

    # 转换为极坐标格式（返回PolarStateEncoded对象）
    initial_state = vector.complex_to_polar_state(complex_tensor)

    # 记录状态历史
    state_history = [initial_state.clone()]  # 初始状态

    if verbose:
        print(f"处理 {len(seq)} 个量子门，系统有 {n_qubits} 个比特")
        print(f"初始状态（前8个）:")
        print(polar_vec_to_string(initial_state.get_polar_vec(), initial_state.get_scale_vec(), range(min(8, n_amps))))

    # 处理每个门
    current_state = initial_state
    for step, gate_tuple in enumerate(seq):
        # 解析序列格式：支持不同的参数数量
        if len(gate_tuple) == 4:
            gate_name = gate_tuple[0]
            gate_params = gate_tuple[2] if len(gate_tuple) > 2 and isinstance(gate_tuple[2], list) else []
            if gate_name.startswith('C') or gate_name in ['CNOT', 'CZ']:
                # 控制门 4元组: (name, params, control_idx, target_idx)
                control_idx, target_idx = gate_tuple[2], gate_tuple[3]
                is_controlled = True
            else:
                # 单比特门: (name, param_str/params, _, qubit_idx)
                qubit_idx = gate_tuple[3]
                control_idx = None
                target_idx = qubit_idx
                is_controlled = False
        elif len(gate_tuple) == 5:
            # 控制门 5元组: (name, param_str, params, control_idx, target_idx)
            gate_name, _, gate_params, control_idx, target_idx = gate_tuple
            is_controlled = True
        else:
            raise ValueError(f"无效的序列格式: {gate_tuple}")

        # 检查是否是控制门（额外确认）
        is_controlled = is_controlled or (gate_name.startswith('C') or gate_name in ['CNOT', 'CZ'])

        if is_controlled:
            # 控制门：使用序列中指定的控制和目标比特位
            current_state = apply_gate_polar_tensor(current_state, gate_name, gate_params, control_idx, target_idx)
            gate_info = f"{gate_name}(控制={control_idx}, 目标={target_idx})"
        else:
            # 单比特门：使用序列中指定的比特位
            current_state = apply_gate_polar_tensor(current_state, gate_name, gate_params, target_idx=qubit_idx)
            gate_info = f"{gate_name}(比特={qubit_idx})"

        # 记录这一步的状态
        state_history.append(current_state.clone())

        if verbose:
            print(f"门 {step + 1}: {gate_info}")
            if step < 2 or step % 10 == 0:  # 只显示前几个和每10个门的状态
                print(f"  状态（前8个）: {polar_vec_to_string(current_state.get_polar_vec(), current_state.get_scale_vec(), range(min(8, n_amps)))}")

    if verbose:
        # 计算最终归一化
        complex_vec = current_state.decode_state()
        norm = torch.sqrt(torch.sum(torch.abs(complex_vec)**2)).item()
        print(f"最终归一化误差: {abs(norm - 1.0):.2e}")

    return current_state, state_history


def benchmark_polar_int16(n_qubits: int = 4, n_sequences: int = 100, sequence_length: int = 50) -> dict:
    """
    基准测试 int16 + int8 混合版本的性能

    Args:
        n_qubits: 量子比特数量
        n_sequences: 测试序列数量
        sequence_length: 每个序列的长度

    Returns:
        性能测试结果字典
    """
    print(f"Benchmarking int16 + int8 polar quantum simulation:")
    print(f"  {n_qubits} qubits, {n_sequences} sequences, {sequence_length} gates each")

    import time
    start_time = time.time()

    results = []

    for seq_idx in range(n_sequences):
        if seq_idx % 10 == 0:
            print(f"  Processing sequence {seq_idx + 1}/{n_sequences}...")

        # 创建随机初始状态
        n_amps = 2**n_qubits
        initial_vec = [complex(random.random(), random.random()) for _ in range(n_amps)]
        # 归一化
        norm = math.sqrt(sum(abs(x)**2 for x in initial_vec))
        initial_vec = [x/norm for x in initial_vec]

        # 创建随机序列
        seq = []
        for _ in range(sequence_length):
            gate_name = random.choice(['H', 'X', 'Y', 'Z', 'S', 'T'])
            if gate_name in ['Rx', 'Ry', 'Rz']:
                gate_params = [random.uniform(0, 2*math.pi)]
            else:
                gate_params = []
            control_idx = None
            target_idx = random.randint(0, n_qubits-1)
            seq.append((gate_name, gate_params, control_idx, target_idx))

        # 处理序列
        final_state, _ = process_sequence_polar(initial_vec, seq, verbose=False)
        results.append(final_state)

    end_time = time.time()
    total_time = end_time - start_time

    return {
        'n_qubits': n_qubits,
        'n_sequences': n_sequences,
        'sequence_length': sequence_length,
        'total_time': total_time,
        'time_per_sequence': total_time / n_sequences,
        'time_per_gate': total_time / (n_sequences * sequence_length),
        'results': results
    }


# ===== 主函数 =====

if __name__ == "__main__":
    print("Int16 + Int8 Mixed Precision Polar Quantum Simulator")
    print("=" * 60)

    # 基本功能测试
    print("Testing basic functionality...")

    # 创建测试状态
    test_state = create_random_polar_state(4)
    print(f"Created random 2-qubit state with shape: {test_state.get_polar_vec().shape}")
    print(f"Data types: r={vector.amplitude_dtype}, theta={vector.phase_dtype}")
    print(f"State string: {polar_vec_to_string(test_state.get_polar_vec(), test_state.get_scale_vec())}")

    # 测试量子门
    print("\nTesting quantum gates...")
    test_state = apply_gate_polar_tensor(test_state, 'H', target_idx=0)
    print(f"After H gate: {polar_vec_to_string(test_state.get_polar_vec(), test_state.get_scale_vec())}")

    test_state = apply_gate_polar_tensor(test_state, 'X', target_idx=1)
    print(f"After X gate: {polar_vec_to_string(test_state.get_polar_vec(), test_state.get_scale_vec())}")

    # 序列处理测试
    print("\nTesting sequence processing...")
    initial_vec = [1.0, 0.0, 0.0, 0.0]  # |00⟩ 状态
    seq = [
        ('H', [], None, 0),     # 对第一个量子比特应用 H 门
        ('X', [], None, 1),     # 对第二个量子比特应用 X 门
        ('CNOT', [], 0, 1),     # CNOT 门，控制=0，目标=1
    ]

    final_state, state_history = process_sequence_polar(initial_vec, seq, verbose=True)
    # 为了向后兼容，提供传统接口
    final_polar_vec, final_scale_vec = final_state.get_polar_vec(), final_state.get_scale_vec()

    print("\nMixed precision polar quantum simulation test completed!")
    print(f"✅ Amplitude: {str(vector.amplitude_dtype).split('.')[-1]} encoding")
    print(f"✅ Phase: {str(vector.phase_dtype).split('.')[-1]} encoding")
    print("✅ Compatible with all quantum gates")


# ===== 兼容性函数 =====

def polar_to_complex_tensor(polar_vec_or_state, scale_vec=None) -> torch.Tensor:
    """
    兼容性函数：将极坐标向量或 PolarStateEncoded 转换为复数向量

    - 若传入 PolarStateEncoded，使用 decode_state() 正确解码（含 amplitude_max/wm_a）
    - 若传入 (polar_vec, scale_vec)，使用 vector.to_complex（WM 模式下无 amplitude_max 时解码可能不准，推荐用 state.decode_state()）
    """
    if isinstance(polar_vec_or_state, PolarStateEncoded):
        return polar_vec_or_state.decode_state()
    return vector.to_complex(polar_vec_or_state, scale_vec)
