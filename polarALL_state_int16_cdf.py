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
from vector_withCDF import PolarVector, PolarStateEncoded

# 设置设备并创建向量实例
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
vector = PolarVector(device=device, amplitude_dtype=torch.int8, phase_dtype=torch.int8)
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

def hadamard_polar_tensor(z0_batch: torch.Tensor, z1_batch: torch.Tensor, params) -> Tuple[torch.Tensor, torch.Tensor, dict]:
    """CDF 版本：Hadamard 门极坐标实现"""
    c0 = vector.polar_to_complex_tensor(z0_batch, params=params)
    c1 = vector.polar_to_complex_tensor(z1_batch, params=params)
    state_vector = torch.stack([c0, c1], dim=1)
    h_mat = H_MAT_TORCH.to(dtype=state_vector.dtype, device=state_vector.device)
    result = torch.matmul(state_vector, h_mat.T)
    c0_result, c1_result = result[:, 0], result[:, 1]
    c_combined = torch.cat([c0_result, c1_result])
    polar_full, params_new = vector.complex_to_polar_tensor(c_combined)
    n = len(c0_result)
    polar0, polar1 = polar_full[:n].clone(), polar_full[n:].clone()
    return polar0, polar1, params_new


def x_polar_tensor(z0_batch: torch.Tensor, z1_batch: torch.Tensor, params) -> Tuple[torch.Tensor, torch.Tensor, dict]:
    """X 门：交换完整的状态向量"""
    return z1_batch, z0_batch, params


def y_polar_tensor(z0_batch: torch.Tensor, z1_batch: torch.Tensor, params) -> Tuple[torch.Tensor, torch.Tensor, dict]:
    """PyTorch 版本：Y 门极坐标实现（int16 + int8 混合版本，批量版本）
    Y 门：|0⟩ -> i|1⟩, |1⟩ -> -i|0⟩
    在极坐标中：交换幅度，调整相位
    """
    result_z0 = z1_batch.clone()
    result_z1 = z0_batch.clone()
    result_z0[:, 1] = vector.add_phase_encoded(z1_batch[:, 1], -math.pi / 2)
    result_z1[:, 1] = vector.add_phase_encoded(z0_batch[:, 1], math.pi / 2)
    return result_z0, result_z1, params


def z_polar_tensor(z0_batch: torch.Tensor, z1_batch: torch.Tensor, params) -> Tuple[torch.Tensor, torch.Tensor, dict]:
    """Z 门极坐标实现"""
    result_z0 = z0_batch.clone()
    result_z1 = z1_batch.clone()
    result_z1[:, 1] = vector.add_phase_encoded(z1_batch[:, 1], math.pi)
    return result_z0, result_z1, params


def s_polar_tensor(z0_batch: torch.Tensor, z1_batch: torch.Tensor, params) -> Tuple[torch.Tensor, torch.Tensor, dict]:
    """S 门极坐标实现"""
    result_z0 = z0_batch.clone()
    result_z1 = z1_batch.clone()
    result_z1[:, 1] = vector.add_phase_encoded(z1_batch[:, 1], math.pi / 2)
    return result_z0, result_z1, params


def t_polar_tensor(z0_batch: torch.Tensor, z1_batch: torch.Tensor, params) -> Tuple[torch.Tensor, torch.Tensor, dict]:
    """T 门极坐标实现"""
    result_z0 = z0_batch.clone()
    result_z1 = z1_batch.clone()
    result_z1[:, 1] = vector.add_phase_encoded(z1_batch[:, 1], math.pi / 4)
    return result_z0, result_z1, params


def rx_polar_tensor(z0_batch: torch.Tensor, z1_batch: torch.Tensor, params, theta: float) -> Tuple[torch.Tensor, torch.Tensor, dict]:
    """Rx 门极坐标实现"""
    c0 = vector.polar_to_complex_tensor(z0_batch, params=params)
    c1 = vector.polar_to_complex_tensor(z1_batch, params=params)
    dev, dtype = c0.device, c0.dtype
    c, s = math.cos(theta / 2), math.sin(theta / 2)
    rx_mat = torch.tensor([[c, -1j * s], [-1j * s, c]], dtype=dtype, device=dev)
    result = torch.matmul(torch.stack([c0, c1], dim=1), rx_mat.T)
    c0_result, c1_result = result[:, 0], result[:, 1]
    c_combined = torch.cat([c0_result, c1_result])
    polar_full, params_new = vector.complex_to_polar_tensor(c_combined)
    n = len(c0_result)
    return polar_full[:n].clone(), polar_full[n:].clone(), params_new


def ry_polar_tensor(z0_batch: torch.Tensor, z1_batch: torch.Tensor, params, theta: float) -> Tuple[torch.Tensor, torch.Tensor, dict]:
    """Ry 门极坐标实现"""
    c0 = vector.polar_to_complex_tensor(z0_batch, params=params)
    c1 = vector.polar_to_complex_tensor(z1_batch, params=params)
    dev, dtype = c0.device, c0.dtype
    c, s = math.cos(theta / 2), math.sin(theta / 2)
    ry_mat = torch.tensor([[c, -s], [s, c]], dtype=dtype, device=dev)
    result = torch.matmul(torch.stack([c0, c1], dim=1), ry_mat.T)
    c0_result, c1_result = result[:, 0], result[:, 1]
    c_combined = torch.cat([c0_result, c1_result])
    polar_full, params_new = vector.complex_to_polar_tensor(c_combined)
    n = len(c0_result)
    return polar_full[:n].clone(), polar_full[n:].clone(), params_new


def rz_polar_tensor(z0_batch: torch.Tensor, z1_batch: torch.Tensor, params, phi: float) -> Tuple[torch.Tensor, torch.Tensor, dict]:
    """Rz(φ) 门：直接相位操作"""
    result_z0 = z0_batch.clone()
    result_z1 = z1_batch.clone()
    result_z0[:, 1] = vector.add_phase_encoded(z0_batch[:, 1], -phi / 2)
    result_z1[:, 1] = vector.add_phase_encoded(z1_batch[:, 1], phi / 2)
    return result_z0, result_z1, params


def u1_polar_tensor(z0_batch: torch.Tensor, z1_batch: torch.Tensor, params, lambda_param: float) -> Tuple[torch.Tensor, torch.Tensor, dict]:
    """U1(λ) 只给 |1⟩ 添加相位 λ"""
    result_z0 = z0_batch.clone()
    result_z1 = z1_batch.clone()
    result_z1[:, 1] = vector.add_phase_encoded(z1_batch[:, 1], lambda_param)
    return result_z0, result_z1, params


def u2_polar_tensor(z0_batch: torch.Tensor, z1_batch: torch.Tensor, params, phi: float, lambda_param: float) -> Tuple[torch.Tensor, torch.Tensor, dict]:
    """U2 门极坐标实现"""
    c0 = vector.polar_to_complex_tensor(z0_batch, params=params)
    c1 = vector.polar_to_complex_tensor(z1_batch, params=params)
    dtype, dev = c0.dtype, c0.device
    inv_sqrt2 = 1 / math.sqrt(2)
    exp_phi = torch.exp(torch.tensor(1j * phi, dtype=dtype, device=dev))
    exp_lambda = torch.exp(torch.tensor(1j * lambda_param, dtype=dtype, device=dev))
    u2_mat = torch.stack([
        torch.stack([torch.tensor(inv_sqrt2, dtype=dtype, device=dev), -inv_sqrt2 * exp_lambda]),
        torch.stack([inv_sqrt2 * exp_phi, inv_sqrt2 * exp_phi * exp_lambda])
    ])
    result = torch.matmul(torch.stack([c0, c1], dim=1), u2_mat.T)
    c0_result, c1_result = result[:, 0], result[:, 1]
    c_combined = torch.cat([c0_result, c1_result])
    polar_full, params_new = vector.complex_to_polar_tensor(c_combined)
    n = len(c0_result)
    return polar_full[:n], polar_full[n:], params_new


def u3_polar_tensor(z0_batch: torch.Tensor, z1_batch: torch.Tensor, params, theta: float, phi: float, lambda_param: float) -> Tuple[torch.Tensor, torch.Tensor, dict]:
    """U3 门极坐标实现"""
    c0 = vector.polar_to_complex_tensor(z0_batch, params=params)
    c1 = vector.polar_to_complex_tensor(z1_batch, params=params)
    dtype, dev = c0.dtype, c0.device
    c_val, s_val = math.cos(theta / 2), math.sin(theta / 2)
    exp_phi = torch.exp(torch.tensor(1j * phi, dtype=dtype, device=dev))
    exp_lambda = torch.exp(torch.tensor(1j * lambda_param, dtype=dtype, device=dev))
    u3_mat = torch.stack([
        torch.stack([torch.tensor(c_val, dtype=dtype, device=dev), -exp_lambda * s_val]),
        torch.stack([exp_phi * s_val, exp_phi * exp_lambda * c_val])
    ])
    result = torch.matmul(torch.stack([c0, c1], dim=1), u3_mat.T)
    c0_result, c1_result = result[:, 0], result[:, 1]
    c_combined = torch.cat([c0_result, c1_result])
    polar_full, params_new = vector.complex_to_polar_tensor(c_combined)
    n = len(c0_result)
    return polar_full[:n].clone(), polar_full[n:].clone(), params_new


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

def apply_polar_gate_tensor(polar_vec: torch.Tensor, params,
                          gate_func, *gate_params, qubit_idx: int = 0) -> Tuple[torch.Tensor, dict]:
    """
    统一的极坐标量子门应用函数（CDF 版本）

    Returns:
        (new_polar_vec, new_params)
    """
    n = polar_vec.shape[0]
    n_qubits = int(math.log2(n))
    assert 2 ** n_qubits == n, f"状态向量长度必须是 2 的幂次，当前长度: {n}"
    assert 0 <= qubit_idx < n_qubits, f"比特位索引必须在 [0, {n_qubits}) 范围内"

    all_indices = torch.arange(n, device=device, dtype=torch.long)
    qubit_mask = 1 << qubit_idx
    idx0_batch = all_indices[(all_indices & qubit_mask) == 0]
    idx1_batch = idx0_batch | qubit_mask

    if len(idx0_batch) > 0:
        states0 = polar_vec[idx0_batch]
        states1 = polar_vec[idx1_batch]
        ret = gate_func(states0, states1, params, *gate_params)
        new_states0, new_states1, new_params = ret[:3]
        polar_vec[idx0_batch] = new_states0
        polar_vec[idx1_batch] = new_states1
        params = new_params

    return polar_vec, params


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

    polar_vec = state.get_polar_vec()
    params = state.get_params()  # CDF: 返回 params

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

        if gate_name == 'CNOT':
            polar_vec, params = apply_cnot_polar_tensor(polar_vec, params, control_idx, target_idx)
        elif gate_name == 'CZ':
            polar_vec, params = apply_cz_polar_tensor(polar_vec, params, control_idx, target_idx)
        elif gate_name == 'CH':
            polar_vec, params = apply_ch_polar_tensor(polar_vec, params, control_idx, target_idx)
        elif gate_name == 'CY':
            polar_vec, params = apply_cy_polar_tensor(polar_vec, params, control_idx, target_idx)
        elif gate_name == 'CS':
            polar_vec, params = apply_cs_polar_tensor(polar_vec, params, control_idx, target_idx)
        elif gate_name == 'CT':
            polar_vec, params = apply_ct_polar_tensor(polar_vec, params, control_idx, target_idx)
        elif gate_name == 'CRx':
            if not gate_params:
                raise ValueError(f"CRx 需要参数 theta")
            polar_vec, params = apply_crx_polar_tensor(polar_vec, params, control_idx, target_idx, gate_params[0])
        elif gate_name == 'CRy':
            if not gate_params:
                raise ValueError(f"CRy 需要参数 theta")
            polar_vec, params = apply_cry_polar_tensor(polar_vec, params, control_idx, target_idx, gate_params[0])
        elif gate_name == 'CRz':
            if not gate_params:
                raise ValueError(f"CRz 需要参数 phi")
            polar_vec, params = apply_crz_polar_tensor(polar_vec, params, control_idx, target_idx, gate_params[0])
        elif gate_name == 'CU1':
            if not gate_params:
                raise ValueError(f"CU1 需要参数 lambda")
            polar_vec, params = apply_cu1_polar_tensor(polar_vec, params, control_idx, target_idx, gate_params[0])
        elif gate_name == 'CU2':
            if len(gate_params) < 2:
                raise ValueError(f"CU2 需要参数 phi, lambda")
            polar_vec, params = apply_cu2_polar_tensor(polar_vec, params, control_idx, target_idx, gate_params[0], gate_params[1])
        elif gate_name == 'CU3':
            if len(gate_params) < 3:
                raise ValueError(f"CU3 需要参数 theta, phi, lambda")
            polar_vec, params = apply_cu3_polar_tensor(polar_vec, params, control_idx, target_idx, gate_params[0], gate_params[1], gate_params[2])
    else:
        polar_vec, params = apply_polar_gate_tensor(
            polar_vec, params, POLAR_GATES_TORCH[gate_name], *gate_params,
            qubit_idx=target_idx or 0)

    state.update_polar_vec(polar_vec, params)
    return state


def apply_controlled_gate_polar_tensor(polar_vec: torch.Tensor, params,
                                     control_idx: int, target_idx: int, gate_func, *gate_params) -> Tuple[torch.Tensor, dict]:
    """
    应用控制门（CDF 版本）- 完全修复局部重量化导致全局参数不一致问题

    根因：gate_func 仅对控制位=1 的子集做矩阵运算并重新量化，生成的 new_params 只反映
    该子集分布。若直接采用 new_params，控制位=0 的状态仍用旧编码，导致同一向量内存在
    两套量化参数，decode 时必有一半完全错误。修复：先将全部状态解码为复数，合并后统一重量化。
    """
    n = polar_vec.shape[0]
    n_qubit = int(math.log2(n))
    device_local = polar_vec.device
    assert 2 ** n_qubit == n and control_idx != target_idx
    assert 0 <= control_idx < n_qubit and 0 <= target_idx < n_qubit

    control_mask = 1 << control_idx
    target_mask = 1 << target_idx
    all_indices = torch.arange(n, device=device_local, dtype=torch.long)

    control_is_0 = (all_indices & control_mask) == 0
    control_is_1 = ~control_is_0

    idx_control_1_target_0 = all_indices[control_is_1 & ((all_indices & target_mask) == 0)]
    idx_control_1_target_1 = idx_control_1_target_0 | target_mask

    if len(idx_control_1_target_0) == 0:
        return polar_vec, params

    idx_control_0 = all_indices[control_is_0]

    if len(idx_control_0) > 0:
        states_control_0 = polar_vec[idx_control_0].clone()
        complex_control_0 = vector.polar_to_complex_tensor(states_control_0, params=params)

    states0 = polar_vec[idx_control_1_target_0].clone()
    states1 = polar_vec[idx_control_1_target_1].clone()
    ret = gate_func(states0, states1, params, *gate_params)
    new_states0, new_states1, new_params = ret[:3]

    complex_control_1_target_0 = vector.polar_to_complex_tensor(new_states0, params=new_params)
    complex_control_1_target_1 = vector.polar_to_complex_tensor(new_states1, params=new_params)

    complex_full = torch.empty(n, dtype=torch.complex64, device=device_local)
    if len(idx_control_0) > 0:
        complex_full[idx_control_0] = complex_control_0
    complex_full[idx_control_1_target_0] = complex_control_1_target_0
    complex_full[idx_control_1_target_1] = complex_control_1_target_1

    polar_vec_new, params_new = vector.complex_to_polar_tensor(complex_full)
    return polar_vec_new, params_new


def apply_cnot_polar_tensor(polar_vec: torch.Tensor, params, control_idx: int, target_idx: int) -> Tuple[torch.Tensor, dict]:
    """应用 CNOT 门：当控制比特为 |1⟩ 时，交换 |...0⟩ 与 |...1⟩（向量化版本）"""
    n = polar_vec.shape[0]
    n_qubit = int(math.log2(n))
    assert 2 ** n_qubit == n and control_idx != target_idx
    control_mask = 1 << control_idx
    target_mask = 1 << target_idx

    all_indices = torch.arange(n, device=device, dtype=torch.long)
    control_is_1 = (all_indices & control_mask) != 0
    target_is_0 = (all_indices & target_mask) == 0

    idx0 = all_indices[control_is_1 & target_is_0]  # 控制=1, 目标=0
    idx1 = idx0 ^ target_mask  # 控制=1, 目标=1

    if len(idx0) > 0:
        temp = polar_vec[idx0].clone()
        polar_vec[idx0] = polar_vec[idx1]
        polar_vec[idx1] = temp

    return polar_vec, params


def apply_ch_polar_tensor(polar_vec: torch.Tensor, params, control_idx: int, target_idx: int):
    """应用 CH 门"""
    return apply_controlled_gate_polar_tensor(polar_vec, params, control_idx, target_idx, hadamard_polar_tensor)


def apply_cs_polar_tensor(polar_vec: torch.Tensor, params, control_idx: int, target_idx: int):
    """应用 CS 门"""
    return apply_controlled_gate_polar_tensor(polar_vec, params, control_idx, target_idx, s_polar_tensor)


def apply_ct_polar_tensor(polar_vec: torch.Tensor, params, control_idx: int, target_idx: int):
    """应用 CT 门"""
    return apply_controlled_gate_polar_tensor(polar_vec, params, control_idx, target_idx, t_polar_tensor)


def apply_cy_polar_tensor(polar_vec: torch.Tensor, params, control_idx: int, target_idx: int):
    """应用 CY 门"""
    return apply_controlled_gate_polar_tensor(polar_vec, params, control_idx, target_idx, y_polar_tensor)


def apply_crx_polar_tensor(polar_vec: torch.Tensor, params, control_idx: int, target_idx: int, theta: float):
    """应用 CRx 门（与 polarALL_state_3 一致）"""
    return apply_controlled_gate_polar_tensor(polar_vec, params, control_idx, target_idx, rx_polar_tensor, theta)


def apply_cry_polar_tensor(polar_vec: torch.Tensor, params, control_idx: int, target_idx: int, theta: float):
    """应用 CRy 门"""
    return apply_controlled_gate_polar_tensor(polar_vec, params, control_idx, target_idx, ry_polar_tensor, theta)


def apply_crz_polar_tensor(polar_vec: torch.Tensor, params, control_idx: int, target_idx: int, phi: float):
    """应用 CRz 门"""
    return apply_controlled_gate_polar_tensor(polar_vec, params, control_idx, target_idx, rz_polar_tensor, phi)


def apply_cu1_polar_tensor(polar_vec: torch.Tensor, params, control_idx: int, target_idx: int, lambda_param: float):
    """应用 CU1 门"""
    return apply_controlled_gate_polar_tensor(polar_vec, params, control_idx, target_idx, u1_polar_tensor, lambda_param)


def apply_cu2_polar_tensor(polar_vec: torch.Tensor, params, control_idx: int, target_idx: int, phi: float, lambda_param: float):
    """应用 CU2 门"""
    return apply_controlled_gate_polar_tensor(polar_vec, params, control_idx, target_idx, u2_polar_tensor, phi, lambda_param)


def apply_cu3_polar_tensor(polar_vec: torch.Tensor, params, control_idx: int, target_idx: int, theta: float, phi: float, lambda_param: float):
    """应用 CU3 门"""
    return apply_controlled_gate_polar_tensor(polar_vec, params, control_idx, target_idx, u3_polar_tensor, theta, phi, lambda_param)


def apply_cz_polar_tensor(polar_vec: torch.Tensor, params, control_idx: int, target_idx: int) -> Tuple[torch.Tensor, dict]:
    """
    应用 CZ 门：当控制比特与目标比特都为 |1⟩ 时，相位翻转 π（向量化版本）
    """
    n = polar_vec.shape[0]
    n_qubit = int(math.log2(n))
    assert 2 ** n_qubit == n and control_idx != target_idx
    assert 0 <= control_idx < n_qubit and 0 <= target_idx < n_qubit

    control_mask = 1 << control_idx
    target_mask = 1 << target_idx

    all_indices = torch.arange(n, device=device, dtype=torch.long)
    both_1 = ((all_indices & control_mask) != 0) & ((all_indices & target_mask) != 0)
    idx_both_1 = all_indices[both_1]

    if len(idx_both_1) > 0:
        new_phase = vector.add_phase_encoded(polar_vec[idx_both_1, 1], math.pi)
        polar_vec[idx_both_1, 1] = new_phase.to(polar_vec.dtype)

    return polar_vec, params


# ===== 辅助函数 =====

def create_random_polar_state(n_amps: int) -> PolarStateEncoded:
    """创建随机极坐标 CDF 状态向量"""
    polar_vec, params = vector.create_random_state(n_amps)
    return PolarStateEncoded(polar_vec, params, vector)


def polar_vec_to_string(polar_vec: torch.Tensor, scale_vec=None, params=None, indices=None) -> str:
    """将极坐标向量转换为字符串表示"""
    return vector.to_string(polar_vec, scale_vec=scale_vec, params=params, indices=indices)


# ===== 核心处理函数 =====

def process_sequence_polar(initial_vec: List[complex], seq: List[Tuple], verbose: bool = False) -> Tuple[PolarStateEncoded, List[PolarStateEncoded]]:
    """
    处理量子门序列（CDF 版本），记录每一步的状态变化

    Returns:
        (final_state, state_history): 最终 PolarStateEncoded 和每一步的状态历史
    """
    n_amps = len(initial_vec)
    n_qubits = int(math.log2(n_amps))
    assert 2 ** n_qubits == n_amps, f"状态向量长度必须是 2 的幂，当前: {n_amps}"

    complex_tensor = torch.tensor(initial_vec, dtype=torch.complex64, device=device)
    initial_state = vector.complex_to_polar_state(complex_tensor)
    state_history = [initial_state.clone()]

    if verbose:
        print(f"处理 {len(seq)} 个量子门，系统有 {n_qubits} 个比特")
        print(f"初始状态（前8个）:")
        print(polar_vec_to_string(initial_state.get_polar_vec(), params=initial_state.params, indices=range(min(8, n_amps))))

    current_state = initial_state
    for step, gate_tuple in enumerate(seq):
        if len(gate_tuple) != 5:
            raise ValueError(f"序列必须是 5 元组格式: (name, param_str, params, control_idx, target_idx)，当前: {gate_tuple}")

        gate_name, _, gate_params, control_idx, target_idx = gate_tuple
        if not isinstance(gate_params, list):
            gate_params = []

        is_controlled = control_idx is not None

        if is_controlled:
            current_state = apply_gate_polar_tensor(current_state, gate_name, gate_params, control_idx, target_idx)
            gate_info = f"{gate_name}(控制={control_idx}, 目标={target_idx})"
        else:
            current_state = apply_gate_polar_tensor(current_state, gate_name, gate_params, target_idx=target_idx)
            gate_info = f"{gate_name}(比特={target_idx})"

        state_history.append(current_state.clone())

        if verbose:
            print(f"门 {step + 1}: {gate_info}")
            if step < 2 or step % 10 == 0:
                print(f"  状态（前8个）: {polar_vec_to_string(current_state.get_polar_vec(), params=current_state.params, indices=range(min(8, n_amps)))}")

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

        # 创建随机序列（统一 5 元组格式）
        seq = []
        for _ in range(sequence_length):
            gate_name = random.choice(['H', 'X', 'Y', 'Z', 'S', 'T'])
            if gate_name in ['Rx', 'Ry', 'Rz']:
                gate_params = [random.uniform(0, 2*math.pi)]
            else:
                gate_params = []
            target_idx = random.randint(0, n_qubits-1)
            seq.append((gate_name, '', gate_params, None, target_idx))

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

    # 测试量子门
    print("\nTesting quantum gates...")
    test_state = apply_gate_polar_tensor(test_state, 'H', target_idx=0)

    test_state = apply_gate_polar_tensor(test_state, 'X', target_idx=1)

    # 序列处理测试
    print("\nTesting sequence processing...")
    initial_vec = [1.0, 0.0, 0.0, 0.0]  # |00⟩ 状态
    seq = [
        ('H', '', [], None, 0),       # H 门，目标比特 0
        ('X', '', [], None, 1),       # X 门，目标比特 1
        ('CNOT', '', [], 0, 1),       # CNOT，控制=0，目标=1
    ]

    final_state, state_history = process_sequence_polar(initial_vec, seq, verbose=True)
    # 为了向后兼容，提供传统接口
    final_polar_vec, final_scale_vec = final_state.get_polar_vec(), final_state.get_params()

    print("\nMixed precision polar quantum simulation test completed!")
    print(f"✅ Amplitude: {str(vector.amplitude_dtype).split('.')[-1]} encoding")
    print(f"✅ Phase: {str(vector.phase_dtype).split('.')[-1]} encoding")
    print("✅ Compatible with all quantum gates")


