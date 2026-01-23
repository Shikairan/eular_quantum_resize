"""
量子计算工具函数
包含随机序列生成和初始向量创建的通用函数
"""

import random
import math
import numpy as np
from typing import List, Tuple

def random_sequence(n: int, m: int, amp_n: int, n_qubits: int = 5, shuffle: bool = False) -> List[Tuple]:
    """
    生成随机量子门序列（通用版本）

    Args:
        n: 无参数单比特门数量
        m: 有参数门数量（单比特和控制门）
        amp_n: 幅度相关门数量（H 和旋转门）
        n_qubits: 量子比特数量
        shuffle: 是否随机打乱顺序

    Returns:
        量子门序列，格式为 [(gate_name, param_str, params, qubit_info...), ...]
    """
    seq = []

    # 单参数门
    single_param_gates = ['Rx', 'Ry', 'Rz', 'U1']
    # 多参数门
    multi_param_gates = ['U2', 'U3']
    # 无参数单比特门
    no_param_gates = ['X', 'Y', 'S', 'H', 'T', 'Z']
    #no_param_gates = ['X', 'Y', 'S', 'T', 'Z']
    # 无参数控制门
    no_param_controlled_gates = ['CX', 'CY', 'CH', 'CZ',  'CS', 'CT']
    #no_param_controlled_gates = ['CX', 'CY', 'CZ',  'CS', 'CT']
    # 单参数控制门
    single_param_controlled_gates = ['CRx', 'CRy', 'CRz', 'CU1']
    # 多参数控制门
    multi_param_controlled_gates = ['CU2', 'CU3']

    # 生成无参数单比特门
    for _ in range(n):
        g = random.choice(no_param_gates)
        qubit_idx = random.randint(0, n_qubits - 1)
        seq.append((g, '', [], qubit_idx))

    # 生成无参数控制门
    for _ in range(n):
        g = random.choice(no_param_controlled_gates)
        qubits = random.sample(range(n_qubits), 2)
        control_idx, target_idx = qubits[0], qubits[1]
        seq.append((g, '', [], control_idx, target_idx))
    # 生成有参数单比特门（单参数和多参数）
    for _ in range(m):
        g = random.choice(single_param_gates + multi_param_gates)
        qubit_idx = random.randint(0, n_qubits - 1)
        if g in single_param_gates:
            param = random.uniform(0, 2 * math.pi)
            seq.append((g, f'({param:.3f})', [param], qubit_idx))
        elif g == 'U2':
            phi = random.uniform(0, 2 * math.pi)
            lambda_param = random.uniform(0, 2 * math.pi)
            seq.append((g, f'({phi:.3f},{lambda_param:.3f})', [phi, lambda_param], qubit_idx))
        elif g == 'U3':
            theta = random.uniform(0, 2 * math.pi)
            phi = random.uniform(0, 2 * math.pi)
            lambda_param = random.uniform(0, 2 * math.pi)
            seq.append((g, f'({theta:.3f},{phi:.3f},{lambda_param:.3f})', [theta, phi, lambda_param], qubit_idx))

    # 生成有参数控制门
    for _ in range(m):
        g = random.choice(single_param_controlled_gates + multi_param_controlled_gates)
        qubits = random.sample(range(n_qubits), 2)
        control_idx, target_idx = qubits[0], qubits[1]
        if g in single_param_controlled_gates:
            param = random.uniform(0, 2 * math.pi)
            seq.append((g, f'({param:.3f})', [param], control_idx, target_idx))
        elif g == 'CU2':
            phi = random.uniform(0, 2 * math.pi)
            lambda_param = random.uniform(0, 2 * math.pi)
            seq.append((g, f'({phi:.3f},{lambda_param:.3f})', [phi, lambda_param], control_idx, target_idx))
        elif g == 'CU3':
            theta = random.uniform(0, 2 * math.pi)
            phi = random.uniform(0, 2 * math.pi)
            lambda_param = random.uniform(0, 2 * math.pi)
            seq.append((g, f'({theta:.3f},{phi:.3f},{lambda_param:.3f})', [theta, phi, lambda_param], control_idx, target_idx))

    # 生成幅度相关的门（H 门和旋转门）
    for _ in range(amp_n):
        g = random.choice(['H', 'Rx', 'Ry', 'Rz'])
        qubit_idx = random.randint(0, n_qubits - 1)
        if g in ['Rx', 'Ry', 'Rz']:
            param = random.uniform(0, 2 * math.pi)
            seq.append((g, f'({param:.3f})', [param], qubit_idx))
        else:
            seq.append((g, '', [], qubit_idx))

    if shuffle:
        random.shuffle(seq)

    return seq

def random_sequence_single_qubit_only(n: int, m: int, amp_n: int, n_qubits: int = 5, shuffle: bool = False) -> List[Tuple]:
    """
    生成只包含单比特门的随机量子门序列（避免控制门问题）

    Args:
        n: 无参数单比特门数量
        m: 有参数单比特门数量
        amp_n: 幅度相关门数量（H 和旋转门）
        n_qubits: 量子比特数量
        shuffle: 是否随机打乱顺序

    Returns:
        只包含单比特门的量子门序列
    """
    seq = []

    # 无参数单比特门
    no_param_gates = ['X', 'Y', 'S', 'H', 'T', 'Z']
    # 单参数门
    single_param_gates = ['Rx', 'Ry', 'Rz']

    # 生成无参数单比特门
    for _ in range(n):
        g = random.choice(no_param_gates)
        qubit_idx = random.randint(0, n_qubits - 1)
        seq.append((g, '', [], qubit_idx))

    # 生成有参数单比特门
    for _ in range(m):
        g = random.choice(single_param_gates)
        qubit_idx = random.randint(0, n_qubits - 1)
        param = random.uniform(0, 2 * math.pi)
        seq.append((g, f'({param:.3f})', [param], qubit_idx))

    # 生成幅度相关的门
    for _ in range(amp_n):
        g = random.choice(['H', 'Rx', 'Ry', 'Rz'])
        qubit_idx = random.randint(0, n_qubits - 1)
        if g in ['Rx', 'Ry', 'Rz']:
            param = random.uniform(0, 2 * math.pi)
            seq.append((g, f'({param:.3f})', [param], qubit_idx))
        else:
            seq.append((g, '', [], qubit_idx))

    if shuffle:
        random.shuffle(seq)

    return seq

def create_initial_vec_complex(n_amps: int, avg=True) -> List[complex]:
    """
    创建归一化的复数初始向量（与 ALLeularV2.py 第903行格式相同）

    Args:
        n_amps: 向量长度（必须是2的幂次）

    Returns:
        归一化的复数向量
    """
    assert n_amps & (n_amps - 1) == 0, f"向量长度 {n_amps} 必须是2的幂次"

    # 创建随机复数向量
    if not avg:
        complex_vector = [complex(random.random(), random.random()) for _ in range(n_amps)]
    else:
        complex_vector = [complex(1,0) for _ in range(n_amps)]
    # 归一化
    norm = np.linalg.norm(complex_vector)
    normalized_vector = complex_vector / norm

    return normalized_vector

def compute_normalization_error(vec: List[complex]) -> float:
    """
    计算向量的归一化误差

    Args:
        vec: 复数向量

    Returns:
        归一化误差 |norm - 1.0|
    """
    norm = math.sqrt(sum(abs(z)**2 for z in vec))
    return abs(norm - 1.0)

# ===== 兼容性函数 =====

# 为保持向后兼容性，提供别名
random_sequence_tensor = random_sequence
create_initial_vec_like_v2 = create_initial_vec_complex
