"""
纯复数版本的量子计算实现
用于对照测试与 ALLeular 版本的计算结果

✨ 主要功能：
1. 纯复数状态向量操作（不使用 int8 编码）
2. 支持完整的量子门集合
3. 统一的门应用函数，支持序列处理
4. 与 ALLeular 版本完全兼容的接口

⚠️ 注意：
- 本文件仅处理复数计算，不涉及 int8 编码
- 用于验证和对照 ALLeular 版本的正确性
"""

import numpy as np
import math
import cmath
import random
from typing import List, Tuple
from utils import random_sequence, create_initial_vec_complex

# ===== 量子门矩阵定义 =====

H_MAT = np.array([[1, 1], [1, -1]], dtype=complex) / math.sqrt(2)
S_MAT = np.array([[1, 0], [0, 1j]], dtype=complex)
T_MAT = np.array([[1, 0], [0, cmath.exp(1j * math.pi / 4)]], dtype=complex)
X_MAT = np.array([[0, 1], [1, 0]], dtype=complex)
Y_MAT = np.array([[0, -1j], [1j, 0]], dtype=complex)
Z_MAT = np.array([[1, 0], [0, -1]], dtype=complex)

# ===== 量子门函数 =====

def hadamard_complex_pair(z0: complex, z1: complex) -> Tuple[complex, complex]:
    return H_MAT @ (z0, z1)

def s_complex_pair(z0: complex, z1: complex) -> Tuple[complex, complex]:
    return S_MAT @ (z0, z1)

def t_complex_pair(z0: complex, z1: complex) -> Tuple[complex, complex]:
    return T_MAT @ (z0, z1)

def x_complex_pair(z0: complex, z1: complex) -> Tuple[complex, complex]:
    return X_MAT @ (z0, z1)

def y_complex_pair(z0: complex, z1: complex) -> Tuple[complex, complex]:
    return Y_MAT @ (z0, z1)

def z_complex_pair(z0: complex, z1: complex) -> Tuple[complex, complex]:
    return Z_MAT @ (z0, z1)

def rx_complex_pair(z0: complex, z1: complex, theta: float) -> Tuple[complex, complex]:
    c, s = math.cos(theta / 2), math.sin(theta / 2)
    m = np.array([[c, -1j * s], [-1j * s, c]], dtype=complex)
    return m @ (z0, z1)

def ry_complex_pair(z0: complex, z1: complex, theta: float) -> Tuple[complex, complex]:
    """Ry 门：绕 Y 轴旋转"""
    c, s = math.cos(theta / 2), math.sin(theta / 2)
    m = np.array([[c, -s], [s, c]], dtype=complex)
    return m @ (z0, z1)

def rz_complex_pair(z0: complex, z1: complex, phi: float) -> Tuple[complex, complex]:
    """Rz 门：绕 Z 轴旋转"""
    exp_neg = cmath.exp(-1j * phi / 2)
    exp_pos = cmath.exp(1j * phi / 2)
    m = np.array([[exp_neg, 0], [0, exp_pos]], dtype=complex)
    return m @ (z0, z1)

def u1_complex_pair(z0: complex, z1: complex, lambda_param: float) -> Tuple[complex, complex]:
    """U1 门：单参数相位门，等价于 Rz(lambda)"""
    exp_phase = cmath.exp(1j * lambda_param)
    m = np.array([[1, 0], [0, exp_phase]], dtype=complex)
    return m @ (z0, z1)

def u2_complex_pair(z0: complex, z1: complex, phi: float, lambda_param: float) -> Tuple[complex, complex]:
    """U2 门：两参数门"""
    inv_sqrt2 = 1 / math.sqrt(2)
    exp_phi = cmath.exp(1j * phi)
    exp_lambda = cmath.exp(1j * lambda_param)
    m = np.array([[inv_sqrt2, -inv_sqrt2 * exp_lambda],
                  [inv_sqrt2 * exp_phi, inv_sqrt2 * exp_phi * exp_lambda]], dtype=complex)
    return m @ (z0, z1)

def u3_complex_pair(z0: complex, z1: complex, theta: float, phi: float, lambda_param: float) -> Tuple[complex, complex]:
    """U3 门：三参数通用单比特门"""
    c, s = math.cos(theta / 2), math.sin(theta / 2)
    exp_phi = cmath.exp(1j * phi)
    exp_lambda = cmath.exp(1j * lambda_param)
    m = np.array([[c, -exp_lambda * s],
                  [exp_phi * s, exp_phi * exp_lambda * c]], dtype=complex)
    return m @ (z0, z1)

# ===== 控制门处理 =====

def cnot_complex_pair(z0: complex, z1: complex) -> Tuple[complex, complex]:
    return z1, z0

def cz_complex_pair(z0: complex, z1: complex) -> Tuple[complex, complex]:
    return z0, -z1

def apply_cy_complex(vec: List[complex], control_idx: int, target_idx: int):
    """应用 CY 门（复数版本）"""
    apply_controlled_gate_complex(vec, control_idx, target_idx, y_complex_pair)

def apply_ch_complex(vec: List[complex], control_idx: int, target_idx: int):
    """应用 CH 门（复数版本）"""
    apply_controlled_gate_complex(vec, control_idx, target_idx, hadamard_complex_pair)

def apply_cs_complex(vec: List[complex], control_idx: int, target_idx: int):
    """应用 CS 门（复数版本）"""
    apply_controlled_gate_complex(vec, control_idx, target_idx, s_complex_pair)

def apply_ct_complex(vec: List[complex], control_idx: int, target_idx: int):
    """应用 CT 门（复数版本）"""
    apply_controlled_gate_complex(vec, control_idx, target_idx, t_complex_pair)

def apply_crx_complex(vec: List[complex], control_idx: int, target_idx: int, theta: float):
    """应用 CRx 门（复数版本）"""
    apply_controlled_gate_complex(vec, control_idx, target_idx, rx_complex_pair, theta)

def apply_cry_complex(vec: List[complex], control_idx: int, target_idx: int, theta: float):
    """应用 CRy 门（复数版本）"""
    apply_controlled_gate_complex(vec, control_idx, target_idx, ry_complex_pair, theta)

def apply_crz_complex(vec: List[complex], control_idx: int, target_idx: int, phi: float):
    """应用 CRz 门（复数版本）"""
    apply_controlled_gate_complex(vec, control_idx, target_idx, rz_complex_pair, phi)

def apply_cu1_complex(vec: List[complex], control_idx: int, target_idx: int, lambda_param: float):
    """应用 CU1 门（复数版本）"""
    apply_controlled_gate_complex(vec, control_idx, target_idx, u1_complex_pair, lambda_param)

def apply_cu2_complex(vec: List[complex], control_idx: int, target_idx: int, phi: float, lambda_param: float):
    """应用 CU2 门（复数版本）"""
    apply_controlled_gate_complex(vec, control_idx, target_idx, u2_complex_pair, phi, lambda_param)

def apply_cu3_complex(vec: List[complex], control_idx: int, target_idx: int, theta: float, phi: float, lambda_param: float):
    """应用 CU3 门（复数版本）"""
    apply_controlled_gate_complex(vec, control_idx, target_idx, u3_complex_pair, theta, phi, lambda_param)

def apply_cnot_complex(vec: List[complex], control_idx: int, target_idx: int):
    """
    应用 CNOT 门（复数版本）

    Args:
        vec: 状态向量
        control_idx: 控制比特索引
        target_idx: 目标比特索引
    """
    n = len(vec)
    nqubit = int(math.log2(n))
    assert 2 ** nqubit == n
    assert control_idx != target_idx
    assert 0 <= control_idx < nqubit and 0 <= target_idx < nqubit

    # 对于 CNOT：当控制比特为 |1⟩ 时，翻转目标比特
    control_mask = 1 << control_idx
    target_mask = 1 << target_idx

    for i in range(n):
        if (i & control_mask) != 0:  # 控制比特为 1
            j = i ^ target_mask  # 翻转目标比特
            if i < j:  # 避免重复处理
                vec[i], vec[j] = vec[j], vec[i]

def apply_cz_complex(vec: List[complex], qubit1_idx: int, qubit2_idx: int):
    """
    应用 CZ 门（复数版本）

    Args:
        vec: 状态向量
        qubit1_idx: 第一个比特索引
        qubit2_idx: 第二个比特索引
    """
    n = len(vec)
    nqubit = int(math.log2(n))
    assert 2 ** nqubit == n
    assert qubit1_idx != qubit2_idx
    assert 0 <= qubit1_idx < nqubit and 0 <= qubit2_idx < nqubit

    # CZ 门：当两个比特都为 |1⟩ 时，相位翻转 π
    mask1 = 1 << qubit1_idx
    mask2 = 1 << qubit2_idx

    for i in range(n):
        if (i & mask1) != 0 and (i & mask2) != 0:  # 两个比特都为 1
            vec[i] = -vec[i]

def apply_controlled_gate_complex(vec: List[complex], control_idx: int, target_idx: int, gate_func, *gate_params):
    """
    应用控制门（复数版本）
    优化为向量化操作

    Args:
        vec: 状态向量
        control_idx: 控制比特索引
        target_idx: 目标比特索引
        gate_func: 要应用的单比特门函数
        *gate_params: 门函数需要的参数
    """
    n = len(vec)
    nqubit = int(math.log2(n))
    assert 2 ** nqubit == n
    assert control_idx != target_idx
    assert 0 <= control_idx < nqubit and 0 <= target_idx < nqubit

    control_mask = 1 << control_idx
    target_mask = 1 << target_idx

    # 使用 NumPy 向量化操作找到所有控制比特为 1 且目标比特为 0 的状态
    all_indices = np.arange(n)
    control_set = (all_indices & control_mask) != 0  # 控制比特为 1
    target_clear = (all_indices & target_mask) == 0  # 目标比特为 0

    # 找到满足条件的状态索引
    idx0_array = all_indices[control_set & target_clear]

    # 计算对应的目标比特为 1 的状态索引
    idx1_array = idx0_array | target_mask

    # 验证这些状态对的有效性
    valid_mask = idx1_array < n
    idx0_array = idx0_array[valid_mask]
    idx1_array = idx1_array[valid_mask]

    # 批量处理状态对
    if len(idx0_array) > 0:
        # 批量提取状态
        states0 = np.array([vec[i] for i in idx0_array])
        states1 = np.array([vec[i] for i in idx1_array])

        # 批量应用门函数
        if gate_params:
            new_states0, new_states1 = gate_func(states0, states1, *gate_params)
        else:
            new_states0, new_states1 = gate_func(states0, states1)

        # 将结果写回原向量
        for i, (idx0, idx1) in enumerate(zip(idx0_array, idx1_array)):
            vec[idx0] = new_states0[i]
            vec[idx1] = new_states1[i]

# ===== 核心门应用函数 =====

def apply_gate_pair(vec: List[complex], gate_func, *args, qubit_idx=0):
    """
    对指定的比特位应用量子门（复数版本）
    使用 NumPy 向量化操作优化性能

    Args:
        vec: 状态向量，长度为 2^n
        gate_func: 门函数
        *args: 门函数需要的参数（如 theta, phi, lambda 等）
        qubit_idx: 要操作的比特位索引（0 表示最低位）
    """
    n = len(vec)
    nqubit = int(math.log2(n))
    assert 2 ** nqubit == n, f"状态向量长度必须是 2 的幂次，当前长度: {n}"
    assert 0 <= qubit_idx < nqubit, f"比特位索引必须在 [0, {n_qubit}) 范围内"

    # 计算步长：对于第 qubit_idx 个比特，状态对之间的间隔是 2^qubit_idx
    step = 1 << qubit_idx
    # 每个块的大小是 2^(qubit_idx+1)
    block_size = step << 1

    # 使用 NumPy 向量化操作收集所有状态对索引
    idx0_list = []
    idx1_list = []

    # 遍历所有块，收集需要处理的状态对
    for base in range(0, n, block_size):
        for offset in range(step):
            idx0 = base + offset
            idx1 = base + offset + step
            if idx1 < n:  # 确保索引有效
                idx0_list.append(idx0)
                idx1_list.append(idx1)

    # 转换为 NumPy 数组以便向量化操作
    if len(idx0_list) > 0:
        idx0_array = np.array(idx0_list)
        idx1_array = np.array(idx1_list)

        # 批量提取状态对
        states0 = np.array([vec[i] for i in idx0_array])
        states1 = np.array([vec[i] for i in idx1_array])

        # 批量应用门函数
        if args:
            new_states0, new_states1 = gate_func(states0, states1, *args)
        else:
            new_states0, new_states1 = gate_func(states0, states1)

        # 将结果写回原向量
        for i, (idx0, idx1) in enumerate(zip(idx0_array, idx1_array)):
            vec[idx0] = new_states0[i]
            vec[idx1] = new_states1[i]

# ===== 门定义字典 =====

CMP_GATES = {
    'X':  x_complex_pair,
    'Y':  y_complex_pair,
    'S':  s_complex_pair,
    'T':  t_complex_pair,
    'Z':  z_complex_pair,
    'CNOT': cnot_complex_pair,
    'CZ': cz_complex_pair,
    'Rx': rx_complex_pair,
    'Ry': ry_complex_pair,
    'Rz': rz_complex_pair,
    'U1': u1_complex_pair,
    'U2': u2_complex_pair,
    'U3': u3_complex_pair,
    'H': hadamard_complex_pair
}

# ===== 序列生成功能 =====

# random_sequence 函数现在从 utils.py 导入

# ===== 统一的门应用函数 =====

def apply_gate_unified_complex(vec: List[complex], gate_name: str, *gate_params, qubit_idx=0, control_idx=None, target_idx=None):
    """
    统一的量子门应用函数（纯复数版本）

    Args:
        vec: 复数状态向量
        gate_name: 门名称
        *gate_params: 门函数需要的参数
        qubit_idx: 单比特门的比特索引
        control_idx: 控制比特索引（用于控制门）
        target_idx: 目标比特索引（用于控制门）
    """
    # 控制门（需要 control_idx 和 target_idx）
    controlled_gates = {
        'CX': apply_cnot_complex,
        'CY': apply_cy_complex,
        'CZ': apply_cz_complex,
        'CH': apply_ch_complex,
        'CS': apply_cs_complex,
        'CT': apply_ct_complex,
        'CRx': apply_crx_complex,
        'CRy': apply_cry_complex,
        'CRz': apply_crz_complex,
        'CU1': apply_cu1_complex,
        'CU2': apply_cu2_complex,
        'CU3': apply_cu3_complex,
    }

    if gate_name in controlled_gates:
        # 控制门
        if control_idx is None:
            control_idx = 0
        if target_idx is None:
            target_idx = 1

        func_complex = controlled_gates[gate_name]

        if gate_name in ['CX', 'CZ']:
            # CNOT 和 CZ 使用特殊函数
            func_complex(vec, control_idx, target_idx)
        else:
            # 其他控制门
            if gate_params:
                func_complex(vec, control_idx, target_idx, *gate_params)
            else:
                func_complex(vec, control_idx, target_idx)

    elif gate_name in ['CNOT', 'CZ']:
        # 兼容旧名称
        if control_idx is None:
            control_idx = 0
        if target_idx is None:
            target_idx = 1
        if gate_name == 'CNOT':
            apply_cnot_complex(vec, control_idx, target_idx)
        elif gate_name == 'CZ':
            apply_cz_complex(vec, control_idx, target_idx)

    else:
        # 单比特门
        apply_gate_pair(vec, CMP_GATES[gate_name], *gate_params, qubit_idx=qubit_idx)

# ===== 辅助函数 =====

# create_initial_vec_like_v2 函数现在从 utils.py 导入为 create_initial_vec_complex 

def compute_normalization_error(vec: List[complex]) -> float:
    """计算归一化误差"""
    norm = math.sqrt(sum(abs(z)**2 for z in vec))
    return abs(norm - 1.0)

def print_state_vector(vec: List[complex], indices=None, title="状态向量") -> None:
    """打印状态向量"""
    if indices is None:
        indices = range(len(vec))

    print(f"\n{title}:")
    n_qubits = int(math.log2(len(vec)))
    for i in indices:
        print(f" {i:0{n_qubits}b}: {vec[i]:>18.5f}")

# ===== 序列处理主函数 =====

def process_sequence_complex(seq: List[Tuple], initial_vec: List[complex], verbose: bool = False) -> Tuple[List[complex], List[List[complex]]]:
    """
    处理量子门序列（纯复数版本），记录每一步的状态变化

    Args:
        seq: 量子门序列，格式为 [(gate_name, param_str, params, qubit_info...), ...]
            对于单比特门: (gate_name, param_str, params, qubit_idx)
            对于控制门: (gate_name, param_str, params, control_idx, target_idx)
        initial_vec: 初始状态向量（与 ALLeularV2.py 第903行格式相同）
        verbose: 是否打印详细信息

    Returns:
        (final_vec, state_history): 最终状态向量和每一步的状态历史
    """
    n_amps = len(initial_vec)
    n_qubits = int(math.log2(n_amps))
    assert 2 ** n_qubits == n_amps, f"状态向量长度必须是 2 的幂，当前: {n_amps}"

    # 使用输入的初始状态向量
    vec = initial_vec.copy()  # 复制以避免修改原向量

    # 记录状态历史
    state_history = [vec.copy()]  # 初始状态

    if verbose:
        print(f"处理 {len(seq)} 个量子门，系统有 {n_qubits} 个比特")
        print(f"初始状态（前8个）:")
        print_state_vector(vec, range(min(8, n_amps)))

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
            apply_gate_unified_complex(vec, name, *gate_params, control_idx=control_idx, target_idx=target_idx)
            gate_info = f"{name}(控制={control_idx}, 目标={target_idx})"
        else:
            # 单比特门：使用序列中指定的比特位
            apply_gate_unified_complex(vec, name, *gate_params, qubit_idx=qubit_idx)
            gate_info = f"{name}(比特={qubit_idx})"

        # 记录每一步的状态
        state_history.append(vec.copy())

        if verbose and (step % 50 == 49 or step == len(seq) - 1):
            param_display = param_str if param_str else ''
            print(f"\nAfter {step+1} gates - Last: {gate_info}{param_display}:")
            norm_error = compute_normalization_error(vec)
            print(f"归一化误差: {norm_error:.2e}")
            print("当前状态（前8个）:")
            print_state_vector(vec, range(min(8, n_amps)))

    if verbose:
        final_norm_error = compute_normalization_error(vec)
        print(f"\n最终归一化误差: {final_norm_error:.2e}")

    return vec, state_history

# ===== 使用示例 =====

if __name__ == "__main__":
    print("complexALL.py 使用示例")
    print("=" * 40)

    # 示例 1: 按照代码块 (1-10) 的使用方式
    print("示例 1: 基本使用方式")

    # 生成随机序列（与 ALLeularV2.py 相同的方式）
    seq = random_sequence(50, 50, 50, shuffle=True)

    from complexALL import process_sequence_complex, create_initial_vec_like_v2

    # 创建初始向量（与 ALLeularV2.py 第903行相同）
    initial_vec = create_initial_vec_like_v2(32)

    # 处理序列，获取最终状态和历史记录
    final_vec, state_history = process_sequence_complex(seq, initial_vec, verbose=True)

    print(f"记录了 {len(state_history)} 步状态变化")
    print(f"每步都可以访问: state_history[step]")
    print(f"最终归一化误差: {compute_normalization_error(final_vec):.2e}")

    print("\n示例完成！")
