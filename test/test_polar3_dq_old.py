#!/usr/bin/env python3
"""
DeepQuantum vs polarALL_state_3 兼容性测试
只输出简洁的误差信息格式，类似test_error_state3.py
"""

import sys
import os
import math
import numpy as np
import torch

# 添加deepquantum到路径
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'deepquantum/src'))

try:
    import deepquantum as dq
    DEEPQUANTUM_AVAILABLE = True
except ImportError:
    DEEPQUANTUM_AVAILABLE = False
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils import create_initial_vec_complex, random_sequence

def bit_order_conversion(state_vector, n_qubits):
    """
    将DeepQuantum的大端序(MSB first)转换为polarALL_state_3的小端序(LSB first)

    DeepQuantum: |q0 q1 q2 ...⟩ where q0 is MSB
    polarALL_state_3:  |q0 q1 q2 ...⟩ where q0 is LSB

    Args:
        state_vector: DeepQuantum格式的状态向量
        n_qubits: 量子比特数量

    Returns:
        转换为polarALL_state_3格式的状态向量
    """
    converted = np.zeros_like(state_vector)

    for i in range(len(state_vector)):
        # 将索引i从MSB first转换为LSB first
        # 例如：对于3比特系统，DeepQuantum的索引2(10₂)对应polarALL_state_3的索引1(01₂)
        binary_str = format(i, f'0{n_qubits}b')
        # 反转比特序
        reversed_str = binary_str[::-1]
        new_index = int(reversed_str, 2)

        converted[new_index] = state_vector[i]

    return converted

def compute_state_difference(polar_state, dq_state):
    """计算两个状态向量之间的差异（来自test_error_state3.py）"""
    # 计算差异
    differences = []
    for i in range(len(polar_state)):
        diff = abs(polar_state[i] - dq_state[i])
        differences.append(diff)

    max_diff = max(differences)
    rms_diff = math.sqrt(sum(d**2 for d in differences) / len(differences))
    mean_diff = sum(differences) / len(differences)

    # 计算相对误差百分比
    error_percent = sum(np.array(differences)/(abs(polar_state)+1e-4)) / len(differences)
    return max_diff, rms_diff, mean_diff, float(error_percent)

def compare_implementations_polar3_dq(n_qubits=4, a=30, b=0, c=0):
    """
    比较polarALL_state_3和DeepQuantum的实现
    只输出简洁的误差信息格式，类似test_error_state3.py
    支持utils.py中random_sequence的所有门类型

    Args:
        n_qubits: 量子比特数量
    """
    if not DEEPQUANTUM_AVAILABLE:
        print("❌ DeepQuantum 未安装")
        return

    # 1. 创建初始向量
    initial_vec = create_initial_vec_complex(2**n_qubits)
    initial_tensor = torch.from_numpy(initial_vec.astype(np.complex64))

    # 2. 使用utils.py中的random_sequence生成序列
    # 参数: n=10(无参数单比特门), m=10(有参数门), amp_n=10(幅度相关门)
    seq = random_sequence(a, b, c, n_qubits=n_qubits, shuffle=False)

    # 3. 运行polarALL_state_3并获取历史
    from polarALL_state_3 import process_sequence_polar
    polar_final, polar_scale_final, polar_history = process_sequence_polar(initial_vec.copy(), seq, verbose=False)

    # 4. 比较初始状态
    dq_initial_converted = bit_order_conversion(initial_tensor.cpu().numpy(), n_qubits)

    # 将polar初始状态转换为复数形式
    from polarALL_state_3 import polar_to_complex_tensor
    polar_initial_complex = polar_to_complex_tensor(polar_history[0][0], polar_history[0][1]).cpu().numpy().flatten()

    max_diff, rms_diff, mean_diff, error_percent = compute_state_difference(polar_initial_complex, dq_initial_converted)
    print(f"门 0 (初始状态): max={max_diff:.2e}, rms={rms_diff:.2e}, mean={mean_diff:.2e} errorP={error_percent:.2e}")

    # 5. 逐步运行DeepQuantum并转换比特序
    gate_map = {
        # 无参数单比特门
        'X': lambda c, w: c.x(wires=w),
        'Y': lambda c, w: c.y(wires=w),
        'Z': lambda c, w: c.z(wires=w),
        'H': lambda c, w: c.h(wires=w),
        'S': lambda c, w: c.s(wires=w),
        'T': lambda c, w: c.t(wires=w),

        # 单参数单比特门
        'Rx': lambda c, w, p: c.rx(wires=w, inputs=p[0]),
        'Ry': lambda c, w, p: c.ry(wires=w, inputs=p[0]),
        'Rz': lambda c, w, p: c.rz(wires=w, inputs=p[0]),
        'U1': lambda c, w, p: c.rz(wires=w, inputs=p[0]),  # U1(λ) = Rz(λ)

        # 多参数单比特门
        'U2': lambda c, w, p: c.u3(wires=w, inputs=[p[0], p[1], 0]),  # U2(φ,λ) ≈ U3(π/2, φ, λ)
        'U3': lambda c, w, p: c.u3(wires=w, inputs=[p[0], p[1], p[2]]),

        # 无参数控制门
        'CX': lambda c, ctrl, targ: c.cx(control=ctrl, target=targ),
        'CY': lambda c, ctrl, targ: c.cy(control=ctrl, target=targ),
        'CZ': lambda c, ctrl, targ: c.cz(control=ctrl, target=targ),
        'CH': lambda c, ctrl, targ: c.ch(control=ctrl, target=targ),
        'CS': lambda c, ctrl, targ: c.cs(control=ctrl, target=targ),
        'CT': lambda c, ctrl, targ: c.ct(control=ctrl, target=targ),

        # 单参数控制门
        'CRx': lambda c, ctrl, targ, p: c.crx(control=ctrl, target=targ, inputs=p[0]),
        'CRy': lambda c, ctrl, targ, p: c.cry(control=ctrl, target=targ, inputs=p[0]),
        'CRz': lambda c, ctrl, targ, p: c.crz(control=ctrl, target=targ, inputs=p[0]),
        'CU1': lambda c, ctrl, targ, p: c.crz(control=ctrl, target=targ, inputs=p[0]),  # CU1(λ) = CRz(λ)

        # 多参数控制门
        'CU2': lambda c, ctrl, targ, p: c.cu(control=ctrl, target=targ, inputs=[p[0], p[1], 0]),  # 近似实现
        'CU3': lambda c, ctrl, targ, p: c.cu(control=ctrl, target=targ, inputs=[p[0], p[1], p[2]]),
    }

    circuit = dq.QubitCircuit(nqubit=n_qubits, init_state=initial_tensor)

    # 6. 逐个应用门并比较
    for step, gate_tuple in enumerate(seq):
        gate_name = gate_tuple[0]

        # 应用门到DeepQuantum电路
        params = gate_tuple[2]  # 参数列表

        if len(gate_tuple) == 4:
            # 单比特门 (gate_name, param_str, params, qubit_idx)
            qubit_idx = int(gate_tuple[3])
            if gate_name in gate_map:
                if params:
                    gate_map[gate_name](circuit, qubit_idx, params)
                else:
                    gate_map[gate_name](circuit, qubit_idx)
        elif len(gate_tuple) == 5:
            # 控制门 (gate_name, param_str, params, control_idx, target_idx)
            control_idx, target_idx = int(gate_tuple[3]), int(gate_tuple[4])
            if gate_name in gate_map:
                if params:
                    gate_map[gate_name](circuit, control_idx, target_idx, params)
                else:
                    gate_map[gate_name](circuit, control_idx, target_idx)

        # 获取当前状态并转换比特序
        dq_state_raw = circuit().cpu().numpy().flatten()
        dq_state_converted = bit_order_conversion(dq_state_raw, n_qubits)

        # 获取polarALL_state_3当前步骤的状态并转换为复数
        polar_state, polar_scale = polar_history[step + 1]
        polar_complex = polar_to_complex_tensor(polar_state, polar_scale).cpu().numpy().flatten()

        # 计算误差并打印（类似test_error_state3.py的格式）
        max_diff, rms_diff, mean_diff, error_percent = compute_state_difference(polar_complex, dq_state_converted)

        # 构建门信息字符串
        if len(gate_tuple) == 4:
            # 单比特门: (gate_name, param_str, params, qubit_idx)
            gate_name, param_str, _, qubit_idx = gate_tuple
            gate_str = f"{gate_name}(比特={qubit_idx})"
        elif len(gate_tuple) == 5:
            # 控制门: (gate_name, param_str, params, control_idx, target_idx)
            gate_name, param_str, _, control_idx, target_idx = gate_tuple
            gate_str = f"{gate_name}(控制={control_idx}, 目标={target_idx})"

        print(f"门 {step+1} ({gate_str}): max={max_diff:.2e}, rms={rms_diff:.2e}, mean={mean_diff:.2e} errorP={error_percent:.2e}")

if __name__ == "__main__":
    # 运行简洁版本测试（类似test_error_state3.py的输出格式）
    # 支持所有门类型：无参数单比特门、有参数门、幅度相关门
    compare_implementations_polar3_dq(n_qubits=14)
