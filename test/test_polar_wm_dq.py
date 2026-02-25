#!/usr/bin/env python3
"""
DeepQuantum vs polarALL_state_int16 兼容性测试
参考 test_polar3_dq_old.py，使用 polarALL_state_int16 替代 polarALL_state_3
只输出简洁的误差信息格式，类似 test_error_state3.py
"""

import sys
import os
import math
import numpy as np
import torch

# 添加 deepquantum 到路径
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'deepquantum/src'))

try:
    import deepquantum as dq
    DEEPQUANTUM_AVAILABLE = True
except ImportError:
    DEEPQUANTUM_AVAILABLE = False

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils import create_initial_vec_complex, random_sequence
from polarALL_state_int16_wm import process_sequence_polar, polar_to_complex_tensor
from vector_withWM import PolarVector

# 兼容性测试用向量：与 polarALL_state_int16_wm 保持一致（int16+int16, WM开启）
# 注意：phase_dtype 必须为 int16，否则 H 门等会引入巨大相位量化误差（int8 仅 256 级 vs int16 的 65535 级）
vector_weak_wm = PolarVector(amplitude_dtype=torch.int16, phase_dtype=torch.int8, wm_a=8)
print(f"Polar vector info: {vector_weak_wm.get_info()}")

def bit_order_conversion(state_vector, n_qubits):
    """
    将 DeepQuantum 的大端序(MSB first)转换为 polarALL_state_int16 的小端序(LSB first)

    DeepQuantum: |q0 q1 q2 ...⟩ where q0 is MSB
    polarALL_state_int16: |q0 q1 q2 ...⟩ where q0 is LSB

    Args:
        state_vector: DeepQuantum 格式的状态向量
        n_qubits: 量子比特数量

    Returns:
        转换为 polarALL_state_int16 格式的状态向量
    """
    converted = np.zeros_like(state_vector)

    for i in range(len(state_vector)):
        binary_str = format(i, f'0{n_qubits}b')
        reversed_str = binary_str[::-1]
        new_index = int(reversed_str, 2)
        converted[new_index] = state_vector[i]

    return converted


def compute_state_difference(polar_state, dq_state):
    """计算两个状态向量之间的差异"""
    differences = []
    for i in range(len(polar_state)):
        diff = abs(polar_state[i] - dq_state[i])
        differences.append(diff)

    max_diff = max(differences)
    rms_diff = math.sqrt(sum(d**2 for d in differences) / len(differences))
    mean_diff = sum(differences) / len(differences)
    error_percent = sum(np.array(differences) / (abs(polar_state) + 1e-4)) / len(differences)
    return max_diff, rms_diff, mean_diff, float(error_percent)


def compare_implementations_polar_dq(n_qubits=4, a=30, b=0, c=0, use_weak_wm=True):
    """
    比较 polarALL_state_int16 和 DeepQuantum 的实现
    支持 utils.py 中 random_sequence 的所有门类型

    Args:
        n_qubits: 量子比特数量
        a, b, c: random_sequence 参数
        use_weak_wm: 是否使用弱WM变换以提高兼容性
    """
    if not DEEPQUANTUM_AVAILABLE:
        print("❌ DeepQuantum 未安装")
        return

    # 临时修改polarALL_state_int16_wm模块的vector实例
    if use_weak_wm:
        import polarALL_state_int16_wm
        original_vector = polarALL_state_int16_wm.vector
        polarALL_state_int16_wm.vector = vector_weak_wm

    try:
        # 1. 创建初始向量
        initial_vec = create_initial_vec_complex(2**n_qubits)
        initial_tensor = torch.from_numpy(initial_vec.astype(np.complex64))

        # 2. 使用 utils.py 中的 random_sequence 生成序列
        seq = random_sequence(a, b, c, n_qubits=n_qubits, shuffle=False)

        # 3. 运行 polarALL_state_int16_wm 并获取历史
        polar_final_state, polar_history = process_sequence_polar(
            initial_vec.copy(), seq, verbose=False
        )

        # 4. 比较初始状态（直接比较输入向量，不考虑WM变换）
        dq_initial_converted = bit_order_conversion(initial_tensor.cpu().numpy(), n_qubits)

        max_diff, rms_diff, mean_diff, error_percent = compute_state_difference(
            initial_vec, dq_initial_converted
        )
        print(f"门 0 (初始状态): max={max_diff:.2e}, rms={rms_diff:.2e}, mean={mean_diff:.2e} errorP={error_percent:.2e}")

        # 5. 逐步运行 DeepQuantum 并转换比特序
        gate_map = {
            'X': lambda c, w: c.x(wires=w),
            'Y': lambda c, w: c.y(wires=w),
            'Z': lambda c, w: c.z(wires=w),
            'H': lambda c, w: c.h(wires=w),
            'S': lambda c, w: c.s(wires=w),
            'T': lambda c, w: c.t(wires=w),
            'Rx': lambda c, w, p: c.rx(wires=w, inputs=p[0]),
            'Ry': lambda c, w, p: c.ry(wires=w, inputs=p[0]),
            'Rz': lambda c, w, p: c.rz(wires=w, inputs=p[0]),
            'U1': lambda c, w, p: c.rz(wires=w, inputs=p[0]),
            'U2': lambda c, w, p: c.u3(wires=w, inputs=[p[0], p[1], 0]),
            'U3': lambda c, w, p: c.u3(wires=w, inputs=[p[0], p[1], p[2]]),
            'CX': lambda c, ctrl, targ: c.cx(control=ctrl, target=targ),
            'CY': lambda c, ctrl, targ: c.cy(control=ctrl, target=targ),
            'CZ': lambda c, ctrl, targ: c.cz(control=ctrl, target=targ),
            'CH': lambda c, ctrl, targ: c.ch(control=ctrl, target=targ),
            'CS': lambda c, ctrl, targ: c.cs(control=ctrl, target=targ),
            'CT': lambda c, ctrl, targ: c.ct(control=ctrl, target=targ),
            'CRx': lambda c, ctrl, targ, p: c.crx(control=ctrl, target=targ, inputs=p[0]),
            'CRy': lambda c, ctrl, targ, p: c.cry(control=ctrl, target=targ, inputs=p[0]),
            'CRz': lambda c, ctrl, targ, p: c.crz(control=ctrl, target=targ, inputs=p[0]),
            'CU1': lambda c, ctrl, targ, p: c.crz(control=ctrl, target=targ, inputs=p[0]),
            'CU2': lambda c, ctrl, targ, p: c.cu(control=ctrl, target=targ, inputs=[p[0], p[1], 0]),
            'CU3': lambda c, ctrl, targ, p: c.cu(control=ctrl, target=targ, inputs=[p[0], p[1], p[2]]),
        }

        circuit = dq.QubitCircuit(nqubit=n_qubits, init_state=initial_tensor)

        # 6. 逐个应用门并比较
        for step, gate_tuple in enumerate(seq):
            gate_name = gate_tuple[0]
            params = gate_tuple[2]

            if len(gate_tuple) == 4:
                qubit_idx = int(gate_tuple[3])
                if gate_name in gate_map:
                    if params:
                        gate_map[gate_name](circuit, qubit_idx, params)
                    else:
                        gate_map[gate_name](circuit, qubit_idx)
            elif len(gate_tuple) == 5:
                control_idx, target_idx = int(gate_tuple[3]), int(gate_tuple[4])
                if gate_name in gate_map:
                    if params:
                        gate_map[gate_name](circuit, control_idx, target_idx, params)
                    else:
                        gate_map[gate_name](circuit, control_idx, target_idx)

            dq_state_raw = circuit().cpu().numpy().flatten()
            dq_state_converted = bit_order_conversion(dq_state_raw, n_qubits)

            polar_state_obj = polar_history[step + 1]
            polar_complex = polar_to_complex_tensor(polar_state_obj).cpu().numpy().flatten()

            max_diff, rms_diff, mean_diff, error_percent = compute_state_difference(
                polar_complex, dq_state_converted
            )

            if len(gate_tuple) == 4:
                gate_name, param_str, _, qubit_idx = gate_tuple
                gate_str = f"{gate_name}(比特={qubit_idx})"
            elif len(gate_tuple) == 5:
                gate_name, param_str, _, control_idx, target_idx = gate_tuple
                gate_str = f"{gate_name}(控制={control_idx}, 目标={target_idx})"

            print(f"门 {step+1} ({gate_str}): max={max_diff:.2e}, rms={rms_diff:.2e}, mean={mean_diff:.2e} errorP={error_percent:.2e}")

    finally:
        # 恢复原始vector实例
        if use_weak_wm:
            polarALL_state_int16_wm.vector = original_vector


if __name__ == "__main__":
    compare_implementations_polar_dq(n_qubits=22, a=120, b=0, c=0)  # 使用较小的参数进行测试
