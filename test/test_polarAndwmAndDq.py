#!/usr/bin/env python3
"""
DeepQuantum vs polarALL_state_int16 vs polarALL_state_int16_wm 三系统兼容性测试

同时比较两个polar量化系统与DeepQuantum的兼容性：
- polarALL_state_int16: 原始int16量化系统
- polarALL_state_int16_wm: 带WM变换的增强系统
- DeepQuantum: 作为标准参考

使用相同的initial_vec和random_sequence，确保公平比较
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
from polarALL_state_int16 import process_sequence_polar as process_polar_original, polar_to_complex_tensor as polar_to_complex_original
from polarALL_state_int16_wm import process_sequence_polar as process_polar_wm, polar_to_complex_tensor as polar_to_complex_wm


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


def compare_three_implementations(n_qubits=4, a=5, b=0, c=0):
    """
    三系统对比测试：DeepQuantum vs polar原始 vs polar+WM

    使用不同的量化配置确保两个系统有差异：
    - WM系统: polarALL_state_int16_wm，使用默认WM变换 (wm_a=4.3)
    - 原始系统: polarALL_state_int16，无WM变换

    Args:
        n_qubits: 量子比特数量
        a, b, c: random_sequence 参数
    """
    if not DEEPQUANTUM_AVAILABLE:
        print("❌ DeepQuantum 未安装")
        return

    print("=" * 80)
    print("三系统量子计算兼容性测试")
    print("=" * 80)
    print(f"系统配置: {n_qubits} 量子比特")
    print(f"序列参数: a={a}, b={b}, c={c}")
    print("WM系统: 使用默认WM变换 (wm_a=4.3)")
    print("原始系统: 无WM变换")
    print()

    # 1. 创建初始向量和序列（所有系统使用相同的）
    initial_vec = create_initial_vec_complex(2**n_qubits)
    initial_tensor = torch.from_numpy(initial_vec.astype(np.complex64))

    # 使用固定的种子确保序列可重现
    seq = random_sequence(a, b, c, n_qubits=n_qubits, shuffle=False)

    print(f"初始向量长度: {len(initial_vec)}")
    print(f"量子门序列长度: {len(seq)}")
    print()

    # 2. 运行 polarALL_state_int16_wm (带WM的系统)
    print("运行 polarALL_state_int16_wm (带WM)...")
    polar_wm_final, polar_wm_history = process_polar_wm(
        initial_vec.copy(), seq, verbose=False
    )

    # 3. 运行 polarALL_state_int16 (原始系统)
    print("运行 polarALL_state_int16 (原始)...")
    polar_orig_final, polar_orig_scale_final, polar_orig_history = process_polar_original(
        initial_vec.copy(), seq, verbose=False
    )

    # 4. 比较初始状态
    dq_initial_converted = bit_order_conversion(initial_tensor.cpu().numpy(), n_qubits)

    # WM系统的初始状态
    polar_wm_initial_complex = polar_to_complex_wm(
        polar_wm_history[0].get_polar_vec(), polar_wm_history[0].get_scale_vec()
    ).cpu().numpy().flatten()

    # 原始系统的初始状态
    polar_orig_initial_complex = polar_to_complex_original(
        polar_orig_history[0][0], polar_orig_history[0][1]
    ).cpu().numpy().flatten()

    print("初始状态对比:")
    print("-" * 50)

    # WM系统 vs DeepQuantum
    wm_max_diff, wm_rms_diff, wm_mean_diff, wm_error_percent = compute_state_difference(
        polar_wm_initial_complex, dq_initial_converted
    )
    print(f"WM系统   vs DeepQ: max={wm_max_diff:.2e}, rms={wm_rms_diff:.2e}, "
          f"mean={wm_mean_diff:.2e}, errorP={wm_error_percent:.2e}")

    # 原始系统 vs DeepQuantum
    orig_max_diff, orig_rms_diff, orig_mean_diff, orig_error_percent = compute_state_difference(
        polar_orig_initial_complex, dq_initial_converted
    )
    print(f"原始系统 vs DeepQ: max={orig_max_diff:.2e}, rms={orig_rms_diff:.2e}, "
          f"mean={orig_mean_diff:.2e}, errorP={orig_error_percent:.2e}")
    print()

    # 5. 设置DeepQuantum门映射
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

    # 6. 逐步运行DeepQuantum并与两个polar系统比较
    circuit = dq.QubitCircuit(nqubit=n_qubits, init_state=initial_tensor)

    print("各量子门后的状态对比:")
    print("-" * 80)
    print(f"{'Gate Operation':<25} {'WM vs DeepQ':<12} {'Orig vs DeepQ':<12}")
    print("-" * 80)

    for step, gate_tuple in enumerate(seq):
        gate_name = gate_tuple[0]
        params = gate_tuple[2]

        # 应用门到DeepQuantum电路
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

        # 获取DeepQuantum结果
        dq_state_raw = circuit().cpu().numpy().flatten()
        dq_state_converted = bit_order_conversion(dq_state_raw, n_qubits)

        # 获取WM系统结果
        polar_wm_state_obj = polar_wm_history[step + 1]
        polar_wm_complex = polar_to_complex_wm(
            polar_wm_state_obj.get_polar_vec(), polar_wm_state_obj.get_scale_vec()
        ).cpu().numpy().flatten()

        # 获取原始系统结果
        polar_orig_state, polar_orig_scale = polar_orig_history[step + 1]
        polar_orig_complex = polar_to_complex_original(
            polar_orig_state, polar_orig_scale
        ).cpu().numpy().flatten()

        # 计算WM系统误差
        wm_max_diff, wm_rms_diff, wm_mean_diff, wm_error_percent = compute_state_difference(
            polar_wm_complex, dq_state_converted
        )

        # 计算原始系统误差
        orig_max_diff, orig_rms_diff, orig_mean_diff, orig_error_percent = compute_state_difference(
            polar_orig_complex, dq_state_converted
        )

        # 格式化门信息
        if len(gate_tuple) == 4:
            gate_name, param_str, _, qubit_idx = gate_tuple
            gate_str = f"{gate_name}(比特={qubit_idx})"
        elif len(gate_tuple) == 5:
            gate_name, param_str, _, control_idx, target_idx = gate_tuple
            gate_str = f"{gate_name}(控制={control_idx}, 目标={target_idx})"

        # 打印对比结果
        print(f"{gate_str:<25} "
              f"{wm_error_percent:.2e} "
              f"{orig_error_percent:.2e}")

    print("-" * 80)
    print("\n测试完成!")
    print("✅ WM系统: 带水印变换的增强量化系统")
    print("✅ 原始系统: 标准int16量化系统")
    print("✅ DeepQ: DeepQuantum作为参考标准")


if __name__ == "__main__":
    # 使用较小的参数进行快速测试
    compare_three_implementations(n_qubits=16, a=40, b=0, c=0)