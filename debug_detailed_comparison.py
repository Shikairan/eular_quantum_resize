#!/usr/bin/env python3
"""
详细调试：检查为什么前几个门误差为0
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

from utils import create_initial_vec_complex
from complexALL import process_sequence_complex

def bit_order_conversion(state_vector, n_qubits):
    """将DeepQuantum的大端序(MSB first)转换为complexALL的小端序(LSB first)"""
    converted = np.zeros_like(state_vector)
    for i in range(len(state_vector)):
        binary_str = format(i, f'0{n_qubits}b')
        reversed_str = binary_str[::-1]
        new_index = int(reversed_str, 2)
        converted[new_index] = state_vector[i]
    return converted

def debug_first_few_gates():
    """调试前几个门的详细状态变化"""
    if not DEEPQUANTUM_AVAILABLE:
        print("❌ DeepQuantum 未安装")
        return

    n_qubits = 4
    n_amps = 2**n_qubits

    # 创建初始向量
    initial_vec = create_initial_vec_complex(n_amps)
    initial_tensor = torch.from_numpy(initial_vec.astype(np.complex64))

    print("初始状态 (complexALL 格式):")
    for i in range(min(8, len(initial_vec))):
        print(".6f")

    # 手动创建几个测试门
    test_seq = [
        ('CS', '', [], 0, 3),  # CS门：控制0，目标3
        ('Y', '', [], 1),      # Y门：比特1
        ('Z', '', [], 1),      # Z门：比特1
    ]

    print("\n测试序列:")
    for i, gate in enumerate(test_seq, 1):
        if len(gate) == 4:
            print(f"  {i}. {gate[0]} 门 -> 比特 {gate[3]}")
        else:
            print(f"  {i}. {gate[0]} 门 -> 控制 {gate[3]}，目标 {gate[4]}")

    # 运行complexALL
    complex_final, complex_history = process_sequence_complex(test_seq, initial_vec.copy(), verbose=False)

    print("\ncomplexALL 状态变化:")
    for step in range(len(complex_history)):
        state = complex_history[step]
        print(f"\n步骤 {step} ({'初始状态' if step == 0 else test_seq[step-1][0]}):")
        for i in range(min(8, len(state))):
            if abs(state[i]) > 1e-10:  # 只显示非零元素
                print(".6f")

    # 运行DeepQuantum
    print("\nDeepQuantum 状态变化:")
    circuit = dq.QubitCircuit(nqubit=n_qubits, init_state=initial_tensor)

    # 初始状态
    dq_state_raw = circuit().cpu().numpy().flatten()
    dq_state_converted = bit_order_conversion(dq_state_raw, n_qubits)
    print("\n步骤 0 (初始状态):")
    for i in range(min(8, len(dq_state_converted))):
        if abs(dq_state_converted[i]) > 1e-10:
            print(".6f")

    # 应用每个门
    for step, gate_tuple in enumerate(test_seq):
        gate_name = gate_tuple[0]

        if len(gate_tuple) == 4:
            qubit_idx = int(gate_tuple[3])
            if gate_name == 'X':
                circuit.x(wires=qubit_idx)
            elif gate_name == 'Y':
                circuit.y(wires=qubit_idx)
            elif gate_name == 'Z':
                circuit.z(wires=qubit_idx)
            elif gate_name == 'H':
                circuit.h(wires=qubit_idx)
            elif gate_name == 'S':
                circuit.s(wires=qubit_idx)
        elif len(gate_tuple) == 5:
            control_idx, target_idx = int(gate_tuple[3]), int(gate_tuple[4])
            if gate_name == 'CX':
                circuit.cx(control=control_idx, target=target_idx)
            elif gate_name == 'CY':
                circuit.cy(control=control_idx, target=target_idx)
            elif gate_name == 'CZ':
                circuit.cz(control=control_idx, target=target_idx)
            elif gate_name == 'CS':
                circuit.cs(control=control_idx, target=target_idx)

        # 获取新状态
        dq_state_raw = circuit().cpu().numpy().flatten()
        dq_state_converted = bit_order_conversion(dq_state_raw, n_qubits)

        print(f"\n步骤 {step+1} ({gate_name}):")
        for i in range(min(8, len(dq_state_converted))):
            if abs(dq_state_converted[i]) > 1e-10:
                print(".6f")

        # 比较
        complex_state = complex_history[step + 1]
        diff = np.abs(complex_state - dq_state_converted)
        max_diff = np.max(diff)
        print(f"  最大差异: {max_diff:.2e}")

if __name__ == "__main__":
    debug_first_few_gates()