#!/usr/bin/env python3
"""
DeepQuantum vs complexALL 兼容性测试
解决比特序差异问题，正确比较两个实现的数值误差
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

from utils import create_initial_vec_complex, random_sequence

def bit_order_conversion(state_vector, n_qubits):
    """
    将DeepQuantum的大端序(MSB first)转换为complexALL的小端序(LSB first)

    DeepQuantum: |q0 q1 q2 ...⟩ where q0 is MSB
    complexALL:  |q0 q1 q2 ...⟩ where q0 is LSB

    Args:
        state_vector: DeepQuantum格式的状态向量
        n_qubits: 量子比特数量

    Returns:
        转换为complexALL格式的状态向量
    """
    converted = np.zeros_like(state_vector)

    for i in range(len(state_vector)):
        # 将索引i从MSB first转换为LSB first
        # 例如：对于3比特系统，DeepQuantum的索引2(10₂)对应complexALL的索引1(01₂)
        binary_str = format(i, f'0{n_qubits}b')
        # 反转比特序
        reversed_str = binary_str[::-1]
        new_index = int(reversed_str, 2)

        converted[new_index] = state_vector[i]

    return converted

def compute_state_difference(complex_state, dq_state):
    """计算两个状态向量之间的差异（来自test_error_state3.py）"""
    # 计算差异
    differences = []
    for i in range(len(complex_state)):
        diff = abs(complex_state[i] - dq_state[i])
        differences.append(diff)

    max_diff = max(differences)
    rms_diff = math.sqrt(sum(d**2 for d in differences) / len(differences))
    mean_diff = sum(differences) / len(differences)

    # 计算相对误差百分比
    error_percent = sum(np.array(differences)/(abs(complex_state)+1e-4)) / len(differences)
    return max_diff, rms_diff, mean_diff,  np.linalg.norm(complex_state - dq_state)#float(error_percent)

def compare_implementations_corrected(n_qubits=3, seq_length=10):
    """
    正确比较DeepQuantum和complexALL的实现（考虑比特序差异）
    只输出简洁的误差信息格式，类似test_error_state3.py

    Args:
        n_qubits: 量子比特数量
        seq_length: 序列长度
    """
    if not DEEPQUANTUM_AVAILABLE:
        print("❌ DeepQuantum 未安装")
        return

    # 1. 创建初始向量
    initial_vec = create_initial_vec_complex(2**n_qubits)
    initial_tensor = torch.from_numpy(initial_vec.astype(np.complex64))

    # 2. 生成相同的随机序列（只使用支持的门）
    seq = []
    for _ in range(seq_length):
        # 随机选择单比特门
        single_gates = ['X', 'Y', 'Z', 'H', 'S', 'T']
        control_gates = ['CX', 'CY', 'CZ', 'CH', 'CS', 'CT']

        if np.random.random() < 0.6:  # 60%概率选择单比特门
            gate_name = np.random.choice(single_gates)
            qubit_idx = np.random.randint(0, n_qubits)
            seq.append((gate_name, '', [], qubit_idx))
        else:  # 40%概率选择控制门
            gate_name = np.random.choice(control_gates)
            qubits = np.random.choice(n_qubits, 2, replace=False)
            control_idx, target_idx = qubits[0], qubits[1]
            seq.append((gate_name, '', [], control_idx, target_idx))

    # 3. 运行complexALL并获取历史
    from complexALL import process_sequence_complex
    complex_final, complex_history = process_sequence_complex(seq, initial_vec.copy(), verbose=False)

    # 4. 比较初始状态
    dq_initial_converted = bit_order_conversion(initial_tensor.cpu().numpy(), n_qubits)
    max_diff, rms_diff, mean_diff, error_percent = compute_state_difference(complex_history[0], dq_initial_converted)
    print(f"门 0 (初始状态): max={max_diff:.2e}, rms={rms_diff:.2e}, mean={mean_diff:.2e} errorP={error_percent:.2e}")

    # 5. 逐步运行DeepQuantum并转换比特序
    gate_map = {
        'X': lambda c, w: c.x(wires=w),
        'Y': lambda c, w: c.y(wires=w),
        'Z': lambda c, w: c.z(wires=w),
        'H': lambda c, w: c.h(wires=w),
        'S': lambda c, w: c.s(wires=w),
        'T': lambda c, w: c.t(wires=w),
        'CX': lambda c, ctrl, targ: c.cx(control=ctrl, target=targ),
        'CY': lambda c, ctrl, targ: c.cy(control=ctrl, target=targ),
        'CZ': lambda c, ctrl, targ: c.cz(control=ctrl, target=targ),
        'CH': lambda c, ctrl, targ: c.ch(control=ctrl, target=targ),
        'CS': lambda c, ctrl, targ: c.cs(control=ctrl, target=targ),
        'CT': lambda c, ctrl, targ: c.ct(control=ctrl, target=targ),
    }

    circuit = dq.QubitCircuit(nqubit=n_qubits, init_state=initial_tensor)

    # 6. 逐个应用门并比较
    for step, gate_tuple in enumerate(seq):
        gate_name = gate_tuple[0]

        # 应用门到DeepQuantum电路
        if len(gate_tuple) == 4:
            # 单比特门
            qubit_idx = int(gate_tuple[3])
            if gate_name in gate_map:
                gate_map[gate_name](circuit, qubit_idx)
        elif len(gate_tuple) == 5:
            # 控制门
            control_idx, target_idx = int(gate_tuple[3]), int(gate_tuple[4])
            if gate_name in gate_map:
                gate_map[gate_name](circuit, control_idx, target_idx)

        # 获取当前状态并转换比特序
        dq_state_raw = circuit().cpu().numpy().flatten()
        dq_state_converted = bit_order_conversion(dq_state_raw, n_qubits)

        # 比较当前步骤的状态
        complex_state = complex_history[step + 1]

        # 计算误差并打印（类似test_error_state3.py的格式）
        max_diff, rms_diff, mean_diff, error_percent = compute_state_difference(complex_state, dq_state_converted)

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

def run_multiple_tests():
    """运行多个测试配置"""
    print("🔬 多配置兼容性测试")
    print("=" * 70)

    test_configs = [
        {"n_qubits": 2, "seq_length": 8, "description": "小型系统"},
        {"n_qubits": 3, "seq_length": 12, "description": "中等系统"},
        {"n_qubits": 4, "seq_length": 16, "description": "较大系统"},
    ]

    results = []

    for config in test_configs:
        print(f"\n🧪 测试配置: {config['description']} ({config['n_qubits']} 量子比特, {config['seq_length']} 门)")
        print("-" * 60)

        result = compare_implementations_corrected(
            n_qubits=config['n_qubits'],
            seq_length=config['seq_length']
        )

        if result:
            results.append({
                'config': config,
                'result': result
            })

            # 简要显示结果
            print(f"   兼容性: {result['compatibility']}")
    # 总体总结
    if results:
        print("\n🎯 总体测试总结")
        print("=" * 50)

        compatibilities = [r['result']['compatibility'] for r in results]
        avg_errors = [r['result']['avg_relative_error'] for r in results]
        max_errors = [r['result']['max_relative_error'] for r in results]

        print(f"   测试配置数: {len(results)}")
        print(f"   兼容性分布: {compatibilities}")

        if all(c == "excellent" for c in compatibilities):
            print("   🎉 结论：DeepQuantum与complexALL完全兼容！")
        elif all(c in ["excellent", "good"] for c in compatibilities):
            print("   ✅ 结论：DeepQuantum与complexALL高度兼容！")
        else:
            print("   ⚠️  结论：DeepQuantum与complexALL存在一些兼容性问题")

if __name__ == "__main__":
    # 运行简洁版本测试（类似test_error_state3.py的输出格式）
    compare_implementations_corrected(n_qubits=16, seq_length=120)
