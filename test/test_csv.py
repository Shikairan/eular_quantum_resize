#!/usr/bin/env python3
"""
DeepQuantum vs polarALL_state_int16 vs polarALL_state_int16_cdf 三系统兼容性测试 - CSV输出版本

同时比较三个量子计算系统的兼容性：
- DeepQuantum: 标准量子计算库
- polarALL_state_int16: 原始int16量化系统
- polarALL_state_int16_cdf: 极坐标CDF量化系统

使用相同的初始向量和门序列，确保公平比较
结果将保存到CSV文件中而不是打印到控制台
"""

import sys
import os
import math
import numpy as np
import torch
import csv
from datetime import datetime

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
from polarALL_state_int16_cdf import process_sequence_polar as process_polar_cdf
from vector_withCDF import *

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


def inverse_bit_order_conversion(state_vector, n_qubits):
    """
    将 polarALL_state_int16 的小端序(LSB first)转换为 DeepQuantum 的大端序(MSB first)

    polarALL_state_int16: |q0 q1 q2 ...⟩ where q0 is LSB
    DeepQuantum: |q0 q1 q2 ...⟩ where q0 is MSB

    Args:
        state_vector: polarALL_state_int16 格式的状态向量
        n_qubits: 量子比特数量

    Returns:
        转换为 DeepQuantum 格式的状态向量
    """
    converted = np.zeros_like(state_vector)

    for i in range(len(state_vector)):
        binary_str = format(i, f'0{n_qubits}b')
        reversed_str = binary_str[::-1]
        original_index = int(reversed_str, 2)
        converted[original_index] = state_vector[i]

    return converted


def compute_state_difference(state_a, state_b):
    """
    计算两个量子态（复数向量）之间的差异（仅加法/乘法，无除法）

    Args:
        state_a: 第一个量子态向量（numpy数组或可转换为numpy的序列）
        state_b: 第二个量子态向量（numpy数组或可转换为numpy的序列）

    Returns:
        (max_diff, total_diff, mean_diff)
        - max_diff: 最大绝对误差 max_i |a_i - b_i|
        - total_diff: 总误差和 sum_i |a_i - b_i|
        - mean_diff: 平均误差和 sum_i |a_i - b_i| / n
    Notes:
        - 使用复数128位精度计算以避免数值误差
        - 向量会自动展平处理，支持任意形状输入
        - 只使用加法、乘法和比较运算，避免除法操作
    """
    a = np.asarray(state_a, dtype=np.complex128).flatten()
    b = np.asarray(state_b, dtype=np.complex128).flatten()
    diff = np.abs(a - b)
    return float(np.max(diff)), float(np.sum(diff)), float(np.mean(diff))


def compare_three_implementations(n_qubits=4, a=30, b=0, c=0, csv_filename=None):
    """
    三系统对比测试：DeepQuantum vs polar原始 vs polar+CDF
    结果保存到CSV文件

    Args:
        n_qubits: 量子比特数量
        a, b, c: random_sequence 参数
        csv_filename: CSV文件名，如果为None则自动生成
    """
    if not DEEPQUANTUM_AVAILABLE:
        print("❌ DeepQuantum 未安装")
        return

    # 生成CSV文件名
    if csv_filename is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        csv_filename = f"quantum_comparison_{n_qubits}qubits_a{a}_b{b}_c{c}_{timestamp}.csv"

    print(f"开始测试: {n_qubits} 量子比特, 参数 a={a}, b={b}, c={c}")
    print(f"结果将保存到: {csv_filename}")

    # 1. 创建初始向量和序列（所有系统使用相同的）
    initial_vec = create_initial_vec_complex(2**n_qubits, False)
    initial_tensor = torch.from_numpy(initial_vec.astype(np.complex64))

    # 使用固定的种子确保序列可重现
    seq = random_sequence(a, b, c, n_qubits=n_qubits, shuffle=False)

    print(f"初始向量长度: {len(initial_vec)}")
    print(f"量子门序列长度: {len(seq)}")

    # 2. 运行 polarALL_state_int16_cdf (CDF系统)
    print("运行 polarALL_state_int16_cdf...")
    polar_wm_final, polar_wm_history = process_polar_cdf(
        initial_vec.copy(), seq, verbose=False
    )

    # 3. 运行 polarALL_state_int16 (原始系统)
    print("运行 polarALL_state_int16 (原始)...")
    polar_orig_final, polar_orig_scale_final, polar_orig_history = process_polar_original(
        initial_vec.copy(), seq, verbose=False
    )

    # 4. 比较初始状态
    dq_initial = initial_tensor.cpu().numpy()
    # CDF 系统的初始状态：从 PolarStateEncoded 解码并转换为 MSB first
    polar_wm_initial_complex = polar_wm_history[0].decode_state().cpu().numpy().flatten()
    polar_wm_initial_converted = inverse_bit_order_conversion(polar_wm_initial_complex, n_qubits)

    # 原始 polar 系统的初始状态：从 (polar_vec, scale_vec) 解码并转换为 MSB first
    polar_orig_initial_complex = polar_to_complex_original(
        polar_orig_history[0][0], polar_orig_history[0][1]
    ).cpu().numpy().flatten()
    polar_orig_initial_converted = inverse_bit_order_conversion(polar_orig_initial_complex, n_qubits)

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
    initial_tensor_converted = bit_order_conversion(initial_tensor, n_qubits)
    circuit = dq.QubitCircuit(nqubit=n_qubits, init_state=torch.from_numpy(initial_tensor_converted.astype(np.complex64)))

    # 准备CSV文件
    with open(csv_filename, 'w', newline='', encoding='utf-8') as csvfile:
        fieldnames = [
            'step', 'gate_operation', 'comparison_type',
            'max_diff', 'total_diff', 'mean_diff',
            'n_qubits', 'a', 'b', 'c', 'timestamp'
        ]
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()

        timestamp = datetime.now().isoformat()

        # 写入初始状态比较
        wm_max_diff, wm_total_diff, wm_mean_diff = compute_state_difference(
            polar_wm_initial_complex, dq_initial
        )
        writer.writerow({
            'step': -1,
            'gate_operation': 'INITIAL',
            'comparison_type': 'CDF_vs_DeepQ',
            'max_diff': wm_max_diff,
            'total_diff': wm_total_diff,
            'mean_diff': wm_mean_diff,
            'n_qubits': n_qubits,
            'a': a, 'b': b, 'c': c,
            'timestamp': timestamp
        })

        orig_max_diff, orig_total_diff, orig_mean_diff = compute_state_difference(
            polar_orig_initial_complex, dq_initial
        )
        writer.writerow({
            'step': -1,
            'gate_operation': 'INITIAL',
            'comparison_type': 'Orig_vs_DeepQ',
            'max_diff': orig_max_diff,
            'total_diff': orig_total_diff,
            'mean_diff': orig_mean_diff,
            'n_qubits': n_qubits,
            'a': a, 'b': b, 'c': c,
            'timestamp': timestamp
        })

        wm_orig_max_diff, wm_orig_total_diff, wm_orig_mean_diff = compute_state_difference(
            polar_wm_initial_converted, polar_orig_initial_converted
        )
        writer.writerow({
            'step': -1,
            'gate_operation': 'INITIAL',
            'comparison_type': 'CDF_vs_Orig',
            'max_diff': wm_orig_max_diff,
            'total_diff': wm_orig_total_diff,
            'mean_diff': wm_orig_mean_diff,
            'n_qubits': n_qubits,
            'a': a, 'b': b, 'c': c,
            'timestamp': timestamp
        })

        # 逐步执行门操作并记录结果
        for step, gate_tuple in enumerate(seq):
            if step % 10 == 0:
                print(f"处理步骤 {step}/{len(seq)}...")

            gate_name, _, params, control_idx, target_idx = gate_tuple
            if not isinstance(params, list):
                params = []

            # 应用门到 DeepQuantum 电路
            if control_idx is None:
                qubit_idx = int(target_idx)
                if gate_name in gate_map:
                    if params:
                        gate_map[gate_name](circuit, qubit_idx, params)
                    else:
                        gate_map[gate_name](circuit, qubit_idx)
            else:
                control_idx, target_idx = int(control_idx), int(target_idx)
                if gate_name in gate_map:
                    if params:
                        gate_map[gate_name](circuit, control_idx, target_idx, params)
                    else:
                        gate_map[gate_name](circuit, control_idx, target_idx)

            # 获取DeepQuantum结果
            dq_state_raw = circuit().cpu().numpy().flatten()
            dq_state_converted = inverse_bit_order_conversion(dq_state_raw, n_qubits)

            # CDF 系统当前状态
            polar_wm_state_obj = polar_wm_history[step + 1]
            polar_wm_complex = polar_wm_state_obj.decode_state().cpu().numpy().flatten()
            polar_wm_converted = inverse_bit_order_conversion(polar_wm_complex, n_qubits)

            # 原始 polar 系统当前状态
            polar_orig_state, polar_orig_scale = polar_orig_history[step + 1]
            polar_orig_complex = polar_to_complex_original(
                polar_orig_state, polar_orig_scale
            ).cpu().numpy().flatten()
            polar_orig_converted = inverse_bit_order_conversion(polar_orig_complex, n_qubits)

            # 计算差异
            wm_max_diff, wm_total_diff, wm_mean_diff = compute_state_difference(
                polar_wm_complex, dq_state_converted
            )
            orig_max_diff, orig_total_diff, orig_mean_diff = compute_state_difference(
                polar_orig_complex, dq_state_converted
            )
            wm_orig_max_diff, wm_orig_total_diff, wm_orig_mean_diff = compute_state_difference(
                polar_wm_converted, polar_orig_converted
            )

            # 构建门操作字符串
            if control_idx is None:
                gate_str = f"{gate_name}(qubit={target_idx})"
            else:
                gate_str = f"{gate_name}(control={control_idx}, target={target_idx})"

            # 写入CSV
            writer.writerow({
                'step': step,
                'gate_operation': gate_str,
                'comparison_type': 'CDF_vs_DeepQ',
                'max_diff': wm_max_diff,
                'total_diff': wm_total_diff,
                'mean_diff': wm_mean_diff,
                'n_qubits': n_qubits,
                'a': a, 'b': b, 'c': c,
                'timestamp': timestamp
            })

            writer.writerow({
                'step': step,
                'gate_operation': gate_str,
                'comparison_type': 'Orig_vs_DeepQ',
                'max_diff': orig_max_diff,
                'total_diff': orig_total_diff,
                'mean_diff': orig_mean_diff,
                'n_qubits': n_qubits,
                'a': a, 'b': b, 'c': c,
                'timestamp': timestamp
            })

            writer.writerow({
                'step': step,
                'gate_operation': gate_str,
                'comparison_type': 'CDF_vs_Orig',
                'max_diff': wm_orig_max_diff,
                'total_diff': wm_orig_total_diff,
                'mean_diff': wm_orig_mean_diff,
                'n_qubits': n_qubits,
                'a': a, 'b': b, 'c': c,
                'timestamp': timestamp
            })

    print(f"\n✅ 测试完成! 结果已保存到 {csv_filename}")
    print("✅ DeepQ: DeepQuantum作为标准参考")
    print("✅ 原始系统: polarALL_state_int16 (int16量化)")
    print("✅ CDF系统: polarALL_state_int16_cdf (极坐标CDF量化)")


if __name__ == "__main__":
    # 使用较小的参数进行测试
    compare_three_implementations(n_qubits=18, a=80, b=20, c=20)