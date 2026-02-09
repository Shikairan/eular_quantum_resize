#!/usr/bin/env python3
"""
调试矩阵乘法：比较DeepQuantum和complexALL的X门和H门实现
"""

import sys
import os
import numpy as np
import torch
import math

# 添加deepquantum到路径
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'deepquantum/src'))

try:
    import deepquantum as dq
    DEEPQUANTUM_AVAILABLE = True
except ImportError:
    DEEPQUANTUM_AVAILABLE = False

def debug_x_gate():
    """调试X门"""
    print("🔍 调试X门实现")

    if not DEEPQUANTUM_AVAILABLE:
        print("❌ DeepQuantum 未安装")
        return

    # 创建X门实例
    x_gate = dq.PauliX()
    print(f"DeepQuantum X门矩阵:\n{x_gate.matrix}")

    # 我们的X门矩阵
    our_x = np.array([[0, 1], [1, 0]], dtype=complex)
    print(f"我们的X门矩阵:\n{our_x}")

    # 测试初始状态 |0⟩
    initial_state = np.array([1.0, 0.0], dtype=complex)
    print(f"\n初始状态 |0⟩: {initial_state}")

    # DeepQuantum应用X门
    dq_result = (x_gate.matrix @ torch.from_numpy(initial_state).to(torch.cfloat)).cpu().numpy()
    print(f"DeepQuantum X|0⟩: {dq_result}")

    # 我们的实现应用X门
    our_result = our_x @ initial_state
    print(f"我们的实现 X|0⟩: {our_result}")

    # 比较
    diff = np.abs(dq_result - our_result)
    print(f"差异: {diff}")
    print(f"最大差异: {np.max(diff)}")

def debug_h_gate():
    """调试H门"""
    print("\n🔍 调试H门实现")

    if not DEEPQUANTUM_AVAILABLE:
        print("❌ DeepQuantum 未安装")
        return

    # 创建H门实例
    h_gate = dq.Hadamard()
    print(f"DeepQuantum H门矩阵:\n{h_gate.matrix}")

    # 我们的H门矩阵
    our_h = np.array([[1, 1], [1, -1]], dtype=complex) / math.sqrt(2)
    print(f"我们的H门矩阵:\n{our_h}")

    # 测试初始状态 |0⟩
    initial_state = np.array([1.0, 0.0], dtype=complex)
    print(f"\n初始状态 |0⟩: {initial_state}")

    # DeepQuantum应用H门
    dq_result = (h_gate.matrix @ torch.from_numpy(initial_state).to(torch.cfloat)).cpu().numpy()
    print(f"DeepQuantum H|0⟩: {dq_result}")

    # 我们的实现应用H门
    our_result = our_h @ initial_state
    print(f"我们的实现 H|0⟩: {our_result}")

    # 比较
    diff = np.abs(dq_result - our_result)
    print(f"差异: {diff}")
    print(f"最大差异: {np.max(diff)}")

def debug_full_system():
    """调试完整系统（多比特）"""
    print("\n🔍 调试多比特系统")

    if not DEEPQUANTUM_AVAILABLE:
        print("❌ DeepQuantum 未安装")
        return

    # 2比特系统，初始状态 |00⟩
    n_qubits = 2
    n_amps = 4

    initial_vec = np.zeros(n_amps, dtype=complex)
    initial_vec[0] = 1.0  # |00⟩

    print(f"初始状态 |00⟩: {initial_vec}")

    # 测试X门在第0比特
    print("\n测试 X门在第0比特:")

    # DeepQuantum方式
    circuit = dq.QubitCircuit(nqubit=n_qubits, init_state=torch.from_numpy(initial_vec.astype(np.complex64)))
    circuit.x(wires=0)
    dq_result = circuit().cpu().numpy().flatten()
    print(f"DeepQuantum 结果: {dq_result}")

    # 我们的方式
    # X门在第0比特相当于交换状态对 (0,1), (2,3), (4,5), (6,7), 但对于2比特系统只有 (0,1)
    our_result = initial_vec.copy()
    our_result[0], our_result[1] = our_result[1], our_result[0]  # 交换 |00⟩ 和 |01⟩
    print(f"我们的实现结果: {our_result}")

    # 比较
    diff = np.abs(dq_result - our_result)
    print(f"差异: {diff}")
    print(f"最大差异: {np.max(diff)}")

    # 测试H门在第0比特
    print("\n测试 H门在第0比特:")

    # DeepQuantum方式
    circuit2 = dq.QubitCircuit(nqubit=n_qubits, init_state=torch.from_numpy(initial_vec.astype(np.complex64)))
    circuit2.h(wires=0)
    dq_result2 = circuit2().cpu().numpy().flatten()
    print(f"DeepQuantum 结果: {dq_result2}")

    # 我们的方式
    # H门在第0比特相当于对每对状态 (2*i, 2*i+1) 应用H门
    our_result2 = np.zeros_like(initial_vec)
    for i in range(2):  # 对于2比特系统，有2对状态
        base_idx = i * 2
        state_pair = initial_vec[base_idx:base_idx+2]
        h_result = np.array([[1, 1], [1, -1]], dtype=complex) @ state_pair / math.sqrt(2)
        our_result2[base_idx:base_idx+2] = h_result

    print(f"我们的实现结果: {our_result2}")

    # 比较
    diff2 = np.abs(dq_result2 - our_result2)
    print(f"差异: {diff2}")
    print(f"最大差异: {np.max(diff2)}")

def check_gate_application_method():
    """检查门的应用方法"""
    print("\n🔍 检查门的应用方法")

    if not DEEPQUANTUM_AVAILABLE:
        print("❌ DeepQuantum 未安装")
        return

    # 创建一个简单的2比特电路
    n_qubits = 2
    initial_vec = np.array([1.0, 0.0, 0.0, 0.0], dtype=complex)  # |00⟩

    circuit = dq.QubitCircuit(nqubit=n_qubits, init_state=torch.from_numpy(initial_vec.astype(np.complex64)))

    print(f"初始状态: {circuit().cpu().numpy().flatten()}")

    # 手动检查X门的应用
    x_gate = dq.PauliX(nqubit=2, wires=[0])  # X门作用在第0比特
    print(f"X门矩阵:\n{x_gate.matrix}")
    print(f"X门wires: {x_gate.wires}")
    print(f"X门controls: {x_gate.controls}")

    # 应用门
    circuit.x(wires=0)
    result = circuit().cpu().numpy().flatten()
    print(f"应用X门后的状态: {result}")

if __name__ == "__main__":
    debug_x_gate()
    debug_h_gate()
    debug_full_system()
    check_gate_application_method()