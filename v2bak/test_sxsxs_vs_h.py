"""
测试 S·X·S·X·S 组合门与 H 门在 5 比特系统上的效果对比
使用纯 NumPy 实现，不依赖 ALLeular 库
"""

import numpy as np
import math

def create_state_vector(n_qubits, initial_state=None):
    """创建 n 比特系统的状态向量"""
    if initial_state is None:
        # 创建随机初始状态
        real_parts = np.random.randn(2**n_qubits)
        imag_parts = np.random.randn(2**n_qubits)
        state = real_parts + 1j * imag_parts
    else:
        # 使用指定的初始状态
        state = np.zeros(2**n_qubits, dtype=complex)
        state[initial_state] = 1.0

    # 归一化
    norm = np.sqrt(np.sum(np.abs(state)**2))
    return state / norm

def apply_single_qubit_gate(state, gate_matrix, qubit_idx, n_qubits):
    """对指定比特应用单比特门"""
    dim = 2**n_qubits

    # 创建完整的门矩阵（张量积）
    full_gate = np.array([[1.0]], dtype=complex)

    for i in range(n_qubits):
        if i == qubit_idx:
            full_gate = np.kron(full_gate, gate_matrix)
        else:
            identity = np.eye(2, dtype=complex)
            full_gate = np.kron(full_gate, identity)

    # 应用门
    return full_gate @ state

def get_s_gate():
    """S 门矩阵"""
    return np.array([[1.0, 0.0], [0.0, 1j]], dtype=complex)

def get_x_gate():
    """X 门矩阵"""
    return np.array([[0.0, 1.0], [1.0, 0.0]], dtype=complex)

def get_h_gate():
    """H 门矩阵"""
    return np.array([[1.0, 1.0], [1.0, -1.0]], dtype=complex) / np.sqrt(2)

def apply_sxsxs_sequence(state, qubit_idx, n_qubits):
    """应用 S·X·S·X·S 序列"""
    s_gate = get_s_gate()
    x_gate = get_x_gate()

    # S
    result = apply_single_qubit_gate(state, s_gate, qubit_idx, n_qubits)
    # X
    result = apply_single_qubit_gate(result, x_gate, qubit_idx, n_qubits)
    # S
    result = apply_single_qubit_gate(result, s_gate, qubit_idx, n_qubits)
    # X
    result = apply_single_qubit_gate(result, x_gate, qubit_idx, n_qubits)
    # S
    result = apply_single_qubit_gate(result, s_gate, qubit_idx, n_qubits)

    return result

def apply_h_gate(state, qubit_idx, n_qubits):
    """应用 H 门"""
    h_gate = get_h_gate()
    return apply_single_qubit_gate(state, h_gate, qubit_idx, n_qubits)

def compute_norm_error(state):
    """计算归一化误差"""
    norm = np.sqrt(np.sum(np.abs(state)**2))
    return abs(norm - 1.0), norm

def analyze_state_vector(state, name):
    """分析状态向量"""
    print(f"\n=== {name} ===")

    # 基本信息
    n_qubits = int(np.log2(len(state)))
    print(f"系统大小: {n_qubits} 比特 ({len(state)} 个状态)")

    # 归一化检查
    norm_error, actual_norm = compute_norm_error(state)
    print(f"实际范数: {actual_norm:.10f}")
    print(f"归一化误差: {norm_error:.2e}")

    # 最大振幅的基态
    max_idx = np.argmax(np.abs(state))
    max_amp = state[max_idx]
    max_prob = np.abs(max_amp)**2

    print(f"最大振幅状态: |{max_idx:0{n_qubits}b}⟩")
    print(f"最大振幅值: {max_amp:.6f}")
    print(f"最大概率: {max_prob:.6f}")

    # 熵计算
    probabilities = np.abs(state)**2
    # 避免 log(0)
    probabilities = np.where(probabilities > 1e-15, probabilities, 1e-15)
    entropy = -np.sum(probabilities * np.log2(probabilities))
    print(f"冯诺依曼熵: {entropy:.6f}")

    return norm_error, actual_norm, entropy

def compare_gates():
    """比较两种门的最终效果"""
    print("S·X·S·X·S 组合门 vs H 门 对比测试")
    print("=" * 60)

    n_qubits = 5
    qubit_idx = 0  # 在第 0 个比特上应用门

    # 测试不同的初始状态
    initial_states = [0, 1, 15, None]  # |00000⟩, |00001⟩, |01111⟩, 随机状态

    for init_state in initial_states:
        print(f"\n测试初始状态: {'随机' if init_state is None else f'|{init_state:05b}⟩'}")
        print("-" * 40)

        # 创建初始状态
        initial_state = create_state_vector(n_qubits, init_state)

        # 应用 S·X·S·X·S 序列
        sxsxs_result = apply_sxsxs_sequence(initial_state.copy(), qubit_idx, n_qubits)

        # 应用 H 门
        h_result = apply_h_gate(initial_state.copy(), qubit_idx, n_qubits)

        # 分析结果
        sxsxs_error, sxsxs_norm, sxsxs_entropy = analyze_state_vector(sxsxs_result, "S·X·S·X·S 结果")
        h_error, h_norm, h_entropy = analyze_state_vector(h_result, "H 门结果")

        # 比较两个结果的差异
        state_diff = np.abs(sxsxs_result - h_result)
        max_state_diff = np.max(state_diff)
        rms_state_diff = np.sqrt(np.mean(state_diff**2))

        print("\n=== 结果对比 ===")
        print(f"最大状态差异: {max_state_diff:.2e}")
        print(f"RMS 状态差异: {rms_state_diff:.2e}")
        print(f"熵差异: {abs(sxsxs_entropy - h_entropy):.6f}")

        # 判断是否等价
        if max_state_diff < 1e-10:
            print("✅ 结果完全等价")
        elif max_state_diff < 1e-6:
            print("✅ 结果基本等价 (在数值精度范围内)")
        else:
            print("❌ 结果显著不同")

def theoretical_analysis():
    """理论分析 S·X·S·X·S 的效果"""
    print("\n" + "="*60)
    print("理论分析：S·X·S·X·S 的数学效果")
    print("=" * 60)

    # S 门矩阵
    S = get_s_gate()
    # X 门矩阵
    X = get_x_gate()
    # H 门矩阵
    H = get_h_gate()

    print("基础门矩阵:")
    print(f"S 门 = \n{S}")
    print(f"X 门 = \n{X}")
    print(f"H 门 = \n{H}")
    print()

    # 计算 S·X·S·X·S
    sxsxs = S @ X @ S @ X @ S

    print("S·X·S·X·S 组合矩阵:")
    print(sxsxs)
    print()

    # 计算 S·X·S·X·S 与 H 的差异
    diff = sxsxs - H
    max_diff = np.max(np.abs(diff))

    print(f"S·X·S·X·S 与 H 的最大差异: {max_diff:.2e}")

    if max_diff < 1e-10:
        print("✅ 理论上：S·X·S·X·S = H")
        print("✅ 这是一个有趣的数学恒等式！")
    else:
        print("❌ 理论上不等价")

    print("\n数学解释:")
    print("S·X·S·X·S 是一个复杂的组合，理论上应该等价于 H 门")
    print("这可能是因为 S 和 X 门的特殊对称性和相位关系")

def performance_test():
    """性能测试"""
    print("\n" + "="*60)
    print("性能测试")
    print("=" * 60)

    n_qubits = 5
    n_tests = 1000

    # 创建测试状态
    test_state = create_state_vector(n_qubits)

    import time

    # 测试 S·X·S·X·S
    start_time = time.time()
    for _ in range(n_tests):
        result = apply_sxsxs_sequence(test_state.copy(), 0, n_qubits)
    sxsxs_time = time.time() - start_time

    # 测试 H 门
    start_time = time.time()
    for _ in range(n_tests):
        result = apply_h_gate(test_state.copy(), 0, n_qubits)
    h_time = time.time() - start_time

    print(f"S·X·S·X·S 序列 ({n_tests} 次): {sxsxs_time:.4f} 秒")
    print(f"H 门 ({n_tests} 次): {h_time:.4f} 秒")
    print(".2f")

if __name__ == "__main__":
    compare_gates()
    theoretical_analysis()
    performance_test()
