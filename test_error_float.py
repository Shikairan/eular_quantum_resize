"""
量子计算误差对比测试 - float32 版本
比较 polarFloat.py 和 complexALL.py 在每个门操作后的数据向量误差
"""

import math
import copy
from polarFloat import process_sequence_polar_float, polar_to_complex_tensor_float, create_initial_vec_like_v2 as create_initial_vec_polar_float
from complexALL import process_sequence_complex, create_initial_vec_like_v2 as create_initial_vec_complex
import numpy as np
import torch

def compute_state_difference(complex_state, polar_state):
    """计算两个状态向量之间的差异"""
    # polar_state 已经是复数形式（从 polarFloat.py 来的）
    polar_complex = polar_state

    # 计算差异
    differences = []
    for i in range(len(complex_state)):
        diff = abs(complex_state[i] - polar_complex[i])
        differences.append(diff)

    max_diff = max(differences)
    rms_diff = math.sqrt(sum(d**2 for d in differences) / len(differences))
    mean_diff = sum(differences) / len(differences)

    return max_diff, rms_diff, mean_diff

def test_error_float(n_amps, seq_length, seed=42):
    """测试 polarFloat.py 和 complexALL.py 在每个门操作后的数据向量误差"""
    import torch
    torch.manual_seed(seed)
    import random
    random.seed(seed)
    import numpy as np
    np.random.seed(seed)

    # 生成随机序列（仅使用单比特门，避免控制门问题）
    n_qubits = int(math.log2(n_amps))

    # 手动生成只包含单比特门的序列
    seq = []
    single_gates = ['X', 'Y', 'Z', 'S', 'T']
    single_param_gates = ['Rz']

    for i in range(seq_length):
        if i % 2 == 0:
            # 无参数门
            gate = np.random.choice(single_gates)
            qubit_idx = np.random.randint(0, n_qubits)
            seq.append((gate, '', [], qubit_idx))
        else:
            # 有参数门
            gate = np.random.choice(single_param_gates)
            param = np.random.uniform(0, 2 * math.pi)
            qubit_idx = np.random.randint(0, n_qubits)
            seq.append((gate, f'({param:.3f})', [param], qubit_idx))
    print(f"生成的量子门序列: {len(seq)} 个门")
    print(f"前10个门: {[f'{name}({param_str})' if param_str else name for name, param_str, _, qubit_idx in seq[:10]]}")

    # 创建相同的初始向量
    initial_vec = create_initial_vec_complex(n_amps)
    initial_vec2 = copy.deepcopy(initial_vec)

    print(f"初始向量创建完成，范数: {np.linalg.norm(initial_vec):.6f}")

    # 调用两个函数
    print("\n运行 complexALL.py (纯复数版本)...")
    complex_final, complex_history = process_sequence_complex(seq, initial_vec, verbose=False)

    print("运行 polarFloat.py (float32 极坐标版本)...")
    polar_final, polar_history = process_sequence_polar_float(initial_vec2, seq, verbose=False)

    # 将 polar 历史转换为复数形式
    polar_complex_history = []
    for polar_state in polar_history:
        complex_state = polar_to_complex_tensor_float(polar_state)
        polar_complex_history.append(complex_state.cpu().numpy())

    # 比较最终结果的归一化误差
    complex_norm_error = abs(np.linalg.norm(complex_final) - 1.0)
    polar_norm_error = abs(np.linalg.norm(polar_complex_history[-1]) - 1.0)

    print(f"\ncomplexALL 最终归一化误差: {complex_norm_error:.2e}")
    print(f"polarFloat 最终归一化误差: {polar_norm_error:.2e}")

    # 比较每个门之后的误差
    print(f"\n==================== 每个门之后的误差 ====================")
    for step in range(len(complex_history)):
        max_diff, rms_diff, mean_diff = compute_state_difference(complex_history[step], polar_complex_history[step])
        gate_info = seq[step-1] if step > 0 else "初始状态"
        if step > 0:
            gate_name, param_str, _, qubit_idx = gate_info
            gate_str = f"{gate_name}(比特={qubit_idx})"
            if param_str:
                gate_str += f"({param_str})"
        else:
            gate_str = "初始状态"
        print(f"门 {step} ({gate_str}): max={max_diff:.2e}, rms={rms_diff:.2e}, mean={mean_diff:.2e}")

def compare_implementations_float(precision="float32", n_amps=32, seq_length=50, seed=42):
    """比较不同实现的误差"""
    print(f"==================== 测试精度: {precision} ====================")
    print("量子计算实现对比测试")
    print("=" * 50)
    print(f"状态向量长度: {n_amps}")
    print(f"序列参数: {seq_length}")
    print(f"随机种子: {seed}")
    print(f"polarFloat 精度: {precision}")

    test_error_float(n_amps=n_amps, seq_length=seq_length, seed=seed)

if __name__ == "__main__":
    # 测试 float32 版本
    compare_implementations_float(precision="float32", n_amps=32, seq_length=30, seed=42)