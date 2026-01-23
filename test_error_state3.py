"""
量子计算误差对比测试 - 三状态版本
比较 polarALL_state_3.py 和 complexALL.py 在每个门操作后的数据向量误差
"""

import math
import copy
from polarALL_state_3 import process_sequence_polar, polar_to_complex_tensor
from complexALL import process_sequence_complex
from utils import random_sequence, create_initial_vec_complex
import numpy as np
import torch

def compute_state_difference(complex_state, polar_state):
    """计算两个状态向量之间的差异"""
    # polar_state 已经是复数形式（从 polarALL_state_3.py 来的）
    polar_complex = polar_state

    # 计算差异
    differences = []
    for i in range(len(complex_state)):
        diff = abs(complex_state[i] - polar_complex[i])
        differences.append(diff)

    max_diff = max(differences)
    rms_diff = math.sqrt(sum(d**2 for d in differences) / len(differences))
    mean_diff = sum(differences) / len(differences)

    error_percent = sum(np.array(differences)/(abs(complex_state)+1e-4)) / len(differences)
    return max_diff, rms_diff, mean_diff, float(error_percent)

def test_error_state3(n_amps, seq_length, seed=42):
    """测试 polarALL_state_3.py 和 complexALL.py 在每个门操作后的数据向量误差"""
    import torch
    torch.manual_seed(seed)
    import random
    random.seed(seed)
    import numpy as np
    np.random.seed(seed)

    # 生成随机序列（仅使用单比特门，避免控制门问题）
    n_qubits = int(math.log2(n_amps))

    # 手动生成只包含单比特门的序列
    seq = random_sequence(30, 0, 0, n_qubits=n_qubits, shuffle=False)
    # 创建相同的初始向量
    initial_vec = create_initial_vec_complex(n_amps)
    initial_vec2 = copy.deepcopy(initial_vec)

    print(f"初始向量创建完成，范数: {np.linalg.norm(initial_vec):.6f}")

    # 调用两个函数
    print("\n运行 complexALL.py (纯复数版本)...")
    complex_final, complex_history = process_sequence_complex(seq, initial_vec, verbose=False)

    print("运行 polarALL_state_3.py (三状态极坐标版本)...")
    polar_final, polar_scale_final, polar_history = process_sequence_polar(initial_vec2, seq, verbose=False)

    # 将 polar 历史转换为复数形式
    polar_complex_history = []
    for polar_state, scale_state in polar_history:
        complex_state = polar_to_complex_tensor(polar_state, scale_state)
        polar_complex_history.append(complex_state.cpu().numpy())

    # 比较最终结果的归一化误差
    complex_norm_error = abs(np.linalg.norm(complex_final) - 1.0)
    polar_norm_error = abs(np.linalg.norm(polar_complex_history[-1]) - 1.0)

    print(f"\ncomplexALL 最终归一化误差: {complex_norm_error:.2e}")
    print(f"polarALL_state_3 最终归一化误差: {polar_norm_error:.2e}")

    # 比较每个门之后的误差
    print(f"\n==================== 每个门之后的误差 ====================")
    for step in range(len(complex_history)):
        max_diff, rms_diff, mean_diff, error_percent = compute_state_difference(complex_history[step], polar_complex_history[step])
        gate_info = seq[step-1] if step > 0 else "初始状态"
        if step > 0:
            if len(gate_info) == 4:
                # 单比特门: (gate_name, param_str, params, qubit_idx)
                gate_name, param_str, _, qubit_idx = gate_info
                gate_str = f"{gate_name}(比特={qubit_idx})"
            elif len(gate_info) == 5:
                # 控制门: (gate_name, param_str, params, control_idx, target_idx)
                gate_name, param_str, _, control_idx, target_idx = gate_info
                gate_str = f"{gate_name}(控制={control_idx}, 目标={target_idx})"
            else:
                gate_str = f"{gate_name}(未知格式)"
            if param_str:
                gate_str += f"({param_str})"
        else:
            gate_str = "初始状态"
        print(f"门 {step} ({gate_str}): max={max_diff:.2e}, rms={rms_diff:.2e}, mean={mean_diff:.2e} errorP={error_percent:.2e}")

def compare_implementations_state3(precision="state3", n_amps=32, seq_length=50, seed=42):
    """比较不同实现的误差"""
    print(f"==================== 测试精度: {precision} ====================")
    print("量子计算实现对比测试")
    print("=" * 50)
    print(f"状态向量长度: {n_amps}")
    print(f"序列参数: {seq_length}")
    print(f"随机种子: {seed}")
    print(f"polarALL_state_3 精度: {precision}")

    test_error_state3(n_amps=n_amps, seq_length=seq_length, seed=seed)

if __name__ == "__main__":
    # 测试三状态版本
    compare_implementations_state3(precision="state3", n_amps=2**8, seq_length=30, seed=42)
