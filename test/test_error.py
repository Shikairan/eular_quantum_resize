"""
量子计算误差对比测试
比较 polarALL.py 和 complexALL.py 在每个门操作后的数据向量误差
"""

import math
import copy
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from polarALL import process_sequence_polar, polar_to_complex_tensor
from complexALL import process_sequence_complex
from utils import random_sequence_tensor, create_initial_vec_complex
import numpy as np
import torch

def compute_state_difference(complex_state, polar_state):
    """计算两个状态向量之间的差异"""
    # 将极坐标转换为复数
    #polar_complex = []
    polar_complex = polar_to_complex_tensor(polar_state)
    norm = torch.norm(polar_complex)
    polar_complex = polar_complex / norm

    # 计算差异
    differences = []
    for i in range(len(complex_state)):
        diff = abs(complex_state[i] - polar_complex[i])
        differences.append(diff)

    max_diff = max(differences)
    rms_diff = math.sqrt(sum(d**2 for d in differences) / len(differences))
    mean_diff = sum(differences) / len(differences)

    return max_diff, rms_diff, mean_diff

def test_error(n_amps, seq_length, seed=42):
    """测试两个函数在每个门操作后的数据向量误差"""
    import torch
    torch.manual_seed(seed)
    import random
    random.seed(seed)
    import numpy as np
    np.random.seed(seed)

    # 生成随机序列
    n_qubits = int(math.log2(n_amps))
    seq = random_sequence_tensor(seq_length, 0, 0, n_qubits=n_qubits, shuffle=False)
    print(seq)
    # 创建相同的初始向量
    initial_vec = create_initial_vec_complex(n_amps)
    initial_vec2 = copy.deepcopy(initial_vec)
    # 调用两个函数
    complex_final, complex_history = process_sequence_complex(seq, initial_vec, verbose=False)
    
    polar_final, polar_history = process_sequence_polar(initial_vec2, seq, verbose=False)
    for i in polar_history:
        print()
    # 比较每个门之后的误差
    print(f"序列长度: {len(seq)} 个门")
    for step in range(len(complex_history)):
        max_diff, rms_diff, mean_diff = compute_state_difference(complex_history[step], polar_history[step])
        if step == 0:
            gate_str = "初始状态"
        else:
            gate_info = seq[step-1]
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
        print(f"门 {step} ({gate_str}): max={max_diff:.2e}, rms={rms_diff:.2e}, mean={mean_diff:.2e}")

if __name__ == "__main__":
    test_error(n_amps=32, seq_length=30, seed=42)
