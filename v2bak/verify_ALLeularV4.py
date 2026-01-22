"""
验证 ALLeularV4.py (PyTorch 版本) 与 ALLeularV2.py (NumPy 版本) 的等价性
"""

import torch
import numpy as np
import math
import random
from ALLeularV4 import *
from ALLeularV2 import *

def compare_states(pytorch_vec, numpy_vec, tolerance=1e-5):
    """比较 PyTorch 和 NumPy 版本的状态向量"""
    # 将 PyTorch 极坐标转换为复数
    pytorch_complex = polar_to_complex_tensor(pytorch_vec).cpu().numpy()

    # 计算差异
    diff = np.abs(pytorch_complex - numpy_vec)
    max_diff = np.max(diff)
    mean_diff = np.mean(diff)

    print(f"最大差异: {max_diff:.2e}")
    print(f"平均差异: {mean_diff:.2e}")
    print(f"差异在容差范围内: {max_diff < tolerance}")

    return max_diff < tolerance

def test_gate_equivalence():
    """测试门操作的等价性"""
    print("=== 门等价性测试 ===")

    # 创建相同的随机种子
    seed = 42
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)

    # 4 比特系统
    N_AMP = 16
    nqubit = 4

    # 创建相同的初始状态
    real_parts = np.random.randn(N_AMP)
    imag_parts = np.random.randn(N_AMP)
    complex_vec = real_parts + 1j * imag_parts
    complex_vec = complex_vec / np.linalg.norm(complex_vec)

    # NumPy 版本
    numpy_byte_vec = [(encode_r(abs(z)), encode_th(math.atan2(z.imag, z.real))) for z in complex_vec]

    # PyTorch 版本
    pytorch_polar_vec = torch.tensor([[encode_r(abs(z)), encode_th(math.atan2(z.imag, z.real))] for z in complex_vec],
                                    dtype=torch.float32, device=device)

    print("初始状态比较:")
    compare_states(pytorch_polar_vec, complex_vec)

    # 测试一系列门操作
    test_sequence = [
        ('H', [], 0),
        ('X', [], 1),
        ('Ry', [math.pi/4], 2),
        ('CX', [], 0, 1),
        ('CZ', [], 2, 3),
        ('U3', [math.pi/3, math.pi/6, math.pi/2], 1),
    ]

    print("\n=== 门序列测试 ===")
    for gate_name, params, *qubits in test_sequence:
        print(f"\n应用 {gate_name} {qubits}:")

        # 复制状态向量
        numpy_vec_copy = numpy_byte_vec.copy()
        pytorch_vec_copy = pytorch_polar_vec.clone()

        # 应用门
        if len(qubits) == 1:
            # NumPy 版本
            apply_gate_unified(numpy_vec_copy, None, gate_name, *params, qubit_idx=qubits[0])

            # PyTorch 版本
            apply_gate_unified_tensor(pytorch_vec_copy, gate_name, *params, qubit_idx=qubits[0])
        else:
            # NumPy 版本
            apply_gate_unified(numpy_vec_copy, None, gate_name, *params, control_idx=qubits[0], target_idx=qubits[1])

            # PyTorch 版本
            apply_gate_unified_tensor(pytorch_vec_copy, gate_name, *params, control_idx=qubits[0], target_idx=qubits[1])

        # 比较结果
        numpy_complex = np.array([byte2complex(z) for z in numpy_vec_copy])
        compare_states(pytorch_vec_copy, numpy_complex)

if __name__ == "__main__":
    print("ALLeularV4.py 与 ALLeularV2.py 等价性验证")
    test_gate_equivalence()
    print("\n验证完成！")
