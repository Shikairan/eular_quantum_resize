"""
演示 complexALL.py 的使用方法
展示如何记录每一步的状态变化
"""

import random
import math
from complexALL import (
    create_initial_vec_like_v2,
    random_sequence,
    process_sequence_complex,
    compute_normalization_error,
    print_state_vector
)

def demo_sequence_processing():
    """演示序列处理功能"""
    print("complexALL.py 功能演示")
    print("=" * 50)

    # 设置参数
    N_AMP = 8  # 3 比特系统，便于观察
    n_qubits = int(math.log2(N_AMP))

    print(f"使用 {n_qubits} 比特系统，状态向量长度为 {N_AMP}")

    # 创建初始向量（与 ALLeularV2.py 第903行完全相同）
    initial_vec = create_initial_vec_like_v2(N_AMP)
    print(f"初始向量创建完成，未归一化范数: {math.sqrt(sum(abs(z)**2 for z in initial_vec)):.6f}")

    # 创建一个简单的测试序列
    test_seq = [
        ('X', '', []),           # X 门
        ('H', '', []),           # H 门
        ('CX', '', []),          # CNOT 门
        ('Ry', '(1.571)', [1.571]),  # Ry(π/2) 门
    ]

    print(f"\n测试序列: {[gate[0] + gate[1] for gate in test_seq]}")

    # 处理序列，记录每一步的状态
    final_vec, state_history = process_sequence_complex(test_seq, initial_vec, verbose=False)

    print(f"\n记录了 {len(state_history)} 步状态变化")

    # 显示每一步的状态变化
    for step, state in enumerate(state_history):
        if step == 0:
            gate_name = "初始状态"
        else:
            gate_info = test_seq[step-1]
            gate_name = gate_info[0] + gate_info[1]

        print(f"\n步骤 {step}: {gate_name}")
        norm_error = compute_normalization_error(state)
        print(".2e")
        print_state_vector(state, title="")

    print("\n最终结果:")
    print(f"总共 {len(test_seq)} 个门")
    print(f"最终归一化误差: {compute_normalization_error(final_vec):.2e}")

def demo_comparison_with_v2():
    """演示如何与 ALLeularV2.py 进行对照"""
    print("\n" + "="*50)
    print("与 ALLeularV2.py 对照演示")
    print("="*50)

    # 使用相同的随机种子确保一致性
    random.seed(42)

    N_AMP = 16  # 4 比特系统
    n_qubits = int(math.log2(N_AMP))

    # 创建相同的初始向量
    initial_vec = create_initial_vec_like_v2(N_AMP)

    # 生成相同的随机序列
    seq = random_sequence(10, 10, 10, shuffle=True)

    print(f"系统: {n_qubits} 比特")
    print(f"序列长度: {len(seq)} 门")
    print(f"初始向量范数: {math.sqrt(sum(abs(z)**2 for z in initial_vec)):.6f}")

    # 处理序列
    final_vec, state_history = process_sequence_complex(seq, initial_vec, verbose=False)

    print("\n处理完成:")
    print(f"状态历史记录: {len(state_history)} 步")
    print(f"最终归一化误差: {compute_normalization_error(final_vec):.2e}")

    # 显示一些关键信息
    print("\n序列中的前5个门:")
    for i, (name, param_str, _) in enumerate(seq[:5]):
        print(f"  {i+1}. {name}{param_str}")

if __name__ == "__main__":
    demo_sequence_processing()
    demo_comparison_with_v2()
