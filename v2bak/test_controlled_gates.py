"""
专门测试控制门性能和正确性的脚本
"""

import torch
import time
from ALLeularV4 import *

def test_controlled_gate_performance():
    """测试控制门性能"""
    print("=== 控制门性能测试 ===")
    print(f"使用设备: {device}")
    print()

    # 测试不同大小的系统
    sizes = [3, 4, 5]  # 比特数

    for n_qubits in sizes:
        n_amps = 2 ** n_qubits
        polar_vec = create_random_polar_state(n_amps)

        print(f"{n_qubits} 比特系统 ({n_amps} 状态)")

        # 测试不同的控制门类型
        controlled_gates = ['CX', 'CY', 'CZ']

        for gate_name in controlled_gates:
            # 预热
            for _ in range(5):
                apply_gate_unified_tensor(polar_vec, gate_name, control_idx=0, target_idx=1)

            # 正式测试
            n_tests = 100 if n_qubits <= 4 else 50  # 较大系统减少测试次数

            torch.cuda.synchronize()
            start_time = time.time()

            for _ in range(n_tests):
                apply_gate_unified_tensor(polar_vec, gate_name, control_idx=0, target_idx=1)

            torch.cuda.synchronize()
            end_time = time.time()

            total_time = end_time - start_time
            avg_time = total_time / n_tests * 1000  # ms

            print("8s")
        print()

def test_controlled_gate_correctness():
    """测试控制门正确性"""
    print("=== 控制门正确性测试 ===")

    # 使用小的 3 比特系统进行精确测试
    n_qubits = 3
    n_amps = 2 ** n_qubits
    polar_vec = create_random_polar_state(n_amps)

    print("初始状态:")
    print(polar_vec_to_string(polar_vec.float(), range(8)))
    print()

    # 测试 CNOT 门
    print("应用 CNOT 门 (控制=0, 目标=1):")
    polar_vec_copy = polar_vec.clone()
    apply_gate_unified_tensor(polar_vec_copy, 'CX', control_idx=0, target_idx=1)
    print(polar_vec_to_string(polar_vec_copy.float(), range(8)))
    print()

    # 验证 CNOT 逻辑：
    # 当控制比特为 1 时，目标比特应该翻转
    # 状态 1 (001) -> 状态 3 (011)
    # 状态 3 (011) -> 状态 1 (001)
    # 状态 5 (101) -> 状态 7 (111)
    # 状态 7 (111) -> 状态 5 (101)

    print("CNOT 验证:")
    print("状态 1 (001) 和状态 3 (011) 应该交换")
    print("状态 5 (101) 和状态 7 (111) 应该交换")
    print("其他状态保持不变")
    print()

    # 测试 CZ 门
    print("应用 CZ 门 (控制=0, 目标=1):")
    polar_vec_copy2 = polar_vec.clone()
    apply_gate_unified_tensor(polar_vec_copy2, 'CZ', control_idx=0, target_idx=1)
    print(polar_vec_to_string(polar_vec_copy2.float(), range(8)))
    print()

    print("CZ 验证:")
    print("状态 3 (011) 的相位应该改变 π")
    print("状态 7 (111) 的相位应该改变 π")
    print("其他状态保持不变")

def benchmark_controlled_vs_single():
    """比较控制门和单比特门的性能"""
    print("=== 控制门 vs 单比特门性能比较 ===")

    n_qubits = 5
    n_amps = 2 ** n_qubits
    polar_vec = create_random_polar_state(n_amps)

    n_tests = 1000

    # 测试单比特门
    torch.cuda.synchronize()
    start_time = time.time()
    for _ in range(n_tests):
        apply_gate_unified_tensor(polar_vec, 'X', qubit_idx=0)
    torch.cuda.synchronize()
    single_time = time.time() - start_time

    # 测试控制门
    torch.cuda.synchronize()
    start_time = time.time()
    for _ in range(n_tests):
        apply_gate_unified_tensor(polar_vec, 'CX', control_idx=0, target_idx=1)
    torch.cuda.synchronize()
    controlled_time = time.time() - start_time

    print(f"单比特门 (X): {single_time:.4f} 秒")
    print(f"控制门 (CX):   {controlled_time:.4f} 秒")
    print(".2f")
    print()

    print("控制门较慢的原因:")
    print("- 需要处理的状态对更少 (2^(n-2) vs 2^(n-1))")
    print("- 但是索引计算更复杂")
    print("- 张量优化仍然显著提升性能")

if __name__ == "__main__":
    test_controlled_gate_performance()
    print("\n" + "="*50 + "\n")
    test_controlled_gate_correctness()
    print("\n" + "="*50 + "\n")
    benchmark_controlled_vs_single()
