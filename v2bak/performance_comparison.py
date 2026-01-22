"""
比较 ALLeularV4.py 优化前后的性能
"""

import torch
import time
import math
from ALLeularV4 import *

def benchmark_gate_application(n_qubits=5, n_gates=1000, gate_name='H'):
    """
    基准测试门应用性能
    """
    print(f"=== 性能测试：{n_qubits} 比特系统，{n_gates} 个 {gate_name} 门 ===")

    # 创建状态向量
    n_amps = 2 ** n_qubits
    polar_vec = create_random_polar_state(n_amps)

    # 选择一个随机的比特索引
    qubit_idx = torch.randint(0, n_qubits, (1,)).item()

    # 预热
    for _ in range(10):
        apply_gate_unified_tensor(polar_vec, gate_name, qubit_idx=qubit_idx)

    # 正式测试
    torch.cuda.synchronize()  # 确保 GPU 操作完成
    start_time = time.time()

    for _ in range(n_gates):
        apply_gate_unified_tensor(polar_vec, gate_name, qubit_idx=qubit_idx)

    torch.cuda.synchronize()  # 确保所有操作完成
    end_time = time.time()

    total_time = end_time - start_time
    avg_time_per_gate = total_time / n_gates * 1000  # 毫秒

    print(f"总时间: {total_time:.4f} 秒")
    print(f"平均时间: {avg_time_per_gate:.2f} ms/门")

    # 检查归一化
    complex_vec = polar_to_complex_tensor(polar_vec.float())
    norm = torch.sqrt(torch.sum(torch.abs(complex_vec)**2)).item()
    print(f"归一化检查: {norm:.6f}")
    return total_time, avg_time_per_gate

def test_different_system_sizes():
    """测试不同系统大小的性能"""
    print("=== 不同系统大小的性能测试 ===")

    sizes = [4, 5, 6, 7]  # 比特数
    results = []

    for n_qubits in sizes:
        try:
            total_time, avg_time = benchmark_gate_application(
                n_qubits=n_qubits,
                n_gates=500,  # 减少测试门数以适应大系统
                gate_name='H'
            )
            results.append((n_qubits, total_time, avg_time))
            print()
        except Exception as e:
            print(f"  {n_qubits} 比特系统测试失败: {e}")
            print()

    print("=== 总结 ===")
    print("比特数 | 总时间(s) | 平均时间(ms/门)")
    print("-" * 35)
    for n_qubits, total_time, avg_time in results:
        print("6.4f")

def test_different_gate_types():
    """测试不同门类型的性能"""
    print("=== 不同门类型的性能测试 ===")

    gates = ['X', 'Y', 'Z', 'H', 'S', 'T']
    n_qubits = 5
    n_gates = 1000

    print(f"在 {n_qubits} 比特系统上测试 {n_gates} 个门")
    print()

    for gate_name in gates:
        try:
            total_time, avg_time = benchmark_gate_application(
                n_qubits=n_qubits,
                n_gates=n_gates,
                gate_name=gate_name
            )
            print()
        except Exception as e:
            print(f"  {gate_name} 门测试失败: {e}")
            print()

if __name__ == "__main__":
    print("ALLeularV4.py 性能测试")
    print(f"使用设备: {device}")
    print()

    # 测试不同系统大小
    test_different_system_sizes()
    print()

    # 测试不同门类型
    test_different_gate_types()
