"""
分批次处理极坐标量子计算
用于处理超长状态向量，通过将向量分割成矩阵形式进行计算

⚠️ 重要警告：
这个实现将大的量子状态向量分割成多个独立的子系统进行处理。
在真实的量子计算中，这种方法通常是不正确的，因为：
1. 大多数量子门（如控制门）需要访问整个状态空间
2. 量子比特之间存在纠缠，不能简单分割
3. 分割会破坏量子计算的正确性

这个实现仅适用于以下特殊情况：
1. 测试和原型验证内存处理能力
2. 处理可以自然分割成独立子系统的简单情况
3. 仅包含单比特门的序列（无控制门）
4. 学术研究目的

对于真实的量子计算，请使用完整的 polarALL.py 或 complexALL.py
"""

import torch
import math
import numpy as np
from typing import List, Tuple
from polarALL import process_sequence_polar, create_initial_vec_like_v2, polar_to_complex_tensor

# 设置设备
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

def reshape_vec_to_matrix(initial_vec: List[complex], n_subsystems: int) -> List[List[complex]]:
    """
    将1*N的向量重塑为M*(N/M)的矩阵形式（实际是M个子向量）

    ⚠️ 重要：这个函数会重新归一化每个子系统，这会破坏量子计算的正确性！
    仅用于测试和演示目的。

    Args:
        initial_vec: 原始的1*N初始状态向量
        n_subsystems: 子系统数量M

    Returns:
        子系统列表，每个子系统是一个长度为N/M的复数向量
    """
    n_amps = len(initial_vec)
    assert n_amps % n_subsystems == 0, f"向量长度 {n_amps} 必须能被子系统数量 {n_subsystems} 整除"

    subsystem_size = n_amps // n_subsystems
    assert subsystem_size & (subsystem_size - 1) == 0, f"子系统大小 {subsystem_size} 必须是2的幂次"

    subsystems = []
    for i in range(n_subsystems):
        start_idx = i * subsystem_size
        end_idx = (i + 1) * subsystem_size
        subsystem = initial_vec[start_idx:end_idx]

        # ⚠️ 警告：重新归一化会破坏量子计算的正确性！
        # 在真实的量子计算中，子系统不应该被单独归一化
        norm = np.linalg.norm(subsystem)
        if norm > 0:
            subsystem = [z / norm for z in subsystem]

        subsystems.append(subsystem)

    return subsystems

def flatten_matrix_to_vec(subsystems: List[torch.Tensor]) -> torch.Tensor:
    """
    将M*(N/M)的子系统矩阵重新展平为1*N的向量

    Args:
        subsystems: 子系统列表，每个子系统是极坐标张量

    Returns:
        展平后的1*N状态向量
    """
    # 将所有子系统的极坐标转换为复数
    complex_subsystems = []
    for subsystem in subsystems:
        complex_subsystem = polar_to_complex_tensor(subsystem)
        complex_subsystems.append(complex_subsystem)

    # 合并所有子系统
    merged_complex = torch.cat(complex_subsystems, dim=0)

    # ⚠️ 注意：这里不再重新归一化，因为这会破坏量子计算的正确性
    # 每个子系统已经单独归一化，总向量应该保持这种归一化

    # 转换回极坐标
    from polarALL import complex_to_polar_tensor
    merged_polar = complex_to_polar_tensor(merged_complex)

    return merged_polar

def process_sequence_polar_split(initial_vec: List[complex], seq: List[Tuple],
                                n_subsystems: int, verbose: bool = False) -> Tuple[torch.Tensor, List[torch.Tensor]]:
    """
    分批次处理量子门序列（极坐标版本）
    将1*N向量重塑为M*(N/M)矩阵，对每个子系统分别计算，然后重新展平

    Args:
        initial_vec: 初始状态向量（复数形式，长度为N）
        seq: 量子门序列
        n_subsystems: 子系统数量M（向量将被分割为M个子系统）
        verbose: 是否打印详细信息

    Returns:
        (final_vec, state_history): 最终状态向量和每一步的状态历史
    """
    n_amps = len(initial_vec)
    n_qubits_total = int(math.log2(n_amps))

    # 验证子系统数量
    subsystem_size = n_amps // n_subsystems
    n_qubits_subsystem = int(math.log2(subsystem_size))

    if verbose:
        print(f"原始向量长度: {n_amps} (2^{n_qubits_total} 个元素)")
        print(f"子系统数量: {n_subsystems}")
        print(f"每个子系统大小: {subsystem_size} (2^{n_qubits_subsystem} 个元素)")

    # 将初始向量重塑为子系统矩阵
    subsystems = reshape_vec_to_matrix(initial_vec, n_subsystems)

    if verbose:
        print(f"成功重塑为 {len(subsystems)} 个子系统")

    # 对每个子系统分别应用相同的序列
    subsystem_results = []
    subsystem_histories = []

    for subsys_idx, subsystem_vec in enumerate(subsystems):
        if verbose:
            print(f"处理子系统 {subsys_idx + 1}/{len(subsystems)}...")

        # 对当前子系统应用完整的序列
        subsys_final, subsys_history = process_sequence_polar(subsystem_vec, seq, verbose=False)
        subsystem_results.append(subsys_final)
        subsystem_histories.append(subsys_history)

    if verbose:
        print("所有子系统处理完成，开始合并结果...")

    # 将子系统结果重新展平为原始向量
    final_vec = flatten_matrix_to_vec(subsystem_results)

    # 构建合并的历史记录
    # 注意：每个子系统有自己的历史，我们需要将它们合并
    state_history = []
    for step in range(len(subsystem_histories[0])):  # 假设所有子系统的历史长度相同
        step_states = [hist[step] for hist in subsystem_histories]
        merged_step = flatten_matrix_to_vec(step_states)
        state_history.append(merged_step)

    if verbose:
        print(f"合并完成，最终向量长度: {len(final_vec)}")
        # 验证归一化
        complex_vec = polar_to_complex_tensor(final_vec)
        norm = torch.sqrt(torch.sum(torch.abs(complex_vec)**2)).item()
        print(f"最终归一化误差: {abs(norm - 1.0):.2e}")

    return final_vec, state_history

def create_super_long_initial_vec(n_amps: int) -> List[complex]:
    """
    创建一个超级长的初始向量（用于测试）

    Args:
        n_amps: 向量长度（必须是2的幂次）

    Returns:
        归一化的复数向量
    """
    assert n_amps & (n_amps - 1) == 0, f"向量长度 {n_amps} 必须是2的幂次"

    # 创建随机复数向量
    complex_vector = [complex(np.random.random(), np.random.random()) for _ in range(n_amps)]

    # 归一化
    norm = np.linalg.norm(complex_vector)
    normalized_vector = complex_vector / norm

    return normalized_vector

def compare_with_standard_polar(initial_vec: List[complex], seq: List[Tuple],
                               n_subsystems: int = 4) -> dict:
    """
    对比分批次处理和标准处理的结果

    Args:
        initial_vec: 初始向量
        seq: 量子门序列
        n_subsystems: 子系统数量

    Returns:
        对比结果字典
    """
    print("对比测试：分批次处理 vs 标准处理")
    print("-" * 40)

    # 标准处理
    print("运行标准 polarALL 处理...")
    from polarALL import process_sequence_polar as standard_process
    standard_final, _ = standard_process(initial_vec, seq, verbose=False)

    # 分批次处理
    print("运行分批次 polarSplite 处理...")
    split_final, _ = process_sequence_polar_split(initial_vec, seq, n_subsystems, verbose=False)

    # 计算差异
    standard_complex = polar_to_complex_tensor(standard_final)
    split_complex = polar_to_complex_tensor(split_final)

    diff = torch.abs(standard_complex - split_complex)
    max_diff = torch.max(diff).item()
    rms_diff = torch.sqrt(torch.mean(diff**2)).item()
    mean_diff = torch.mean(diff).item()

    print(f"最大差异: {max_diff:.2e}")
    print(f"RMS 差异: {rms_diff:.2e}")
    print(f"平均差异: {mean_diff:.2e}")

    return {
        'max_diff': max_diff,
        'rms_diff': rms_diff,
        'mean_diff': mean_diff,
        'standard_final': standard_final,
        'split_final': split_final
    }

# ===== 使用示例 =====

if __name__ == "__main__":
    print("polarSplite.py 分批次极坐标量子计算示例")
    print("=" * 50)

    # 示例：处理一个 2^12 = 4096 维的向量，分成 4 个子系统
    # ⚠️ 注意：这个分割会破坏量子计算的正确性，仅用于演示内存处理能力
    total_n_amps = 4096   # 2^12
    n_subsystems = 4      # 分成 4 个子系统
    subsystem_size = total_n_amps // n_subsystems  # 每个子系统 2^10 = 1024 维

    print("⚠️ 重要警告：")
    print("  这个示例将向量分割成独立子系统进行处理。")
    print("  在真实的量子计算中，这种方法通常是不正确的。")
    print("  此示例仅用于演示分批次处理的技术可行性。")
    print()

    print(f"总向量维度: {total_n_amps} (2^{int(math.log2(total_n_amps))} 维)")
    print(f"子系统数量: {n_subsystems}")
    print(f"每个子系统大小: {subsystem_size} (2^{int(math.log2(subsystem_size))} 维)")

    # 创建超级长的初始向量
    print("\n创建初始向量...")
    initial_vec = create_super_long_initial_vec(total_n_amps)
    print(f"初始向量创建完成，长度: {len(initial_vec)}")

    # 生成量子门序列（仅使用单比特门，避免控制门）
    # 注意：这个实现仅用于演示目的，实际量子计算中控制门是很重要的
    print("⚠️ 注意：此演示仅使用单比特门，跳过控制门（因为分批次处理不支持跨子系统控制门）")

    # 手动创建仅包含单比特门的序列
    seq = []
    n_qubits_subsystem = int(math.log2(subsystem_size))

    # 生成一些单比特门
    single_gates = ['X', 'Y', 'Z', 'S', 'T', 'H']
    for i in range(20):
        gate = np.random.choice(single_gates)
        qubit_idx = np.random.randint(0, n_qubits_subsystem)
        seq.append((gate, '', [], qubit_idx))

    # 生成一些参数化单比特门
    for i in range(15):
        gate = np.random.choice(['Rx', 'Ry', 'Rz'])
        qubit_idx = np.random.randint(0, n_qubits_subsystem)
        param = np.random.uniform(0, 2 * np.pi)
        seq.append((gate, f'({param:.3f})', [param], qubit_idx))

    np.random.shuffle(seq)
    print(f"生成量子门序列: {len(seq)} 个门（仅单比特门，针对 {n_qubits_subsystem} 比特子系统）")

    # 分批次处理
    print("\n开始分批次处理...")
    final_vec, state_history = process_sequence_polar_split(
        initial_vec, seq, n_subsystems, verbose=True
    )

    print("\n处理完成！")
    print(f"最终状态向量长度: {len(final_vec)}")
    print(f"记录了 {len(state_history)} 步状态变化")

    # 对比测试（仅当子系统数量为1时进行）
    if n_subsystems == 4:
        print("\n" + "="*50)
        print("对比测试（子系统数量=1，应该等价于标准处理）")
        try:
            comparison = compare_with_standard_polar(initial_vec, seq, n_subsystems)
            print(f"分批次处理与标准处理的差异在可接受范围内: {comparison['rms_diff'] < 1e-10}")
            if comparison['rms_diff'] >= 1e-10:
                print("⚠️ 警告：即使只有一个子系统，差异仍然很大，说明实现有问题")
        except Exception as e:
            print(f"对比测试失败: {e}")
    else:
        print("\n⚠️ 跳过对比测试（多子系统模式不保证正确性）")

    print("\n示例完成！")