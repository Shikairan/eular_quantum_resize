#!/usr/bin/env python3
"""
测试 DeepQuantum 接受 create_initial_vec_complex(n_amps) 作为初始向量的兼容性
"""

import sys
import os
import numpy as np
import torch

# 添加deepquantum到路径（假设它在当前目录）
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'deepquantum/src'))

try:
    import deepquantum as dq
    DEEPQUANTUM_AVAILABLE = True
except ImportError:
    print("警告：deepquantum 未安装，将模拟其行为")
    DEEPQUANTUM_AVAILABLE = False

# 导入我们的工具函数
from utils import create_initial_vec_complex

def test_deepquantum_initial_state(n_amps=8, n_qubits=3):
    """
    测试 DeepQuantum 是否能正确接受 create_initial_vec_complex 的输出作为初始状态

    Args:
        n_amps: 向量长度（必须是2的幂）
        n_qubits: 量子比特数量
    """
    print(f"测试 DeepQuantum 初始状态兼容性")
    print(f"n_amps={n_amps}, n_qubits={n_qubits}")
    print("=" * 60)

    # 验证参数一致性
    assert 2 ** n_qubits == n_amps, f"参数不匹配：2^{n_qubits} != {n_amps}"

    # 创建初始向量
    print("\n1. 创建初始向量...")
    initial_vec = create_initial_vec_complex(n_amps)
    print(f"   向量类型: {type(initial_vec)}")
    print(f"   向量形状: {initial_vec.shape}")
    print(f"   向量长度: {len(initial_vec)}")
    print(f"   向量类型: {type(initial_vec[0])}")
    print(f"   第一个元素: {initial_vec[0]}")
    print(f"   最后5个元素: {initial_vec[-5:]}")

    # 检查归一化
    norm = np.linalg.norm(initial_vec)

    if DEEPQUANTUM_AVAILABLE:
        print("\n2. 测试 DeepQuantum QubitState 初始化...")

        try:
            # 将numpy数组转换为torch张量
            initial_tensor = torch.from_numpy(initial_vec.astype(np.complex64))

            # 创建 DeepQuantum 量子态
            qstate = dq.QubitState(nqubit=n_qubits, state=initial_tensor)
            print("   ✅ QubitState 创建成功")

            # 检查状态向量
            state_tensor = qstate.state
            print(f"   状态张量形状: {state_tensor.shape}")
            print(f"   状态张量类型: {state_tensor.dtype}")
            print(f"   状态张量设备: {state_tensor.device}")

            # 验证状态是否正确
            # 由于DeepQuantum可能进行内部处理，我们比较数值
            original_as_tensor = np.array(initial_vec, dtype=np.complex64)
            dq_state_as_numpy = state_tensor.detach().cpu().numpy().flatten()

            # 计算差异
            diff = np.abs(original_as_tensor - dq_state_as_numpy)
            max_diff = np.max(diff)
            rms_diff = np.sqrt(np.mean(diff**2))

            # 检查归一化是否保持
            dq_norm = np.linalg.norm(dq_state_as_numpy)

            if max_diff < 1e-6 and abs(dq_norm - 1.0) < 1e-10:
                print("   ✅ 状态向量正确加载")
                return True
            else:
                print("   ❌ 状态向量加载有误")
                return False

        except Exception as e:
            print(f"   ❌ QubitState 创建失败: {e}")
            return False

    else:
        print("\n2. 模拟 DeepQuantum 行为...")

        # 模拟 DeepQuantum 的行为
        try:
            # 转换为torch张量（模拟DeepQuantum的处理）
            if not isinstance(initial_vec, np.ndarray):
                state_array = np.array(initial_vec, dtype=np.complex64)
            else:
                state_array = initial_vec.astype(np.complex64)

            # 模拟amplitude_encoding（如果需要的话）
            # 这里我们假设向量已经是正确的幅度编码格式

            print(f"   模拟状态数组形状: {state_array.shape}")
            print(f"   模拟状态数组类型: {state_array.dtype}")

            # 检查是否是有效的量子状态
            if abs(np.linalg.norm(state_array) - 1.0) < 1e-10:
                print("   ✅ 向量是有效的归一化量子状态")
                return True
            else:
                print("   ❌ 向量不是有效的归一化量子状态")
                return False

        except Exception as e:
            print(f"   ❌ 模拟处理失败: {e}")
            return False

def test_different_sizes():
    """测试不同大小的向量"""
    test_cases = [
        (4, 2),   # 2 qubits
        (8, 3),   # 3 qubits
        (16, 4),  # 4 qubits
        (32, 5),  # 5 qubits
    ]

    print("\n测试不同大小的向量...")
    print("-" * 40)

    all_passed = True
    for n_amps, n_qubits in test_cases:
        print(f"\n测试 n_amps={n_amps}, n_qubits={n_qubits}")
        passed = test_deepquantum_initial_state(n_amps, n_qubits)
        if not passed:
            all_passed = False

    return all_passed

def create_deepquantum_circuit_example(n_amps=8, n_qubits=3):
    """创建使用我们初始向量的DeepQuantum电路示例"""
    if not DEEPQUANTUM_AVAILABLE:
        print("\nDeepQuantum不可用，跳过电路示例")
        return

    print("\n创建 DeepQuantum 电路示例...")
    print("-" * 40)

    # 创建初始向量
    initial_vec = create_initial_vec_complex(n_amps)

    # 将numpy数组转换为torch张量
    initial_tensor = torch.from_numpy(initial_vec.astype(np.complex64))

    # 创建电路
    cir = dq.QubitCircuit(nqubit=n_qubits, init_state=initial_tensor)
    print("   ✅ 电路创建成功")

    # 添加一些门
    cir.h(wires=0)
    cir.x(wires=1)
    cir.cx(control=0, target=2)
    print("   ✅ 门添加成功")

    # 执行电路
    final_state = cir()
    print(f"   最终状态形状: {final_state.shape}")
    print("   ✅ 电路执行成功")

    # 检查归一化
    norm = torch.norm(final_state)

    return cir

if __name__ == "__main__":
    print("DeepQuantum 初始向量兼容性测试")
    print("=" * 60)

    # 测试基本功能
    success = test_deepquantum_initial_state()

    if success:
        print("\n🎉 基本测试通过！")

        # 测试不同大小
        all_sizes_success = test_different_sizes()

        if all_sizes_success:
            print("\n🎉 所有大小测试通过！")

            # 创建电路示例
            create_deepquantum_circuit_example()
        else:
            print("\n⚠️  某些大小测试失败")
    else:
        print("\n❌ 基本测试失败")

    print("\n测试完成")
