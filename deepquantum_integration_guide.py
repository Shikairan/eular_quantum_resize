#!/usr/bin/env python3
"""
DeepQuantum 与 create_initial_vec_complex 集成指南

本文件展示了如何将您现有的 create_initial_vec_complex 函数
与 DeepQuantum 库正确集成使用。
"""

import sys
import os
import numpy as np
import torch

# 添加deepquantum到路径
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'deepquantum/src'))

try:
    import deepquantum as dq
    DEEPQUANTUM_AVAILABLE = True
except ImportError:
    DEEPQUANTUM_AVAILABLE = False

# 导入我们的工具函数
from utils import create_initial_vec_complex

def create_deepquantum_compatible_state(n_amps: int, n_qubits: int) -> torch.Tensor:
    """
    创建与 DeepQuantum 兼容的初始状态向量

    Args:
        n_amps: 向量长度（必须是2的幂）
        n_qubits: 量子比特数量

    Returns:
        torch.Tensor: DeepQuantum兼容的初始状态张量
    """
    # 验证参数
    assert 2 ** n_qubits == n_amps, f"参数不匹配：2^{n_qubits} != {n_amps}"

    # 创建初始向量（numpy数组）
    initial_vec = create_initial_vec_complex(n_amps)

    # 转换为torch张量并确保正确的dtype
    initial_tensor = torch.from_numpy(initial_vec.astype(np.complex64))

    return initial_tensor

def example_deepquantum_circuit_with_custom_initial_state():
    """
    使用自定义初始状态创建DeepQuantum电路的完整示例
    """
    if not DEEPQUANTUM_AVAILABLE:
        print("DeepQuantum不可用，请确保已正确安装")
        return

    print("=== DeepQuantum 电路示例：使用自定义初始状态 ===")

    # 参数设置
    n_qubits = 3
    n_amps = 2 ** n_qubits  # 8

    # 1. 创建兼容的初始状态
    print("1. 创建初始状态...")
    initial_state = create_deepquantum_compatible_state(n_amps, n_qubits)
    print(f"   初始状态形状: {initial_state.shape}")
    print(f"   初始状态类型: {initial_state.dtype}")
    print(f"   归一化检查: {torch.norm(initial_state).item():.6f}")

    # 2. 创建DeepQuantum电路
    print("\n2. 创建DeepQuantum电路...")
    circuit = dq.QubitCircuit(nqubit=n_qubits, init_state=initial_state)
    print("   ✅ 电路创建成功")

    # 3. 添加量子门
    print("\n3. 添加量子门...")
    circuit.h(wires=0)                    # Hadamard门
    circuit.x(wires=1)                    # Pauli-X门
    circuit.cx(control=0, target=2)       # CNOT门
    circuit.ry(wires=1, inputs=np.pi/4)   # 旋转门
    print("   ✅ 门添加成功")

    # 4. 执行电路
    print("\n4. 执行电路...")
    final_state = circuit()
    print(f"   最终状态形状: {final_state.shape}")
    final_norm = torch.norm(final_state).item()
    print(f"   最终状态归一化: {final_norm:.6f}")
    # 5. 可选：添加观测算符
    print("\n5. 添加观测算符...")
    # 在第0和第2比特上添加Z⊗Z观测
    circuit.observable([0, 2])
    expectation_value = circuit.expectation()
    print(f"   <Z⊗Z>期望值: {expectation_value.item():.6f}")

    print("\n✅ 示例完成！")
    return circuit

def compare_with_polar_complex_implementations():
    """
    与您现有的 polarALL_state_3.py 和 complexALL.py 进行比较
    """
    print("\n=== 与现有实现的比较 ===")

    from polarALL_state_3 import process_sequence_polar
    from complexALL import process_sequence_complex
    from utils import random_sequence

    n_qubits = 3
    n_amps = 2 ** n_qubits

    # 创建相同的初始状态
    initial_vec = create_initial_vec_complex(n_amps)
    initial_tensor = torch.from_numpy(initial_vec.astype(np.complex64))

    # 创建简单的测试序列
    seq = [
        ('H', '', (), 0),
        ('X', '', (), 1),
        ('CX', '', (), 0, 2)
    ]

    print("初始状态类型比较：")
    print(f"   complexALL: {type(initial_vec)} {initial_vec.shape}")
    print(f"   polarALL_state_3: {type(initial_tensor)} {initial_tensor.shape}")
    print(f"   DeepQuantum: {type(initial_tensor)} {initial_tensor.shape}")

    if DEEPQUANTUM_AVAILABLE:
        # 测试DeepQuantum
        circuit = dq.QubitCircuit(nqubit=n_qubits, init_state=initial_tensor)
        circuit.h(0)
        circuit.x(1)
        circuit.cx(0, 2)
        dq_final = circuit().flatten()

        # 测试complexALL
        complex_final, _ = process_sequence_complex(seq, initial_vec.copy(), verbose=False)

        # 测试polarALL_state_3
        polar_final, polar_scale, _ = process_sequence_polar(initial_vec.copy(), seq, verbose=False)
        # 将极坐标转换回复数
        from polarALL_state_3 import polar_to_complex_tensor
        polar_complex_tensor = polar_to_complex_tensor(polar_final, polar_scale)
        polar_complex = polar_complex_tensor.cpu().numpy().flatten()

        print("\n最终状态比较：")
        print(f"   DeepQuantum norm: {torch.norm(dq_final).item():.6f}")
        print(f"   complexALL norm: {np.linalg.norm(complex_final):.6f}")
        print(f"   polarALL_state_3 norm: {np.linalg.norm(polar_complex):.6f}")

        # 计算差异
        dq_numpy = dq_final.cpu().numpy()
        diff_dq_complex = np.linalg.norm(dq_numpy - complex_final)
        diff_dq_polar = np.linalg.norm(dq_numpy - polar_complex)

        print("\n互操作性检查：")
        print(".2e")
        print(".2e")
def deepquantum_sequence_adapter_demo():
    """
    演示如何将您的序列格式适配到DeepQuantum
    """
    if not DEEPQUANTUM_AVAILABLE:
        return

    print("\n=== 序列格式适配演示 ===")

    from utils import random_sequence

    # 创建您的序列格式
    seq = random_sequence(5, 2, 1, n_qubits=3, shuffle=False)
    print("您的序列格式示例：")
    for i, gate in enumerate(seq[:3]):  # 只显示前3个
        print(f"   {gate}")

    # 适配器函数
    def convert_to_deepquantum(circuit, seq):
        """将序列转换为DeepQuantum门调用"""
        gate_map = {
            'X': lambda c, w: c.x(wires=w),
            'Y': lambda c, w: c.y(wires=w),
            'Z': lambda c, w: c.z(wires=w),
            'H': lambda c, w: c.h(wires=w),
            'S': lambda c, w: c.s(wires=w),
            'T': lambda c, w: c.t(wires=w),
            'Rx': lambda c, w, p: c.rx(wires=w, inputs=p[0]),
            'Ry': lambda c, w, p: c.ry(wires=w, inputs=p[0]),
            'Rz': lambda c, w, p: c.rz(wires=w, inputs=p[0]),
            'CX': lambda c, ctrl, targ: c.cx(control=ctrl, target=targ),
            'CY': lambda c, ctrl, targ: c.cy(control=ctrl, target=targ),
            'CZ': lambda c, ctrl, targ: c.cz(control=ctrl, target=targ),
        }

        for gate_tuple in seq:
            if len(gate_tuple) == 4:  # 单比特门
                gate_name, _, params, wires = gate_tuple
                if gate_name in gate_map:
                    if params:
                        gate_map[gate_name](circuit, wires, params)
                    else:
                        gate_map[gate_name](circuit, wires)
            elif len(gate_tuple) == 5:  # 控制门
                gate_name, _, params, control, target = gate_tuple
                if gate_name in gate_map:
                    gate_map[gate_name](circuit, control, target)

    # 创建电路并应用适配器
    n_qubits = 3
    initial_state = create_deepquantum_compatible_state(2**n_qubits, n_qubits)
    circuit = dq.QubitCircuit(nqubit=n_qubits, init_state=initial_state)

    convert_to_deepquantum(circuit, seq)

    print("✅ 序列成功转换为DeepQuantum电路")
    print(f"   电路深度: {circuit.depth}")
    print(f"   门数量: {len(circuit.operators)}")

if __name__ == "__main__":
    print("DeepQuantum 与 create_initial_vec_complex 集成指南")
    print("=" * 60)

    if DEEPQUANTUM_AVAILABLE:
        # 运行所有示例
        example_deepquantum_circuit_with_custom_initial_state()
        compare_with_polar_complex_implementations()
        deepquantum_sequence_adapter_demo()

        print("\n🎉 所有集成示例完成！")
    else:
        print("❌ DeepQuantum 未安装，无法运行示例")
        print("\n安装说明：")
        print("1. 确保 deepquantum 在当前目录")
        print("2. 或使用: pip install deepquantum")
        print("\n兼容性说明：")
        print("- create_initial_vec_complex() 返回 numpy 数组")
        print("- DeepQuantum 接受 torch.Tensor")
        print("- 需要手动转换: torch.from_numpy(arr.astype(np.complex64))")