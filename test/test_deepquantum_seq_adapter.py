#!/usr/bin/env python3
"""
测试 DeepQuantum 接受自定义序列格式的完整适配器

这个测试代码展示了如何：
1. 使用 create_initial_vec_complex(n_amps) 作为初始向量
2. 使用 random_sequence() 生成的序列作为线路构建工具
3. 将序列中的门顺序转换为 DeepQuantum 的门调用
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
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils import create_initial_vec_complex, random_sequence
from complexALL import process_sequence_complex

class DeepQuantumSequenceAdapter:
    """
    DeepQuantum 序列适配器
    将自定义序列格式转换为 DeepQuantum 门调用
    """

    def __init__(self):
        # 门映射：自定义格式 -> DeepQuantum 方法
        self.gate_map = {
            # 无参数单比特门
            'X': lambda circuit, wires: circuit.x(wires=wires),
            'Y': lambda circuit, wires: circuit.y(wires=wires),
            'Z': lambda circuit, wires: circuit.z(wires=wires),
            'H': lambda circuit, wires: circuit.h(wires=wires),
            'S': lambda circuit, wires: circuit.s(wires=wires),
            'T': lambda circuit, wires: circuit.t(wires=wires),

            # 旋转门（单参数）
            'Rx': lambda circuit, wires, params: circuit.rx(wires=wires, inputs=params[0]),
            'Ry': lambda circuit, wires, params: circuit.ry(wires=wires, inputs=params[0]),
            'Rz': lambda circuit, wires, params: circuit.rz(wires=wires, inputs=params[0]),
            'U1': lambda circuit, wires, params: circuit.u1(wires=wires, inputs=params[0]),

            # U2 门（两参数）
            'U2': lambda circuit, wires, params: circuit.u2(wires=wires, inputs=params[0], inputs2=params[1]),

            # U3 门（三参数）
            'U3': lambda circuit, wires, params: circuit.u3(wires=wires, inputs=params[0], inputs2=params[1], inputs3=params[2]),

            # 无参数控制门
            'CX': lambda circuit, control, target: circuit.cx(control=control, target=target),
            'CY': lambda circuit, control, target: circuit.cy(control=control, target=target),
            'CZ': lambda circuit, control, target: circuit.cz(control=control, target=target),
            'CH': lambda circuit, control, target: circuit.ch(control=control, target=target),
            'CS': lambda circuit, control, target: circuit.cs(control=control, target=target),
            'CT': lambda circuit, control, target: circuit.ct(control=control, target=target),

            # 有参数控制门
            'CRx': lambda circuit, control, target, params: circuit.crx(control=control, target=target, inputs=params[0]),
            'CRy': lambda circuit, control, target, params: circuit.cry(control=control, target=target, inputs=params[0]),
            'CRz': lambda circuit, control, target, params: circuit.crz(control=control, target=target, inputs=params[0]),
            'CU1': lambda circuit, control, target, params: circuit.cu1(control=control, target=target, inputs=params[0]),
            'CU2': lambda circuit, control, target, params: circuit.cu2(control=control, target=target, inputs=params[0], inputs2=params[1]),
            'CU3': lambda circuit, control, target, params: circuit.cu3(control=control, target=target, inputs=params[0], inputs2=params[1], inputs3=params[2]),
        }

    def apply_gate_to_circuit(self, circuit, gate_tuple):
        """
        将单个门元组应用到 DeepQuantum 电路

        Args:
            circuit: DeepQuantum QubitCircuit 实例
            gate_tuple: 门元组 (gate_name, param_str, params, qubit_info...)
        """
        if len(gate_tuple) < 4:
            raise ValueError(f"无效的门格式: {gate_tuple}")

        gate_name = gate_tuple[0]
        params = gate_tuple[2]

        if len(gate_tuple) == 4:
            # 单比特门: (gate_name, param_str, params, qubit_idx)
            qubit_idx = gate_tuple[3]

            if gate_name in self.gate_map:
                if params:
                    self.gate_map[gate_name](circuit, qubit_idx, params)
                else:
                    self.gate_map[gate_name](circuit, qubit_idx)
            else:
                raise ValueError(f"不支持的单比特门: {gate_name}")

        elif len(gate_tuple) == 5:
            # 控制门: (gate_name, param_str, params, control_idx, target_idx)
            control_idx = gate_tuple[3]
            target_idx = gate_tuple[4]

            if gate_name in self.gate_map:
                if params:
                    self.gate_map[gate_name](circuit, control_idx, target_idx, params)
                else:
                    self.gate_map[gate_name](circuit, control_idx, target_idx)
            else:
                raise ValueError(f"不支持的控制门: {gate_name}")
        else:
            raise ValueError(f"无效的门格式长度: {len(gate_tuple)}")

    def apply_sequence_to_circuit(self, circuit, sequence, verbose=False):
        """
        将整个序列应用到 DeepQuantum 电路

        Args:
            circuit: DeepQuantum QubitCircuit 实例
            sequence: 门序列列表
            verbose: 是否打印详细信息
        """
        if verbose:
            print(f"应用序列到电路，共 {len(sequence)} 个门")

        for i, gate_tuple in enumerate(sequence):
            if verbose and (i % 10 == 0 or i == len(sequence) - 1):
                print(f"  应用门 {i+1}/{len(sequence)}: {gate_tuple[0]}")

            self.apply_gate_to_circuit(circuit, gate_tuple)

        if verbose:
            print("序列应用完成")

def test_deepquantum_sequence_adapter():
    """
    测试 DeepQuantum 序列适配器的完整功能
    """
    if not DEEPQUANTUM_AVAILABLE:
        print(" DeepQuantum 未安装，无法运行测试")
        return

    print(" DeepQuantum 序列适配器测试")
    print("=" * 60)

    # 参数设置
    n_qubits = 4
    n_amps = 2 ** n_qubits

    # 1. 创建初始向量
    print("\n1. 创建初始向量...")
    initial_vec = create_initial_vec_complex(n_amps)
    initial_tensor = torch.from_numpy(initial_vec.astype(np.complex64))
    print(f"   初始向量创建完成，形状: {initial_tensor.shape}")

    # 2. 生成随机序列
    print("\n2. 生成随机序列...")
    seq = random_sequence(30, 0, 0, n_qubits=n_qubits, shuffle=False)
    print(f"   序列生成完成，共 {len(seq)} 个门")
    print("   前5个门示例:")
    for i, gate in enumerate(seq[:5]):
        print(f"     {i+1}. {gate}")

    # 3. 创建 DeepQuantum 电路
    print("\n3. 创建 DeepQuantum 电路...")
    circuit = dq.QubitCircuit(nqubit=n_qubits, init_state=initial_tensor)
    print("    电路创建完成")

    # 4. 创建适配器并应用序列
    print("\n4. 应用序列到电路...")
    adapter = DeepQuantumSequenceAdapter()
    adapter.apply_sequence_to_circuit(circuit, seq, verbose=True)

    # 5. 执行电路
    print("\n5. 执行电路...")
    final_state = circuit()
    final_norm = torch.norm(final_state).item()
    print(f"   电路执行完成")
    print(f"   最终状态形状: {final_state.shape}")

    # 6. 验证与现有实现的兼容性
    print("\n6. 与现有实现比较...")
    from complexALL import process_sequence_complex

    # 使用相同的序列运行 complexALL
    complex_result, _ = process_sequence_complex(seq, initial_vec.copy(), verbose=False)
    complex_norm = np.linalg.norm(complex_result)

    # 计算差异
    dq_numpy = final_state.cpu().numpy().flatten()
    diff = np.linalg.norm(dq_numpy - complex_result)
    relative_error = diff / max(np.linalg.norm(dq_numpy), np.linalg.norm(complex_result))

    # 7. 测试序列解析的准确性
    print("\n7. 序列解析验证...")
    test_gates = [
        ('X', '', [], 0),                    # X门
        ('H', '', [], 1),                    # H门
        ('Rx', '(1.570)', [1.57], 2),       # 参数化Rx门
        ('CX', '', [], 0, 1),               # CNOT门
        ('CRy', '(0.785)', [0.785], 1, 2),  # 控制Ry门
    ]

    print("   测试门解析:")
    for gate in test_gates:
        try:
            # 创建临时电路进行测试
            test_circuit = dq.QubitCircuit(nqubit=n_qubits)
            adapter.apply_gate_to_circuit(test_circuit, gate)
            print(f"    {gate[0]} 门解析成功")
        except Exception as e:
            print(f"   {gate[0]} 门解析失败: {e}")

    print("\n DeepQuantum 序列适配器测试完成！")

    return {
        'success': relative_error < 1e-10,
        'final_norm': final_norm,
        'relative_error': relative_error,
        'n_gates': len(seq)
    }

def demo_sequence_adapter_usage():
    """
    演示序列适配器的使用方法
    """
    if not DEEPQUANTUM_AVAILABLE:
        return

    print("\n DeepQuantum 序列适配器使用演示")
    print("=" * 50)

    # 示例：创建一个简单的量子电路
    n_qubits = 3
    n_amps = 2 ** n_qubits

    # 创建初始状态 |000⟩
    initial_vec = np.zeros(n_amps, dtype=complex)
    initial_vec[0] = 1.0  # |000⟩状态
    initial_tensor = torch.from_numpy(initial_vec.astype(np.complex64))

    # 定义一个简单的序列
    custom_sequence = [
        ('H', '', [], 0),           # 对第0个比特应用H门
        ('X', '', [], 1),           # 对第1个比特应用X门
        ('Rx', '(π/2)', [np.pi/2], 2),  # 对第2个比特应用Rx(π/2)
        ('CX', '', [], 0, 1),       # CNOT门：控制0，目标1
        ('CRy', '(π/4)', [np.pi/4], 1, 2),  # 控制Ry：控制1，目标2
    ]

    print("自定义序列:")
    for i, gate in enumerate(custom_sequence, 1):
        gate_name = gate[0]
        if len(gate) == 4:
            qubit = gate[3]
            print(f"  {i}. {gate_name} 门 -> 第{qubit}个比特")
        elif len(gate) == 5:
            control, target = gate[3], gate[4]
            print(f"  {i}. {gate_name} 门 -> 控制第{control}个比特，目标第{target}个比特")

    # 使用适配器创建电路
    circuit = dq.QubitCircuit(nqubit=n_qubits, init_state=initial_tensor)
    adapter = DeepQuantumSequenceAdapter()
    adapter.apply_sequence_to_circuit(circuit, custom_sequence, verbose=True)

    # 执行并显示结果
    result = circuit()
    print("\n电路执行结果:")
    print(f"  最终状态归一化: {torch.norm(result).item():.6f}")
    print(f"  状态向量形状: {result.shape}")

    print("\n 演示完成！")

if __name__ == "__main__":
    # 运行完整测试
    test_result = test_deepquantum_sequence_adapter()

    # 如果测试成功，运行演示
    if test_result and test_result.get('success', False):
        demo_sequence_adapter_usage()
    else:
        print("\n  测试未完全通过，可能存在兼容性问题")
