"""
简化测试 polarFloat.py 的功能
只测试单比特门，确保基本功能正常
"""

from polarFloat import process_sequence_polar_float, create_initial_vec_like_v2, polar_to_complex_tensor_float
import torch
import numpy as np

def test_single_qubit_gates():
    """测试单比特门"""
    print("测试单比特门")
    print("=" * 40)

    # 创建 2^5 = 32 维系统
    n_amps = 32
    initial_vec = create_initial_vec_like_v2(n_amps)

    # 创建一系列单比特门序列
    seq = []
    n_qubits = 5
    gates = ['X', 'Y', 'Z', 'H', 'S', 'T']

    for i in range(10):  # 10 个门
        gate = np.random.choice(gates)
        qubit_idx = np.random.randint(0, n_qubits)
        seq.append((gate, '', [], qubit_idx))

    print(f"测试向量长度: {len(initial_vec)}")
    print(f"序列长度: {len(seq)}")
    print(f"序列示例: {seq[:3]}")

    # 处理序列
    final_vec, state_history = process_sequence_polar_float(initial_vec, seq, verbose=False)

    # 验证最终归一化
    complex_final = polar_to_complex_tensor_float(final_vec)
    final_norm = torch.sqrt(torch.sum(torch.abs(complex_final)**2)).item()
    norm_error = abs(final_norm - 1.0)

    print("处理结果:")
    print(f"  最终向量长度: {len(final_vec)}")
    print(f"  状态历史长度: {len(state_history)}")
    print(".2e")
    if norm_error < 1e-6:
        print("  ✅ 归一化保持良好")
        return True
    else:
        print("  ❌ 归一化误差过大")
        return False

def test_comparison_with_complex():
    """与纯复数版本比较"""
    print("\n与 complexALL.py 比较")
    print("=" * 40)

    from complexALL import process_sequence_complex

    # 创建相同的输入
    n_amps = 16  # 使用较小的系统以加快测试
    initial_vec = create_initial_vec_like_v2(n_amps)

    # 创建单比特门序列
    seq = [('H', '', [], 0), ('X', '', [], 1), ('Y', '', [], 2), ('Z', '', [], 3)]

    print(f"测试向量长度: {len(initial_vec)}")
    print(f"测试序列: {seq}")

    # complexALL 处理
    complex_final, _ = process_sequence_complex(seq, initial_vec.copy(), verbose=False)

    # polarFloat 处理
    polar_final, _ = process_sequence_polar_float(initial_vec, seq, verbose=False)
    polar_complex = polar_to_complex_tensor_float(polar_final)

    # 比较结果
    diff = torch.abs(torch.tensor(complex_final, device=polar_complex.device) - polar_complex)
    max_diff = torch.max(diff).item()
    rms_diff = torch.sqrt(torch.mean(diff**2)).item()

    print("比较结果:")
    print(".2e")
    print(".2e")
    if max_diff < 1e-3:
        print("  ✅ 结果基本一致")
        return True
    else:
        print("  ❌ 结果差异较大")
        return False

if __name__ == "__main__":
    print("polarFloat.py 简化测试")
    print("=" * 60)

    test1_passed = test_single_qubit_gates()
    test2_passed = test_comparison_with_complex()

    print("\n" + "=" * 60)
    print("测试总结:")
    print(f"单比特门测试: {'通过' if test1_passed else '失败'}")
    print(f"对比测试: {'通过' if test2_passed else '失败'}")

    if test1_passed and test2_passed:
        print("🎉 所有测试通过！polarFloat.py 工作正常")
    else:
        print("⚠️ 部分测试失败，需要进一步调试")