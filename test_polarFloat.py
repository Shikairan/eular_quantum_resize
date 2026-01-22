"""
测试 polarFloat.py 的功能
"""

from polarFloat import process_sequence_polar_float, create_initial_vec_like_v2, polar_to_complex_tensor_float
from polarALL import random_sequence_tensor
import torch
import numpy as np

def test_polar_float():
    """测试 polarFloat.py 的基本功能"""
    print("测试 polarFloat.py")
    print("=" * 40)

    # 创建测试向量
    n_amps = 64  # 2^6 = 64
    initial_vec = create_initial_vec_like_v2(n_amps)
    print(f"创建初始向量，长度: {len(initial_vec)}")

    # 生成测试序列（仅使用单比特门，避免复杂的控制门）
    seq = []
    n_qubits = 6
    for i in range(20):
        gate = np.random.choice(['X', 'Y', 'Z',])
        qubit_idx = np.random.randint(0, n_qubits)
        seq.append((gate, '', [], qubit_idx))

    print(f"测试序列示例: {seq[:3]}...")

    print(f"生成测试序列: {len(seq)} 个单比特门")

    # 使用 polarFloat.py 处理
    print("\n使用 polarFloat.py (float32) 处理...")
    final_vec, state_history = process_sequence_polar_float(initial_vec, seq, verbose=False)

    # 验证结果
    complex_final = polar_to_complex_tensor_float(final_vec)
    norm = torch.sqrt(torch.sum(torch.abs(complex_final)**2)).item()
    norm_error = abs(norm - 1.0)

    print("处理完成！")
    print(f"最终向量长度: {len(final_vec)}")
    print(f"状态历史长度: {len(state_history)}")
    print(f"归一化误差: {norm_error:.2e}")

    if norm_error < 1e-6:
        print("✅ 测试通过：归一化保持良好")
    else:
        print("❌ 测试失败：归一化误差过大")

    return norm_error < 1e-6

def compare_precision():
    """比较 int8 和 float32 版本的精度"""
    print("\n比较 int8 vs float32 精度")
    print("=" * 40)

    # 创建相同的输入
    n_amps = 32
    initial_vec = create_initial_vec_like_v2(n_amps)

    # 生成简单的测试序列（仅单比特门）
    seq = [('H', '', [], 0), ('X', '', [], 1), ('Y', '', [], 2), ('Z', '', [], 3)]

    print(f"测试向量长度: {len(initial_vec)}")
    print(f"测试序列: {seq}")

    # int8 版本
    print("\n运行 polarALL.py (int8)...")
    from polarALL import process_sequence_polar, polar_to_complex_tensor
    final_int8, _ = process_sequence_polar(initial_vec, seq, verbose=False)
    complex_int8 = polar_to_complex_tensor(final_int8)

    # float32 版本
    print("运行 polarFloat.py (float32)...")
    final_float, _ = process_sequence_polar_float(initial_vec, seq, verbose=False)
    complex_float = polar_to_complex_tensor_float(final_float)

    # 计算差异
    diff = torch.abs(complex_int8 - complex_float)
    max_diff = torch.max(diff).item()
    rms_diff = torch.sqrt(torch.mean(diff**2)).item()

    print("\n精度对比:")
    print(f"最大差异: {max_diff:.2e}")
    print(f"RMS 差异: {rms_diff:.2e}")

    # 检查归一化
    norm_int8 = torch.sqrt(torch.sum(torch.abs(complex_int8)**2)).item()
    norm_float = torch.sqrt(torch.sum(torch.abs(complex_float)**2)).item()

    print("\n归一化检查:")
    print(f"int8 版本归一化误差: {abs(norm_int8 - 1.0):.2e}")
    print(f"float32 版本归一化误差: {abs(norm_float - 1.0):.2e}")

    return max_diff, rms_diff

if __name__ == "__main__":
    # 运行测试
    success = test_polar_float()
    max_diff, rms_diff = compare_precision()

    print("\n" + "=" * 40)
    print("测试总结:")
    print(f"基本功能测试: {'通过' if success else '失败'}")
    print(f"最大精度差异: {max_diff:.2e}")
    print(f"RMS 精度差异: {rms_diff:.2e}")

    if max_diff < 0.1:
        print("✅ 两个版本结果基本一致")
    else:
        print("⚠️ 两个版本结果差异较大，可能存在实现问题")
