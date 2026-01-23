"""
测试 polarALL_state_3.py 的三状态极坐标量子计算
"""

from polarALL_state_3 import process_sequence_polar, create_initial_vec_like_v2, polar_to_complex_tensor
from utils import random_sequence
import torch

def main():
    print("polarALL_state_3.py 三状态极坐标量子计算测试")
    print("=" * 50)

    # 创建测试向量
    initial_vec = create_initial_vec_like_v2(32)
    seq = random_sequence(3, 2, 1, n_qubits=5)[:3]

    print(f"初始向量长度: {len(initial_vec)}")
    print(f"量子门序列: {seq}")

    # 执行三状态计算
    final_vec, final_scale_vec, state_history = process_sequence_polar(initial_vec, seq, verbose=False)

    print(f"✅ 计算完成")
    print(f"最终极坐标向量形状: {final_vec.shape}")
    print(f"最终缩放向量形状: {final_scale_vec.shape}")

    # 验证重构和归一化
    reconstructed = polar_to_complex_tensor(final_vec, final_scale_vec)
    norm = torch.sqrt(torch.sum(torch.abs(reconstructed)**2)).item()
    print(f"最终归一化: {norm:.6f}")

    if abs(norm - 1.0) < 1e-3:
        print("✅ 测试通过：三状态版本工作正常")
    else:
        print("❌ 测试失败：归一化误差过大")

if __name__ == "__main__":
    main()