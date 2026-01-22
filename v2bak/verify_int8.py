"""
验证 ALLeularV4.py 是否正确使用 int8 向量
"""

import torch
from ALLeularV4 import *

def test_int8_usage():
    """测试 int8 向量使用"""
    print("=== 测试 int8 向量使用 ===")

    # 创建状态向量
    polar_vec = create_random_polar_state(8)  # 3 比特系统

    print(f"状态向量形状: {polar_vec.shape}")
    print(f"状态向量数据类型: {polar_vec.dtype}")
    print(f"状态向量设备: {polar_vec.device}")

    # 检查是否为 int8
    assert polar_vec.dtype == torch.int8, f"期望 int8，但得到 {polar_vec.dtype}"

    # 检查值范围
    r_values = polar_vec[:, 0]
    th_values = polar_vec[:, 1]

    print(f"幅度编码值范围: {r_values.min().item()} 到 {r_values.max().item()}")
    print(f"相位编码值范围: {th_values.min().item()} 到 {th_values.max().item()}")

    # 验证值在合理范围内
    assert r_values.min() >= -128 and r_values.max() <= 127, "幅度编码超出 int8 范围"
    assert th_values.min() >= -128 and th_values.max() <= 127, "相位编码超出 int8 范围"

    # 测试一个门操作
    print("\n应用 H 门前:")
    print(f"数据类型: {polar_vec.dtype}")
    print(f"前4个元素: {polar_vec[:4]}")

    # 应用 H 门
    apply_gate_unified_tensor(polar_vec, 'H', qubit_idx=0)

    print("\n应用 H 门后:")
    print(f"数据类型: {polar_vec.dtype}")
    print(f"前4个元素: {polar_vec[:4]}")

    # 验证仍然是 int8
    assert polar_vec.dtype == torch.int8, f"门操作后数据类型改变为 {polar_vec.dtype}"

    print("\n✅ int8 向量测试通过！")

if __name__ == "__main__":
    test_int8_usage()
