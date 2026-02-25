#!/usr/bin/env python3
"""
测试 vector.py 模块的功能
"""

import torch
import numpy as np
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from vector import PolarVector

def test_basic_functionality():
    """测试基本功能"""
    print("=== 测试 PolarVector 基本功能 ===")

    # 创建向量实例
    vector = PolarVector()
    print(f"向量类信息: {vector.get_info()}")

    # 创建随机状态
    polar_vec, scale_vec = vector.create_random_state(4)
    print(f"随机状态形状: {polar_vec.shape}")
    print(f"数据类型: 幅度={polar_vec[:, 0].dtype}, 相位={polar_vec[:, 1].dtype}")
    print(f"编码范围: 幅度=[{polar_vec[:, 0].min()}, {polar_vec[:, 0].max()}], 相位=[{polar_vec[:, 1].min()}, {polar_vec[:, 1].max()}]")

    # 测试字符串表示
    print(f"状态向量: {vector.to_string(polar_vec, scale_vec)}")

    # 测试转换
    complex_vec = vector.to_complex(polar_vec, scale_vec)
    print(f"转换为复数: {complex_vec}")

    # 测试验证
    is_valid = PolarVector.validate_vector(polar_vec, scale_vec)
    print(f"向量验证: {'通过' if is_valid else '失败'}")

    print()

def test_conversion_accuracy():
    """测试转换精度"""
    print("=== 测试转换精度 ===")

    vector = PolarVector()

    # 创建测试复数向量（确保在正确的设备上）
    test_complex = torch.tensor([
        1.0 + 0.0j,      # 实数
        0.0 + 1.0j,      # 纯虚数
        0.707 + 0.707j,  # 45度
        -0.5 + 0.866j    # 120度
    ], dtype=torch.complex64, device=vector.device)

    print(f"原始复数向量: {test_complex}")

    # 转换为极坐标
    polar_vec, scale_vec = vector.complex_to_polar_tensor(test_complex)
    print(f"极坐标编码: 幅度={polar_vec[:, 0]}, 相位={polar_vec[:, 1]}")
    print(f"缩放因子: {scale_vec}")

    # 转换回复数
    reconstructed = vector.to_complex(polar_vec, scale_vec)
    print(f"重构复数向量: {reconstructed}")

    # 计算误差
    error = torch.abs(test_complex - reconstructed).max().item()
    print(".2e")

    # 计算相对误差
    relative_error = torch.abs((test_complex - reconstructed) / test_complex).max().item()
    print(".2e")

    print()

def test_from_complex():
    """测试从复数创建"""
    print("=== 测试从复数创建向量 ===")

    vector = PolarVector()

    # 创建 |00⟩ + |11⟩ 贝尔态
    bell_state = torch.tensor([
        0.7071067811865476 + 0.0j,  # |00⟩
        0.0 + 0.0j,                # |01⟩
        0.0 + 0.0j,                # |10⟩
        0.7071067811865476 + 0.0j  # |11⟩
    ], dtype=torch.complex64, device=vector.device)

    polar_vec, scale_vec = vector.create_from_complex(bell_state)
    print(f"贝尔态极坐标: {vector.to_string(polar_vec, scale_vec)}")

    print()

def test_backward_compatibility():
    """测试向后兼容性"""
    print("=== 测试向后兼容性函数 ===")

    from vector import create_random_polar_state, polar_vec_to_string, polar_to_complex_tensor, complex_to_polar_tensor

    # 测试兼容性函数
    polar_vec, scale_vec = create_random_polar_state(4)
    print(f"兼容性函数创建的向量: {polar_vec_to_string(polar_vec, scale_vec)}")

    # 测试转换函数
    vector = PolarVector()
    complex_vec = torch.tensor([1.0, 0.0, 0.0, 0.0], dtype=torch.complex64, device=vector.device)
    polar_vec, scale_vec = complex_to_polar_tensor(complex_vec)
    back_to_complex = polar_to_complex_tensor(polar_vec, scale_vec)
    print(f"转换循环测试: {torch.allclose(complex_vec, back_to_complex)}")

    print()

if __name__ == "__main__":
    print("测试 PolarVector 类")
    print("=" * 50)

    test_basic_functionality()
    test_conversion_accuracy()
    test_from_complex()
    test_backward_compatibility()

    print("✅ 所有测试完成！")