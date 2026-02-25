#!/usr/bin/env python3
"""
测试 vector.py 中不同的精度配置
"""

import torch
import numpy as np
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from vector import PolarVector

def test_precision_configs():
    """测试不同的精度配置"""
    print("=== 测试不同精度配置 ===")

    configs = [
        ("int16 (高精度)", torch.int16),
        ("int8 (内存优化)", torch.int8),
    ]

    for config_name, amplitude_dtype in configs:
        print(f"\n--- 测试 {config_name} ---")

        # 创建向量实例
        vector = PolarVector(amplitude_dtype=amplitude_dtype)
        print(f"配置信息: {vector.get_info()}")

        # 创建随机状态
        polar_vec, scale_vec = vector.create_random_state(4)
        print(f"随机状态形状: {polar_vec.shape}")
        print(f"数据类型: 幅度={polar_vec[:, 0].dtype}, 相位={polar_vec[:, 1].dtype}")
        print(f"编码范围: 幅度=[{polar_vec[:, 0].min()}, {polar_vec[:, 0].max()}], 相位=[{polar_vec[:, 1].min()}, {polar_vec[:, 1].max()}]")

        # 测试字符串表示
        print(f"状态向量: {vector.to_string(polar_vec, scale_vec)}")

        # 测试验证（传入幅值与相位 dtype）
        is_valid = PolarVector.validate_vector(polar_vec, scale_vec, amplitude_dtype, vector.phase_dtype)
        print(f"向量验证: {'通过' if is_valid else '失败'}")

    print()

def test_conversion_accuracy():
    """测试不同精度的转换精度"""
    print("=== 测试转换精度 (不同配置) ===")

    configs = [
        ("int16", torch.int16),
        ("int8", torch.int8),
    ]

    # 创建测试复数向量
    test_complex = torch.tensor([
        1.0 + 0.0j,      # 实数
        0.0 + 1.0j,      # 纯虚数
        0.707 + 0.707j,  # 45度
        -0.5 + 0.866j    # 120度
    ], dtype=torch.complex64)

    print(f"原始复数向量: {test_complex}")

    for config_name, amplitude_dtype in configs:
        print(f"\n--- {config_name} 精度 ---")

        vector = PolarVector(amplitude_dtype=amplitude_dtype, device=test_complex.device)

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

def test_memory_usage():
    """测试不同精度的内存使用情况"""
    print("=== 测试内存使用情况 ===")

    n_qubits = 10  # 1024 个幅度
    n_states = 2**n_qubits

    configs = [
        ("int16", torch.int16, 2),  # int16 + int8 = 3 bytes per amplitude
        ("int8", torch.int8, 1),    # int8 + int8 = 2 bytes per amplitude
    ]

    for config_name, amplitude_dtype, bytes_per_amplitude in configs:
        vector = PolarVector(amplitude_dtype=amplitude_dtype)

        # 创建状态向量
        polar_vec, scale_vec = vector.create_random_state(n_states)

        # 计算内存使用
        polar_memory = polar_vec.numel() * polar_vec.element_size()  # bytes
        scale_memory = scale_vec.numel() * scale_vec.element_size()  # bytes (float32)
        total_memory = polar_memory + scale_memory

        print(f"{config_name}:")
        print(f"  向量形状: {polar_vec.shape}")
        print(f"  极坐标内存: {polar_memory / 1024:.1f} KB")
        print(f"  缩放因子内存: {scale_memory / 1024:.1f} KB")
        print(f"  总内存: {total_memory / 1024:.1f} KB")
        print(f"  对比传统 float32+float32: {n_states * 8 / 1024:.1f} KB")
        print(".1f")

    print()

def test_backward_compatibility():
    """测试向后兼容性"""
    print("=== 测试向后兼容性 ===")

    # 默认配置 (int16) 应该与之前的行为相同
    vector_default = PolarVector()
    vector_explicit = PolarVector(amplitude_dtype=torch.int16)

    # 创建相同的随机种子以确保相同结果
    torch.manual_seed(42)
    polar1, scale1 = vector_default.create_random_state(4)

    torch.manual_seed(42)
    polar2, scale2 = vector_explicit.create_random_state(4)

    # 应该完全相同
    are_equal = torch.allclose(polar1.float(), polar2.float()) and torch.allclose(scale1, scale2)
    print(f"默认配置与显式 int16 配置结果相同: {are_equal}")

    print()

if __name__ == "__main__":
    print("测试 PolarVector 不同精度配置")
    print("=" * 50)

    test_precision_configs()
    test_conversion_accuracy()
    test_memory_usage()
    test_backward_compatibility()

    print("✅ 所有精度测试完成！")