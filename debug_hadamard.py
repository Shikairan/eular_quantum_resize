#!/usr/bin/env python3
"""
调试 Hadamard 门实现
"""

import torch
import math
import numpy as np
from polarALL_state_3 import polar_to_complex_tensor, complex_to_polar_tensor, hadamard_polar_tensor, H_MAT_TORCH, device

def debug_hadamard():
    print("=== Hadamard门调试 ===")

    # 测试 |0⟩ 状态的极坐标表示
    # |0⟩ = [1, 0] 在复数域，极坐标为 [(1, 0), (0, 0)]
    # 其中 (1, 0) 表示幅度1，相位0；(0, 0) 表示幅度0，相位任意

    # 创建 |0⟩ 状态的极坐标编码
    # 幅度编码：1 -> ?, 0 -> ?
    # 相位编码：0 -> ?, 0 -> ?

    r1 = torch.tensor([1.0], dtype=torch.float32, device=device)
    r2 = torch.tensor([0.0], dtype=torch.float32, device=device)
    th1 = torch.tensor([0.0], dtype=torch.float32, device=device)
    th2 = torch.tensor([0.0], dtype=torch.float32, device=device)  # 幅度为0时相位任意

    # 手动编码
    R_MIN, R_MAX = 0.0, 1.0
    TH_SCALE = 128 / math.pi

    # 编码幅度
    r1_encoded = torch.round((r1 - R_MIN) / (R_MAX - R_MIN) * 255.0 - 128.0).to(torch.int8).to(device)  # 应为 127
    r2_encoded = torch.round((r2 - R_MIN) / (R_MAX - R_MIN) * 255.0 - 128.0).to(torch.int8).to(device)  # 应为 -128

    # 编码相位
    th1_encoded = torch.round(th1 * TH_SCALE).to(torch.int8).to(device)  # 应为 0
    th2_encoded = torch.round(th2 * TH_SCALE).to(torch.int8).to(device)  # 应为 0

    print(f"r1_encoded: {r1_encoded.item()}, r2_encoded: {r2_encoded.item()}")
    print(f"th1_encoded: {th1_encoded.item()}, th2_encoded: {th2_encoded.item()}")

    # 组合极坐标向量 - 修正形状
    z0_batch = torch.stack([r1_encoded.squeeze(), th1_encoded.squeeze()], dim=0).unsqueeze(0)  # (1, 2)
    z1_batch = torch.stack([r2_encoded.squeeze(), th2_encoded.squeeze()], dim=0).unsqueeze(0)  # (1, 2)
    scale_batch = torch.ones(1, dtype=torch.float32, device=device)

    print(f"z0_batch shape: {z0_batch.shape}")
    print(f"z1_batch shape: {z1_batch.shape}")

    print(f"z0_batch: {z0_batch}")
    print(f"z1_batch: {z1_batch}")
    print(f"scale_batch: {scale_batch}")

    # 转换为复数检查
    c0 = polar_to_complex_tensor(z0_batch, scale_batch)
    c1 = polar_to_complex_tensor(z1_batch, scale_batch)
    print(f"c0 (复数): {c0}, shape: {c0.shape}")
    print(f"c1 (复数): {c1}, shape: {c1.shape}")

    # 手动计算期望的 H 门结果
    state_vec = torch.tensor([1.0, 0.0], dtype=torch.complex64, device=device)
    expected = torch.matmul(state_vec, H_MAT_TORCH)
    print(f"期望结果: {expected}")

    # 测试矩阵乘法
    test_state = torch.stack([c0, c1], dim=1)
    print(f"test_state shape: {test_state.shape}")
    print(f"H_MAT_TORCH shape: {H_MAT_TORCH.shape}")

    # 应用 Hadamard 门
    result_z0, result_z1, result_scale = hadamard_polar_tensor(z0_batch, z1_batch, scale_batch)

    print(f"result_z0: {result_z0}")
    print(f"result_z1: {result_z1}")
    print(f"result_scale: {result_scale}")

    # 转换回复数
    c0_result = polar_to_complex_tensor(result_z0.unsqueeze(0), result_scale)
    c1_result = polar_to_complex_tensor(result_z1.unsqueeze(0), result_scale)

    print(f"c0_result: {c0_result}")
    print(f"c1_result: {c1_result}")

    # 计算误差
    actual = torch.stack([c0_result.squeeze(), c1_result.squeeze()])
    error = torch.norm(actual - expected).item()
    print(f"实际结果: {actual}")
    print(f"期望结果: {expected}")
    print(f"总误差: {error}")

    # 检查编码值
    print(f"Hadamard结果编码值: r=52, th=0")
    print(f"解码计算: (52 + 128) / 255 = {180/255:.6f}")
    print(f"期望值: {1/math.sqrt(2):.6f}")
    print(f"编码误差: {abs(180/255 - 1/math.sqrt(2)):.6f}")

if __name__ == "__main__":
    debug_hadamard()